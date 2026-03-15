"""Base Scenario Generator - Comprehensive Abstract Class

This module defines the abstract base class for all scenario generators with
a unified interface for analysis, fitting, generation, and reduction.

The design follows the scikit-learn pattern with fit/generate methods and
integrates directly with the DataLoader for seamless data handling.

Classes
-------
BaseScenarioGenerator
    Abstract base class with comprehensive scenario generation interface.
AnalysisResult
    Dataclass for storing analysis results (order selection, stationarity, etc.)
GenerationResult
    Dataclass for storing generation results (scenarios, probabilities, metadata)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import pandas as pd
import numpy as np
import logging
from enum import Enum


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class StationarityResult:
    """Results from stationarity tests."""
    variable: str
    is_stationary: bool
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: Optional[float] = None
    kpss_pvalue: Optional[float] = None
    critical_values: Optional[Dict[str, float]] = None
    recommendation: str = ""


@dataclass
class OrderSelectionResult:
    """Results from order/lag selection analysis."""
    selected_order: int
    method: str  # 'aic', 'bic', 'hqic', etc.
    all_scores: Dict[int, float] = field(default_factory=dict)
    max_order_tested: int = 12
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticResult:
    """Results from model diagnostic tests."""
    test_name: str
    passed: bool
    statistic: Optional[float] = None
    pvalue: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class FitDiagnosticsReport:
    """
    Comprehensive fit diagnostics report.
    
    Provides a unified view of model fit quality with summary statistics,
    visual dashboard, and actionable recommendations.
    
    Attributes
    ----------
    model_name : str
        Name of the fitted model
    diagnostics : List[DiagnosticResult]
        Individual diagnostic test results
    residual_stats : Dict[str, Dict]
        Per-variable residual statistics (mean, std, skew, kurtosis)
    distribution_fit : Dict[str, Dict]
        Distribution fit comparison (Normal vs t vs Skew-Normal)
    model_info : Dict[str, Any]
        Model-specific information (order, parameters, etc.)
    """
    model_name: str
    diagnostics: List[DiagnosticResult]
    residual_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    distribution_fit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    recommended_distribution: str = "normal"
    recommended_params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_passed(self) -> bool:
        """Check if all diagnostic tests passed."""
        return all(d.passed for d in self.diagnostics)
    
    @property
    def n_passed(self) -> int:
        """Number of passed tests."""
        return sum(1 for d in self.diagnostics if d.passed)
    
    @property
    def n_failed(self) -> int:
        """Number of failed tests."""
        return sum(1 for d in self.diagnostics if not d.passed)
    
    @property
    def n_warnings(self) -> int:
        """Number of tests with warnings (passed but marginal)."""
        return sum(1 for d in self.diagnostics 
                   if d.passed and d.pvalue is not None and d.pvalue < 0.10)
    
    def summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all diagnostic tests.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: Test, Variable, Status, Statistic, P-value
        """
        rows = []
        for d in self.diagnostics:
            # Extract variable from test name if present
            variable = d.details.get('variable', 'All')
            
            # Determine status
            if d.passed:
                if d.pvalue is not None and d.pvalue < 0.10:
                    status = '⚠ Warning'
                else:
                    status = '✓ Pass'
            else:
                status = '✗ Fail'
            
            rows.append({
                'Test': d.test_name.split(' (')[0],  # Remove variable suffix
                'Variable': variable,
                'Status': status,
                'Statistic': f"{d.statistic:.4f}" if d.statistic is not None else '-',
                'P-value': f"{d.pvalue:.4f}" if d.pvalue is not None else '-',
                'Recommendation': d.recommendation or '-',
            })
        
        return pd.DataFrame(rows)
    
    def plot(self, figsize: tuple = (14, 10)) -> None:
        """
        Plot diagnostic dashboard.
        
        Creates a multi-panel figure with:
        - Residual time series
        - ACF/PACF
        - Q-Q plots
        - Distribution fits
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        residuals = self.model_info.get('residuals')
        if residuals is None:
            print("No residuals available for plotting.")
            return
        
        n_vars = len(residuals.columns)
        fig, axes = plt.subplots(n_vars, 4, figsize=figsize)
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(residuals.columns):
            resid = residuals[col].dropna()
            
            # 1. Time series
            axes[i, 0].plot(residuals.index, residuals[col], linewidth=0.8, alpha=0.8)
            axes[i, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[i, 0].set_title(f'{col} - Residuals')
            axes[i, 0].set_ylabel('Residual')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 2. ACF
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(resid, ax=axes[i, 1], lags=20, alpha=0.05)
                axes[i, 1].set_title(f'{col} - ACF')
            except Exception:
                axes[i, 1].text(0.5, 0.5, 'ACF unavailable', ha='center', va='center')
            
            # 3. Q-Q Plot
            stats.probplot(resid, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'{col} - Q-Q Plot')
            axes[i, 2].grid(True, alpha=0.3)
            
            # 4. Distribution fit
            x = np.linspace(resid.min(), resid.max(), 100)
            axes[i, 3].hist(resid, bins=30, density=True, alpha=0.5, 
                           edgecolor='white', label='Empirical')
            
            # Normal
            mu, sigma = resid.mean(), resid.std()
            axes[i, 3].plot(x, stats.norm.pdf(x, mu, sigma), 
                           'k-', lw=2, label='Normal')
            
            # t-distribution
            try:
                df_t, loc_t, scale_t = stats.t.fit(resid)
                axes[i, 3].plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
                               'r--', lw=2, label=f't (df={df_t:.1f})')
            except Exception:
                pass
            
            axes[i, 3].set_title(f'{col} - Distribution')
            axes[i, 3].legend(fontsize=8)
            axes[i, 3].grid(True, alpha=0.3)
        
        # Overall title with summary
        status_icon = '✓' if self.all_passed else '⚠' if self.n_failed < 3 else '✗'
        fig.suptitle(
            f'{self.model_name} Fit Diagnostics {status_icon}  |  '
            f'Passed: {self.n_passed}/{len(self.diagnostics)}  |  '
            f'Recommended: {self.recommended_distribution}',
            fontsize=12, fontweight='bold', y=1.01
        )
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """Pretty print summary."""
        lines = [
            f"{'='*60}",
            f"  {self.model_name} FIT DIAGNOSTICS REPORT",
            f"{'='*60}",
            "",
        ]
        
        # Overall status
        if self.all_passed:
            lines.append("  Status: ✓ ALL TESTS PASSED")
        else:
            lines.append(f"  Status: ⚠ {self.n_failed} TEST(S) FAILED")
        
        lines.append(f"  Tests: {self.n_passed} passed, {self.n_warnings} warnings, {self.n_failed} failed")
        lines.append("")
        
        # Model info
        if self.model_info:
            lines.append("  Model Info:")
            for key, val in self.model_info.items():
                if key != 'residuals':
                    lines.append(f"    {key}: {val}")
            lines.append("")
        
        # Test results by category
        categories = {}
        for d in self.diagnostics:
            cat = d.test_name.split(' (')[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(d)
        
        lines.append("  Diagnostic Tests:")
        for cat, tests in categories.items():
            passed = sum(1 for t in tests if t.passed)
            icon = '✓' if passed == len(tests) else '⚠' if passed > 0 else '✗'
            lines.append(f"    {icon} {cat}: {passed}/{len(tests)}")
        lines.append("")
        
        # Per-variable residual analysis
        if self.residual_stats:
            lines.append("  Residual Statistics (per variable):")
            lines.append("  " + "-"*56)
            lines.append(f"  {'Variable':<8} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10}")
            lines.append("  " + "-"*56)
            
            for var, stats in self.residual_stats.items():
                skew = stats.get('skewness', 0)
                kurt = stats.get('kurtosis', 0)
                
                # Interpret skewness
                if abs(skew) < 0.5:
                    skew_flag = ""
                elif abs(skew) < 1.0:
                    skew_flag = "~"
                else:
                    skew_flag = "!"
                
                # Interpret kurtosis (excess kurtosis, normal=0)
                if kurt < 1.0:
                    kurt_flag = ""
                elif kurt < 3.0:
                    kurt_flag = "~"  # moderate tails
                else:
                    kurt_flag = "!"  # heavy tails
                
                lines.append(
                    f"  {var:<8} {stats.get('mean', 0):>10.6f} {stats.get('std', 0):>10.6f} "
                    f"{skew:>9.2f}{skew_flag} {kurt:>9.2f}{kurt_flag}"
                )
            
            lines.append("  " + "-"*56)
            lines.append("  Legend: ~ moderate departure, ! significant departure")
            lines.append("")
        
        # Distribution fit comparison
        if self.distribution_fit:
            lines.append("  Distribution Fit Comparison (by BIC):")
            lines.append("  " + "-"*56)
            
            for var, fits in self.distribution_fit.items():
                # Find best distribution by BIC
                sorted_fits = sorted(fits.items(), key=lambda x: x[1]['bic'])
                best_name, best_fit = sorted_fits[0]
                
                # Get t-distribution df if available
                t_info = ""
                if 'Student-t' in fits:
                    t_df = fits['Student-t']['params'].get('df', 0)
                    t_info = f"(t df={t_df:.1f})"
                
                lines.append(f"  {var}: Best = {best_name} {t_info}")
                
                # Show BIC comparison
                bic_str = "       "
                for dist_name, fit_data in sorted_fits:
                    bic_str += f"{dist_name[:6]}: {fit_data['bic']:.1f}  "
                lines.append(bic_str)
            
            lines.append("")
        
        # Interpretation section
        lines.append("  Interpretation:")
        lines.append("  " + "-"*56)
        
        # Check normality
        jb_tests = [d for d in self.diagnostics if 'Jarque-Bera' in d.test_name]
        jb_failed = sum(1 for t in jb_tests if not t.passed)
        if jb_failed > 0:
            lines.append(f"  ⚠ Normality: {jb_failed}/{len(jb_tests)} variables fail Jarque-Bera test")
            lines.append("    → Residuals have non-normal distribution (fat tails/skewness)")
            lines.append("    → Use Student-t distribution for more realistic scenarios")
        else:
            lines.append("  ✓ Normality: All variables pass Jarque-Bera test")
        
        # Check autocorrelation
        lb_tests = [d for d in self.diagnostics if 'Ljung-Box' in d.test_name]
        lb_failed = sum(1 for t in lb_tests if not t.passed)
        if lb_failed > 0:
            lines.append(f"  ⚠ Autocorrelation: {lb_failed}/{len(lb_tests)} variables show residual correlation")
            lines.append("    → Consider higher VAR order to capture remaining dynamics")
        else:
            lines.append("  ✓ Autocorrelation: No significant serial correlation in residuals")
            lines.append("    → VAR model captures temporal dynamics well")
        
        # Check stability
        stability_tests = [d for d in self.diagnostics if 'Stability' in d.test_name]
        if stability_tests and all(t.passed for t in stability_tests):
            lines.append("  ✓ Stability: VAR system is stable (eigenvalues inside unit circle)")
        elif stability_tests:
            lines.append("  ✗ Stability: VAR system may be unstable!")
        
        lines.append("")
        
        # Recommended distribution
        lines.append(f"  Recommended Distribution: {self.recommended_distribution}")
        if self.recommended_params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.recommended_params.items())
            lines.append(f"  Parameters: {params_str}")
        
        # Final verdict
        lines.append("")
        lines.append("  Verdict:")
        if self.all_passed:
            lines.append("  ✓ Model is well-specified. Standard Normal shocks are appropriate.")
        elif self.n_failed <= 3 and jb_failed > 0 and lb_failed == 0:
            lines.append("  ✓ Model dynamics are good. Use Student-t shocks for fat tails.")
        elif lb_failed > 0:
            lines.append("  ⚠ Consider increasing VAR order or reviewing data preprocessing.")
        else:
            lines.append("  ⚠ Review model specification and data quality.")
        
        lines.append("")
        lines.append("  Use .summary() for detailed DataFrame")
        lines.append("  Use .plot() for visual dashboard")
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)


@dataclass
class AnalysisResult:
    """Comprehensive analysis results before fitting."""
    stationarity: List[StationarityResult]
    order_selection: OrderSelectionResult
    diagnostics: List[DiagnosticResult] = field(default_factory=list)
    recommended_order: int = 1
    warnings: List[str] = field(default_factory=list)
    
    @property
    def all_stationary(self) -> bool:
        """Check if all variables are stationary."""
        return all(s.is_stationary for s in self.stationarity)
    
    @property
    def all_diagnostics_passed(self) -> bool:
        """Check if all diagnostic tests passed."""
        return all(d.passed for d in self.diagnostics)


@dataclass
class GenerationResult:
    """Results from scenario generation."""
    scenarios: pd.DataFrame  # Long-form: Date, Scenario, D, P, C
    probabilities: pd.Series  # Indexed by scenario ID
    n_scenarios: int
    horizon: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate generation results."""
        if not np.isclose(self.probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got {self.probabilities.sum()}")


@dataclass
class ReductionResult:
    """Results from scenario reduction."""
    scenarios: pd.DataFrame  # Reduced scenarios
    probabilities: pd.Series  # Updated probabilities
    original_n_scenarios: int
    reduced_n_scenarios: int
    method: str  # 'kmedoids', 'kmeans', etc.
    stress_scenarios_included: int = 0
    cluster_assignments: Optional[pd.Series] = None
    reduction_ratio: float = 0.0
    
    def __post_init__(self):
        self.reduction_ratio = 1 - (self.reduced_n_scenarios / self.original_n_scenarios)


@dataclass
class StressConfig:
    """
    Configuration for stress scenario preservation during reduction.
    
    Controls which extreme scenarios are guaranteed inclusion in the
    reduced set. Fully generic — makes ZERO assumptions about variable
    names or domain semantics.
    
    Three modes of operation (mutually exclusive):
    
    1. **Variable-specific** (``variable_stress``): identify extreme
       scenarios per variable independently. Use when you care about
       tails of specific variables.
       
    2. **Composite** (``composite_weights``): compute a single weighted
       score across variables and pick scenarios from the tails of that
       distribution. Use when you have a domain-meaningful aggregate
       (e.g., profit proxy = +price −cost).
       
    3. **Force-include** (``force_include``): manually specify scenario
       IDs that must survive reduction.
    
    You may combine ``force_include`` with either of the other two modes.
    ``variable_stress`` and ``composite_weights`` are mutually exclusive.
    
    Parameters
    ----------
    pct : float, default 0.05
        Fraction of the *reduced* set reserved for stress scenarios.
        n_stress = ceil(n_clusters * pct).
    variable_stress : dict, optional
        Per-variable tail selection.
        Key = column name, value = 'upper', 'lower', or 'both'.
        Example: ``{'P': 'lower', 'C': 'upper'}``
    composite_weights : dict, optional
        Weighted composite score.
        Positive weight → higher is *better* (e.g., revenue).
        Negative weight → higher is *worse* (e.g., cost).
        Example: ``{'P': 1.0, 'C': -1.0}``
    composite_direction : str, default 'both'
        Which tail of the composite score to protect:
        'lower' (worst), 'upper' (best), or 'both'.
    force_include : list, optional
        Scenario IDs to always include regardless of clustering.
    
    Examples
    --------
    No stress (pure clustering):
    
    >>> stress = None  # pass to reduce()
    
    Protect low-price and high-cost tails:
    
    >>> stress = StressConfig(
    ...     pct=0.05,
    ...     variable_stress={'P': 'lower', 'C': 'upper'},
    ... )
    
    Protect worst composite outcomes (low price, high cost):
    
    >>> stress = StressConfig(
    ...     pct=0.05,
    ...     composite_weights={'P': 1.0, 'C': -1.0},
    ...     composite_direction='lower',
    ... )
    
    Protect both tails of all variables:
    
    >>> stress = StressConfig(
    ...     pct=0.10,
    ...     variable_stress={'D': 'both', 'P': 'both', 'C': 'both'},
    ... )
    """
    pct: float = 0.05
    variable_stress: Optional[Dict[str, str]] = None
    composite_weights: Optional[Dict[str, float]] = None
    composite_direction: str = 'both'
    force_include: Optional[List] = None
    
    def __post_init__(self):
        # Validate pct
        if not 0.0 <= self.pct <= 1.0:
            raise ValueError(f"pct must be in [0, 1], got {self.pct}")
        
        # Mutual exclusivity
        if self.variable_stress and self.composite_weights:
            raise ValueError(
                "variable_stress and composite_weights are mutually exclusive. "
                "Use one or the other."
            )
        
        # Validate variable_stress directions
        valid_directions = {'upper', 'lower', 'both'}
        if self.variable_stress:
            for var, direction in self.variable_stress.items():
                if direction not in valid_directions:
                    raise ValueError(
                        f"Invalid direction '{direction}' for variable '{var}'. "
                        f"Must be one of {valid_directions}."
                    )
        
        # Validate composite_direction
        if self.composite_direction not in valid_directions:
            raise ValueError(
                f"composite_direction must be one of {valid_directions}, "
                f"got '{self.composite_direction}'"
            )
        
        # Default force_include to empty list
        if self.force_include is None:
            self.force_include = []


# =============================================================================
# Generator Configuration
# =============================================================================

class ShockDistribution(Enum):
    """Available shock distributions for scenario generation."""
    NORMAL = "normal"
    T = "t"
    SKEWED_T = "skewed_t"
    LAPLACE = "laplace"
    UNIFORM = "uniform"
    EMPIRICAL = "empirical"  # Bootstrap from residuals


@dataclass
class GeneratorConfig:
    """Configuration for scenario generator."""
    # Shock distribution
    shock_distribution: ShockDistribution = ShockDistribution.NORMAL
    distribution_params: Dict[str, Any] = field(default_factory=dict)
    
    # Order selection
    max_order: int = 12
    order_selection_method: str = 'bic'  # 'aic', 'bic', 'hqic'
    
    # Generation
    default_seed: int = 42
    
    def to_dict(self) -> Dict:
        return {
            'shock_distribution': self.shock_distribution.value,
            'distribution_params': self.distribution_params,
            'max_order': self.max_order,
            'order_selection_method': self.order_selection_method,
        }

# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseScenarioGenerator(ABC):
    """
    Abstract base class for scenario generators.
    
    This class provides a unified interface for:
    1. **Analysis**: Order selection, stationarity tests, diagnostics
    2. **Fitting**: Learn model parameters from historical data
    3. **Generation**: Create future scenarios with uncertainty
    4. **Reduction**: Reduce scenarios via clustering (K-medoids)
    5. **Visualization**: Plot scenarios, distributions, diagnostics
    
    The generator integrates with DataLoader for seamless data handling.
    
    Workflow
    --------
    >>> from data.loader import DataLoader
    >>> from scenario import VARScenarioGenerator
    >>> 
    >>> # Load and preprocess data
    >>> loader = DataLoader(fred_api_key="your_key")
    >>> loader.load_from_fred().subset(n_observations=180)
    >>> 
    >>> # Create generator with config
    >>> config = GeneratorConfig(shock_distribution=ShockDistribution.T)
    >>> generator = VARScenarioGenerator(config=config)
    >>> 
    >>> # Analyze, fit, generate
    >>> analysis = generator.analyze(loader)
    >>> generator.fit(loader, order=analysis.recommended_order)
    >>> result = generator.generate(n_scenarios=1000, horizon=12)
    >>> 
    >>> # Reduce scenarios
    >>> reduced = generator.reduce(result, n_clusters=50)
    
    Parameters
    ----------
    config : GeneratorConfig, optional
        Configuration for the generator.
    variable_names : List[str], optional
        Variable names to generate. Default: ['D', 'P', 'C']
    
    Attributes
    ----------
    config : GeneratorConfig
        Generator configuration
    is_fitted : bool
        Whether the model has been fitted
    analysis_result : AnalysisResult or None
        Results from analyze() (stored for reference)
    fit_result : dict or None
        Model-specific fit results
    """
    
    # Class-level logging configuration
    _log_level: str = "INFO"
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        variable_names: Optional[List[str]] = None,
        variable_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the scenario generator.
        
        Parameters
        ----------
        config : GeneratorConfig, optional
            Configuration. Uses defaults if not provided.
        variable_names : List[str], optional
            Variables to forecast. Default: ['D', 'P', 'C']
        variable_labels : dict, optional
            Human-readable axis labels keyed by variable name.
            E.g. {'D': 'Demand (Tn)', 'P': 'Price (€/Tn)', 'C': 'Scrap Cost (€/Tn)'}
        """
        self.config = config or GeneratorConfig()
        self._variable_names = variable_names or ['D', 'P', 'C']
        self._variable_labels: Dict[str, str] = variable_labels or {}
        
        # State
        self._is_fitted: bool = False
        self._data_loader: Optional[Any] = None  # DataLoader reference
        self._historical_data: Optional[pd.DataFrame] = None
        self._log_returns: Optional[pd.DataFrame] = None
        
        # Results storage
        self._analysis_result: Optional[AnalysisResult] = None
        self._fit_result: Optional[Dict[str, Any]] = None
        self._last_generation: Optional[GenerationResult] = None
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted
    
    @property
    def variable_names(self) -> List[str]:
        """Variable names being forecasted."""
        return self._variable_names
    
    @property
    def historical_data(self) -> Optional[pd.DataFrame]:
        """Historical data (levels) used for fitting."""
        return self._historical_data
    
    @property
    def log_returns(self) -> Optional[pd.DataFrame]:
        """Log returns of historical data."""
        return self._log_returns
    
    @property
    def analysis_result(self) -> Optional[AnalysisResult]:
        """Results from the last analyze() call."""
        return self._analysis_result
    
    @property
    def fit_result(self) -> Optional[Dict[str, Any]]:
        """Model-specific fit results."""
        return self._fit_result
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/type of the scenario generation model."""
        pass
    
    @property
    @abstractmethod
    def supports_irf(self) -> bool:
        """Whether this model supports Impulse Response Function analysis."""
        pass
    
    @property
    @abstractmethod
    def requires_stationarity(self) -> bool:
        """Whether this model requires stationary data."""
        pass
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def _select_order(
        self,
        data: pd.DataFrame,
        max_order: int,
        method: str,
    ) -> OrderSelectionResult:
        """
        Model-specific order/lag selection.
        
        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed data (e.g., log returns)
        max_order : int
            Maximum order to test
        method : str
            Selection criterion ('aic', 'bic', 'hqic')
            
        Returns
        -------
        result : OrderSelectionResult
        """
        pass
    
    @abstractmethod
    def _fit_model(
        self,
        data: pd.DataFrame,
        order: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Model-specific fitting logic.
        
        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed data
        order : int
            Model order/lag
        **kwargs
            Additional model-specific parameters
            
        Returns
        -------
        fit_result : dict
            Model-specific results (coefficients, residuals, etc.)
        """
        pass
    
    @abstractmethod
    def _generate_scenarios(
        self,
        n_scenarios: int,
        horizon: int,
        start_date: pd.Timestamp,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Model-specific scenario generation.
        
        Parameters
        ----------
        n_scenarios : int
            Number of scenarios
        horizon : int
            Forecast horizon (periods)
        start_date : pd.Timestamp
            Start date for scenarios
        seed : int, optional
            Random seed
            
        Returns
        -------
        scenarios : pd.DataFrame
            Long-form scenarios
        probabilities : pd.Series
            Scenario probabilities
        """
        pass
    
    @abstractmethod
    def _get_model_diagnostics(self) -> List[DiagnosticResult]:
        """
        Get model-specific diagnostic tests.
        
        Returns
        -------
        diagnostics : List[DiagnosticResult]
        """
        pass
    
    # =========================================================================
    # Concrete Methods - Shared implementations
    # =========================================================================
    
    def analyze(
        self,
        data_loader: "DataLoader",
        max_order: Optional[int] = None,
        method: Optional[str] = None,
        run_stationarity: bool = True,
        plot_results: bool = True,
    ) -> AnalysisResult:
        """
        Perform pre-fitting analysis: order selection, stationarity tests.
        
        This method helps you understand the data before fitting:
        - Tests stationarity of each variable
        - Selects optimal model order using information criteria
        - Provides recommendations and warnings
        
        Parameters
        ----------
        data_loader : DataLoader
            DataLoader instance with loaded and preprocessed data
        max_order : int, optional
            Maximum order to test. Default from config.
        method : str, optional
            Order selection method. Default from config.
        run_stationarity : bool, default True
            Whether to run stationarity tests
        plot_results : bool, default True
            Whether to plot analysis results
            
        Returns
        -------
        result : AnalysisResult
            Comprehensive analysis results
        """
        logger = self._setup_logger()
        logger.info(f"[{self.model_name}] Running pre-fit analysis...")
        
        # Get preprocessed data from loader
        if data_loader.log_returns_data is None:
            data_loader.compute_log_returns()
        
        log_returns = data_loader.log_returns_data
        self._data_loader = data_loader
        self._historical_data = data_loader.data.copy()
        self._log_returns = log_returns.copy()
        
        max_order = max_order or self.config.max_order
        method = method or self.config.order_selection_method
        
        warnings = []
        
        # 1. Stationarity tests
        stationarity_results = []
        if run_stationarity:
            logger.debug("  Running stationarity tests...")
            stationarity_results = self._test_stationarity(log_returns)
            
            if self.requires_stationarity and not all(s.is_stationary for s in stationarity_results):
                non_stationary = [s.variable for s in stationarity_results if not s.is_stationary]
                warnings.append(
                    f"Variables {non_stationary} may not be stationary. "
                    "Consider differencing or using log returns."
                )
        
        # 2. Order selection
        logger.debug(f"  Selecting order (max={max_order}, method={method})...")
        order_result = self._select_order(log_returns, max_order, method)
        
        # 3. Create result
        result = AnalysisResult(
            stationarity=stationarity_results,
            order_selection=order_result,
            recommended_order=order_result.selected_order,
            warnings=warnings,
        )
        
        self._analysis_result = result
        
        # 4. Logging & plotting
        logger.info(f"[✓] Analysis complete. Recommended order: {result.recommended_order}")
        if result.warnings:
            for w in result.warnings:
                logger.warning(f"  ⚠ {w}")
        
        if plot_results:
            self._plot_analysis(result, log_returns)
        
        return result
    
    def fit(
        self,
        data_loader: "DataLoader",
        order: Optional[int] = None,
        auto_select: bool = True,
        **kwargs
    ) -> "BaseScenarioGenerator":
        """
        Fit the scenario generator to historical data.
        
        Parameters
        ----------
        data_loader : DataLoader
            DataLoader instance with loaded data
        order : int, optional
            Model order/lag. If None and auto_select=True, uses analysis result.
        auto_select : bool, default True
            If order is None, automatically run analysis and select order.
        **kwargs
            Additional model-specific fitting parameters
            
        Returns
        -------
        self : BaseScenarioGenerator
            Returns self for method chaining
        """
        logger = self._setup_logger()
        
        # Get data from loader
        if data_loader.log_returns_data is None:
            data_loader.compute_log_returns()
        
        self._data_loader = data_loader
        self._historical_data = data_loader.data.copy()
        self._log_returns = data_loader.log_returns_data.copy()
        
        # Determine order
        if order is None:
            if auto_select:
                if self._analysis_result is None:
                    logger.info("Running auto order selection...")
                    self.analyze(data_loader, plot_results=False)
                order = self._analysis_result.recommended_order
            else:
                order = 1  # Default fallback
        
        logger.info(f"[{self.model_name}] Fitting with order={order}...")
        
        # Call model-specific fitting
        self._fit_result = self._fit_model(self._log_returns, order, **kwargs)
        self._is_fitted = True
        
        # Get diagnostics
        diagnostics = self._get_model_diagnostics()
        if self._analysis_result is not None:
            self._analysis_result.diagnostics = diagnostics
        
        logger.info(f"[✓] Model fitted successfully!")
        
        return self
    
    def fit_diagnostics(
        self,
        plot: bool = False,
        detailed: bool = False,
    ) -> FitDiagnosticsReport:
        """
        Get comprehensive fit diagnostics report.
        
        Provides a unified view of model fit quality including:
        - Diagnostic test results (stability, autocorrelation, normality)
        - Residual statistics (mean, std, skewness, kurtosis)
        - Distribution fit comparison (Normal vs t vs Skew-Normal)
        - Recommendations for shock distribution
        
        Parameters
        ----------
        plot : bool, default False
            If True, display diagnostic dashboard
        detailed : bool, default False
            If True, print detailed summary (otherwise returns silently)
            
        Returns
        -------
        report : FitDiagnosticsReport
            Comprehensive diagnostics report with .summary(), .plot() methods
            
        Raises
        ------
        RuntimeError
            If model hasn't been fitted
            
        Examples
        --------
        >>> generator.fit(loader, order=2)
        >>> report = generator.fit_diagnostics()
        >>> print(report)  # Formatted summary
        >>> report.summary()  # DataFrame
        >>> report.plot()  # Visual dashboard
        """
        self._check_fitted()
        
        from scipy import stats as sp_stats
        
        # Get model diagnostics
        diagnostics = self._get_model_diagnostics()
        
        # Get residuals (from fit result)
        residuals = self._fit_result.get('residuals') if self._fit_result else None
        
        # Compute residual statistics
        residual_stats = {}
        distribution_fit = {}
        
        if residuals is not None:
            for col in residuals.columns:
                r = residuals[col].dropna()
                residual_stats[col] = {
                    'mean': float(r.mean()),
                    'std': float(r.std()),
                    'skewness': float(r.skew()),
                    'kurtosis': float(r.kurtosis()),
                    'n': len(r),
                }
                
                # Fit distributions and compute AIC/BIC
                n = len(r)
                
                # Normal
                mu, sigma = r.mean(), r.std()
                ll_norm = np.sum(sp_stats.norm.logpdf(r, mu, sigma))
                aic_norm = 4 - 2 * ll_norm  # k=2
                bic_norm = 2 * np.log(n) - 2 * ll_norm
                
                dist_fits = {
                    'Normal': {'aic': aic_norm, 'bic': bic_norm, 'params': {'mu': mu, 'sigma': sigma}},
                }
                
                # Student-t
                try:
                    df_t, loc_t, scale_t = sp_stats.t.fit(r)
                    ll_t = np.sum(sp_stats.t.logpdf(r, df_t, loc_t, scale_t))
                    aic_t = 6 - 2 * ll_t  # k=3
                    bic_t = 3 * np.log(n) - 2 * ll_t
                    dist_fits['Student-t'] = {'aic': aic_t, 'bic': bic_t, 'params': {'df': df_t, 'loc': loc_t, 'scale': scale_t}}
                except Exception:
                    pass
                
                # Skew-Normal
                try:
                    a_sn, loc_sn, scale_sn = sp_stats.skewnorm.fit(r)
                    ll_sn = np.sum(sp_stats.skewnorm.logpdf(r, a_sn, loc_sn, scale_sn))
                    aic_sn = 6 - 2 * ll_sn  # k=3
                    bic_sn = 3 * np.log(n) - 2 * ll_sn
                    dist_fits['Skew-Normal'] = {'aic': aic_sn, 'bic': bic_sn, 'params': {'alpha': a_sn, 'loc': loc_sn, 'scale': scale_sn}}
                except Exception:
                    pass
                
                distribution_fit[col] = dist_fits
        
        # Determine recommended distribution (majority vote by BIC)
        dist_votes = []
        dist_params_all = []
        
        for col, fits in distribution_fit.items():
            best = min(fits.items(), key=lambda x: x[1]['bic'])
            dist_votes.append(best[0])
            if 'df' in best[1]['params']:
                dist_params_all.append(best[1]['params']['df'])
        
        if dist_votes:
            from collections import Counter
            vote_counts = Counter(dist_votes)
            recommended_dist = vote_counts.most_common(1)[0][0]
        else:
            recommended_dist = 'Normal'
        
        # Compute recommended params
        recommended_params = {}
        if recommended_dist == 'Student-t' and dist_params_all:
            recommended_params['df'] = round(np.mean(dist_params_all))
        
        # Model info
        model_info = {
            'order': self._fit_result.get('order') if self._fit_result else None,
            'n_observations': len(self._log_returns) if self._log_returns is not None else 0,
            'variables': self._variable_names,
            'residuals': residuals,
        }
        
        # Create report
        report = FitDiagnosticsReport(
            model_name=self.model_name,
            diagnostics=diagnostics,
            residual_stats=residual_stats,
            distribution_fit=distribution_fit,
            model_info=model_info,
            recommended_distribution=recommended_dist,
            recommended_params=recommended_params,
        )
        
        if detailed:
            print(report)
        
        if plot:
            report.plot()
        
        return report
    
    def generate(
        self,
        n_scenarios: int = 3000,
        horizon: int = 12,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        seed: Optional[int] = None,
        convert_to_levels: bool = True,
    ) -> GenerationResult:
        """
        Generate future scenarios.
        
        Parameters
        ----------
        n_scenarios : int, default 1000
            Number of scenarios to generate
        horizon : int, default 12
            Forecast horizon in periods (months)
        start_date : str or Timestamp, optional
            Start date for scenarios. Default: day after last historical date.
        seed : int, optional
            Random seed for reproducibility. Default from config.
        convert_to_levels : bool, default True
            If True, convert log returns to levels using historical anchor.
            
        Returns
        -------
        result : GenerationResult
            Generation results with scenarios and probabilities
            
        Raises
        ------
        RuntimeError
            If model hasn't been fitted
        """
        self._check_fitted()
        logger = self._setup_logger()
        
        seed = seed or self.config.default_seed
        
        # Determine start date
        if start_date is None:
            last_date = self._historical_data.index.max()
            start_date = last_date + pd.DateOffset(months=1)
        else:
            start_date = pd.to_datetime(start_date)
        
        logger.info(
            f"[{self.model_name}] Generating {n_scenarios} scenarios, "
            f"horizon={horizon}, start={start_date.strftime('%Y-%m')}"
        )
        
        # Generate scenarios (model-specific)
        scenarios, probabilities = self._generate_scenarios(
            n_scenarios=n_scenarios,
            horizon=horizon,
            start_date=start_date,
            seed=seed,
        )
        
        # Convert to levels if requested
        if convert_to_levels:
            scenarios = self._convert_to_levels(scenarios, start_date)
        
        # Create result
        end_date = start_date + pd.DateOffset(months=horizon - 1)
        result = GenerationResult(
            scenarios=scenarios,
            probabilities=probabilities,
            n_scenarios=n_scenarios,
            horizon=horizon,
            start_date=start_date,
            end_date=end_date,
            metadata={
                'model': self.model_name,
                'seed': seed,
                'convert_to_levels': convert_to_levels,
            }
        )
        
        self._last_generation = result
        logger.info(f"[✓] Generated {n_scenarios} scenarios successfully!")
        
        return result
    
    def reduce(
        self,
        generation_result: GenerationResult,
        n_clusters: int = 50,
        method: str = 'kmedoids',
        stress: Optional["StressConfig"] = None,
        seed: Optional[int] = None,
        max_iter: int = 100,
        init_method: str = 'k-medoids++',
    ) -> ReductionResult:
        """
        Reduce scenarios using clustering with optional stress preservation.
        
        Parameters
        ----------
        generation_result : GenerationResult
            Result from generate()
        n_clusters : int, default 50
            Target number of representative scenarios
        method : str, default 'kmedoids'
            Clustering method: 'kmedoids'
        stress : StressConfig, optional
            Stress scenario configuration. If None, pure clustering
            with no stress scenario preservation.
        seed : int, optional
            Random seed for reproducibility
        max_iter : int, default 100
            Maximum iterations for clustering algorithm
        init_method : str, default 'k-medoids++'
            K-Medoids initialization method
            
        Returns
        -------
        result : ReductionResult
            Reduced scenarios with updated probabilities
            
        Examples
        --------
        Pure clustering (no stress):
        
        >>> reduced = generator.reduce(result, n_clusters=50)
        
        With variable-specific stress:
        
        >>> stress = StressConfig(
        ...     pct=0.05,
        ...     variable_stress={'P': 'lower', 'C': 'upper'},
        ... )
        >>> reduced = generator.reduce(result, n_clusters=50, stress=stress)
        
        With composite stress:
        
        >>> stress = StressConfig(
        ...     pct=0.05,
        ...     composite_weights={'P': 1.0, 'C': -1.0},
        ...     composite_direction='lower',
        ... )
        >>> reduced = generator.reduce(result, n_clusters=50, stress=stress)
        """
        logger = self._setup_logger()
        seed = seed or self.config.default_seed
        
        original_n = generation_result.n_scenarios
        scenarios = generation_result.scenarios.copy()
        prob = generation_result.probabilities.copy()
        
        # Validate stress config against scenario data
        if stress is not None:
            self._validate_stress(stress, scenarios)
        
        stress_label = f", stress={stress.pct:.0%}" if stress else ""
        logger.info(
            f"[{self.model_name}] Reducing {original_n} -> {n_clusters} scenarios "
            f"(method={method}{stress_label})"
        )
        
        # Perform reduction
        if method == 'kmedoids':
            reduced_scenarios, reduced_prob, assignments, stress_count = \
                self._reduce_kmedoids(
                    scenarios, prob, n_clusters, stress, seed, max_iter, init_method
                )
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        result = ReductionResult(
            scenarios=reduced_scenarios,
            probabilities=reduced_prob,
            original_n_scenarios=original_n,
            reduced_n_scenarios=len(reduced_prob),
            method=method,
            stress_scenarios_included=stress_count,
            cluster_assignments=assignments,
        )
        
        logger.info(
            f"[OK] Reduced to {result.reduced_n_scenarios} scenarios "
            f"({result.reduction_ratio:.0%} reduction, {stress_count} stress scenarios)"
        )
        
        return result
    
    # =========================================================================
    # Analysis & Visualization Methods
    # =========================================================================
    
    def compute_irf(
        self,
        periods: int = 24,
        orthogonalized: bool = True,
        plot: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute Impulse Response Functions (if supported by model).
        
        Parameters
        ----------
        periods : int, default 24
            Number of periods for IRF
        orthogonalized : bool, default True
            Use orthogonalized shocks
        plot : bool, default True
            Whether to plot IRF
            
        Returns
        -------
        irf_result : dict or None
            IRF results, or None if not supported
        """
        if not self.supports_irf:
            logger = self._setup_logger()
            logger.warning(f"{self.model_name} does not support IRF analysis.")
            return None
        
        self._check_fitted()
        return self._compute_irf_impl(periods, orthogonalized, plot)
    
    def plot_scenarios(
        self,
        result: Union[GenerationResult, ReductionResult],
        show_quantiles: bool = True,
        figsize: Tuple[int, int] = (14, 10),
        actual_data: Optional[pd.DataFrame] = None,
        history_periods: Optional[int] = None,
        variable_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Plot generated scenarios with historical context.
        
        Parameters
        ----------
        result : GenerationResult or ReductionResult
            Generation or reduction result
        show_quantiles : bool, default True
            Show probability-weighted quantile bands
        figsize : tuple
            Figure size
        actual_data : pd.DataFrame, optional
            Realized values to overlay (e.g. from DataLoader.get_future_data()).
            Index should be DatetimeIndex, columns should match variable names.
        history_periods : int, optional
            Number of historical data points to show (from the last date backwards).
            If None, shows all historical data.
        variable_labels : dict, optional
            Override axis labels per variable. Falls back to labels set at init.
        """
        labels = {**self._variable_labels, **(variable_labels or {})}
        self._plot_scenarios_impl(result, show_quantiles, figsize, actual_data, history_periods, labels)
    
    # =========================================================================
    # Utility Methods (Concrete)
    # =========================================================================
    
    def _test_stationarity(self, data: pd.DataFrame) -> List[StationarityResult]:
        """Run stationarity tests (ADF, KPSS) on each variable."""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        results = []
        for col in self._variable_names:
            if col not in data.columns:
                continue
            
            series = data[col].dropna()
            
            # ADF test (null: non-stationary)
            adf_result = adfuller(series, autolag='AIC')
            adf_stat, adf_pval, _, _, crit_vals, _ = adf_result
            
            # KPSS test (null: stationary)
            try:
                kpss_stat, kpss_pval, _, _ = kpss(series, regression='c', nlags='auto')
            except Exception:
                kpss_stat, kpss_pval = None, None
            
            # Determine stationarity (ADF p < 0.05 AND KPSS p > 0.05)
            is_stationary = adf_pval < 0.05
            if kpss_pval is not None:
                is_stationary = is_stationary and kpss_pval > 0.05
            
            recommendation = "Stationary" if is_stationary else "Consider differencing"
            
            results.append(StationarityResult(
                variable=col,
                is_stationary=is_stationary,
                adf_statistic=adf_stat,
                adf_pvalue=adf_pval,
                kpss_statistic=kpss_stat,
                kpss_pvalue=kpss_pval,
                critical_values=crit_vals,
                recommendation=recommendation,
            ))
        
        return results
    
    def _convert_to_levels(
        self,
        returns_scenarios: pd.DataFrame,
        start_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Convert log returns scenarios to price levels."""
        # Get anchor values from historical data
        anchor_date = self._historical_data.index.max()
        anchor_values = self._historical_data.loc[anchor_date].to_dict()
        
        scenarios = returns_scenarios.copy()
        scenario_ids = scenarios['Scenario'].unique()
        
        # For each scenario, reconstruct levels
        reconstructed = []
        for sid in scenario_ids:
            mask = scenarios['Scenario'] == sid
            scenario_data = scenarios[mask].sort_values('Date')
            
            for var in self._variable_names:
                if var not in scenario_data.columns:
                    continue
                
                # Cumulative sum of log returns, then exp
                returns = scenario_data[var].values
                cumsum = np.cumsum(returns)
                levels = anchor_values[var] * np.exp(cumsum)
                scenarios.loc[mask, var] = levels
        
        return scenarios
    
    def _validate_stress(
        self,
        stress: "StressConfig",
        scenarios: pd.DataFrame,
    ) -> None:
        """
        Validate StressConfig against scenario data.
        
        Raises ValueError if referenced variables are not in the scenario
        DataFrame or if configuration is inconsistent.
        """
        scenario_cols = set(scenarios.columns) - {'Date', 'Scenario'}
        
        if stress.variable_stress:
            missing = set(stress.variable_stress.keys()) - scenario_cols
            if missing:
                raise ValueError(
                    f"variable_stress references columns not in scenarios: {missing}. "
                    f"Available: {scenario_cols}"
                )
        
        if stress.composite_weights:
            missing = set(stress.composite_weights.keys()) - scenario_cols
            if missing:
                raise ValueError(
                    f"composite_weights references columns not in scenarios: {missing}. "
                    f"Available: {scenario_cols}"
                )
        
        if stress.force_include:
            scenario_ids = set(scenarios['Scenario'].unique())
            missing = set(stress.force_include) - scenario_ids
            if missing:
                raise ValueError(
                    f"force_include references scenario IDs not in data: {missing}"
                )
    
    def _reduce_kmedoids(
        self,
        scenarios: pd.DataFrame,
        prob: pd.Series,
        n_clusters: int,
        stress: Optional["StressConfig"],
        seed: int,
        max_iter: int = 100,
        init_method: str = 'k-medoids++',
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, int]:
        """
        K-medoids scenario reduction with generic stress preservation.
        
        Three modes:
        - No stress (stress=None): pure K-medoids clustering
        - Variable-specific: identify extremes per variable independently
        - Composite: weighted score across variables, pick from tails
        
        Parameters
        ----------
        scenarios : pd.DataFrame
            Long-form with 'Date', 'Scenario', and variable columns
        prob : pd.Series
            Probability per scenario
        n_clusters : int
            Target reduced count
        stress : StressConfig or None
            Stress configuration
        seed : int
            Random seed
        max_iter : int
            Max K-medoids iterations
        init_method : str
            K-medoids initialization
            
        Returns
        -------
        reduced_scenarios : pd.DataFrame
        reduced_prob : pd.Series
        assignments : pd.Series
        n_stress : int
        """
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial.distance import pdist, squareform
        import kmedoids as _kmedoids
        
        logger = self._setup_logger()
        
        scenario_ids = scenarios['Scenario'].unique()
        n_original = len(scenario_ids)
        variable_cols = [c for c in self._variable_names if c in scenarios.columns]
        
        if n_clusters >= n_original:
            raise ValueError(
                f"n_clusters ({n_clusters}) must be < original count ({n_original})"
            )
        
        # =====================================================================
        # Build feature matrix: each row = flattened scenario trajectory
        # =====================================================================
        dates = sorted(scenarios['Date'].unique())
        
        # Pivot to wide form per variable, then concatenate
        pivots = {}
        for var in variable_cols:
            pivot = scenarios.pivot(index='Date', columns='Scenario', values=var)
            pivot = pivot.reindex(columns=scenario_ids)
            pivots[var] = pivot
        
        # Feature matrix: (n_scenarios, n_vars * n_dates)
        X = np.column_stack([
            pivots[var][scenario_ids].values.T  # (n_scenarios, n_dates)
            for var in variable_cols
        ])
        
        # Handle NaN
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), col_means[np.newaxis, :], X)
        
        # Standardize so all variables contribute equally to distance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # =====================================================================
        # Identify stress scenarios
        # =====================================================================
        stress_ids: set = set()
        
        if stress is not None:
            n_stress_budget = int(np.ceil(n_clusters * stress.pct))
            
            # Always include force_include
            stress_ids.update(stress.force_include)
            
            # Compute per-scenario averages for scoring
            scenario_avg = scenarios.groupby('Scenario')[variable_cols].mean()
            scenario_avg = scenario_avg.reindex(scenario_ids)
            
            remaining_budget = max(0, n_stress_budget - len(stress_ids))
            
            if stress.variable_stress and remaining_budget > 0:
                # --- Variable-specific mode ---
                # Distribute budget across variables proportionally
                vars_requested = list(stress.variable_stress.keys())
                n_tails = sum(
                    2 if d == 'both' else 1
                    for d in stress.variable_stress.values()
                )
                per_tail = max(1, remaining_budget // n_tails)
                
                for var, direction in stress.variable_stress.items():
                    values = scenario_avg[var].values
                    sorted_idx = np.argsort(values)
                    
                    if direction in ('lower', 'both'):
                        lower_ids = [
                            scenario_ids[i] for i in sorted_idx[:per_tail]
                        ]
                        stress_ids.update(lower_ids)
                    
                    if direction in ('upper', 'both'):
                        upper_ids = [
                            scenario_ids[i] for i in sorted_idx[-per_tail:]
                        ]
                        stress_ids.update(upper_ids)
                
                logger.debug(
                    f"  Variable stress: {len(stress_ids)} scenarios "
                    f"from {vars_requested}"
                )
            
            elif stress.composite_weights and remaining_budget > 0:
                # --- Composite mode ---
                # Build weighted score: sum(w_i * normalized_i)
                scores = np.zeros(n_original)
                for var, weight in stress.composite_weights.items():
                    vals = scenario_avg[var].values
                    # Min-max normalize to [0, 1]
                    vmin, vmax = vals.min(), vals.max()
                    normalized = (vals - vmin) / (vmax - vmin + 1e-10)
                    scores += weight * normalized
                
                sorted_idx = np.argsort(scores)
                
                direction = stress.composite_direction
                if direction in ('lower', 'both'):
                    lower_ids = [
                        scenario_ids[i]
                        for i in sorted_idx[:remaining_budget]
                    ]
                    stress_ids.update(lower_ids)
                
                if direction in ('upper', 'both'):
                    upper_ids = [
                        scenario_ids[i]
                        for i in sorted_idx[-remaining_budget:]
                    ]
                    stress_ids.update(upper_ids)
                
                logger.debug(
                    f"  Composite stress ({direction}): "
                    f"{len(stress_ids)} scenarios"
                )
            
            elif not stress.variable_stress and not stress.composite_weights \
                    and remaining_budget > 0:
                # --- Fallback: distance from median (no domain knowledge) ---
                median = np.median(X_scaled, axis=0)
                distances = np.linalg.norm(X_scaled - median, axis=1)
                extreme_idx = np.argsort(distances)[-remaining_budget:]
                stress_ids.update(scenario_ids[i] for i in extreme_idx)
                
                logger.debug(
                    f"  Distance-based stress: {len(stress_ids)} scenarios"
                )
        
        n_stress = len(stress_ids)
        
        # =====================================================================
        # K-medoids clustering on non-stress scenarios
        # =====================================================================
        non_stress_mask = ~np.isin(scenario_ids, list(stress_ids))
        X_cluster = X_scaled[non_stress_mask]
        ids_cluster = scenario_ids[non_stress_mask]
        original_indices = np.where(non_stress_mask)[0]
        
        n_regular = max(1, n_clusters - n_stress)
        actual_clusters = min(n_regular, len(X_cluster))
        
        dist_matrix = squareform(pdist(X_cluster, metric='euclidean'))
        km_result = _kmedoids.fasterpam(
            dist_matrix,
            actual_clusters,
            max_iter=max_iter,
            random_state=seed,
        )
        labels = np.asarray(km_result.labels)
        medoid_indices = np.asarray(km_result.medoids)
        
        selected_regular = [ids_cluster[i] for i in medoid_indices]
        selected_all = list(stress_ids) + selected_regular
        
        # =====================================================================
        # Recalculate probabilities
        # =====================================================================
        new_probs = {}
        
        if n_stress > 0 and stress is not None:
            # Stress scenarios share stress.pct of total probability equally
            stress_weight_each = stress.pct / n_stress
            for sid in stress_ids:
                new_probs[sid] = stress_weight_each
            
            # Regular scenarios share the remaining (1 - stress.pct)
            regular_total = 1.0 - stress.pct
        else:
            regular_total = 1.0
        
        # Each medoid gets weight proportional to cluster size
        cluster_sizes = []
        for cluster_id, medoid_idx in enumerate(medoid_indices):
            size = int(np.sum(labels == cluster_id))
            cluster_sizes.append(size)
        total_regular_members = sum(cluster_sizes)
        
        for cluster_id, medoid_idx in enumerate(medoid_indices):
            medoid_sid = ids_cluster[medoid_idx]
            cluster_prob = (cluster_sizes[cluster_id] / total_regular_members) * regular_total
            new_probs[medoid_sid] = cluster_prob
        
        # Normalize to handle floating-point drift
        total = sum(new_probs.values())
        reduced_prob = pd.Series({k: v / total for k, v in new_probs.items()})
        
        # =====================================================================
        # Build output
        # =====================================================================
        reduced_scenarios = scenarios[scenarios['Scenario'].isin(selected_all)].copy()
        reduced_scenarios = reduced_scenarios.sort_values(
            ['Scenario', 'Date']
        ).reset_index(drop=True)
        
        # Assignment mapping: which medoid represents each original scenario
        assignments = pd.Series(index=scenario_ids, dtype=object)
        for cluster_id, medoid_idx in enumerate(medoid_indices):
            medoid_sid = ids_cluster[medoid_idx]
            for sid in ids_cluster[labels == cluster_id]:
                assignments[sid] = medoid_sid
        for sid in stress_ids:
            assignments[sid] = sid  # stress scenarios represent themselves
        
        return reduced_scenarios, reduced_prob, assignments, n_stress
    
    def _check_fitted(self) -> None:
        """Raise if model not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit(data_loader) first."
            )
    
    @classmethod
    def set_log_level(cls, level: str = "INFO") -> None:
        """Set logging level for all generators."""
        cls._log_level = level.upper()
    
    def _setup_logger(self) -> logging.Logger:
        """Create configured logger."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self._log_level, logging.INFO))
        logger.handlers = []
        
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self._log_level, logging.INFO))
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    # =========================================================================
    # Validation & Utility Methods (backward-compatible from old base)
    # =========================================================================
    
    def validate_data(
        self,
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Validate input DataFrame for fitting.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data to validate.
        required_columns : list, optional
            Columns to check for. Defaults to self._variable_names.
            
        Raises
        ------
        ValueError
            If data is None/empty, missing DatetimeIndex, missing columns, or has nulls.
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty.")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex.")
        
        cols = required_columns or self._variable_names
        missing_cols = [col for col in cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Data missing required columns: {missing_cols}. "
                f"Available: {list(data.columns)}"
            )
        
        if data[cols].isnull().any().any():
            null_cols = [c for c in cols if data[c].isnull().any()]
            raise ValueError(f"Data contains null values in columns: {null_cols}")
    
    def validate_scenarios(self, scenarios: pd.DataFrame) -> None:
        """
        Validate generated scenarios DataFrame.
        
        Parameters
        ----------
        scenarios : pd.DataFrame
            Scenarios to validate.
            
        Raises
        ------
        ValueError
            If required columns are missing.
        """
        required_cols = ['Date', 'Scenario'] + self._variable_names
        missing = [col for col in required_cols if col not in scenarios.columns]
        if missing:
            raise ValueError(f"Scenarios missing required columns: {missing}")
    
    def validate_probabilities(
        self,
        prob: pd.Series,
        scenarios: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Validate scenario probabilities.
        
        Parameters
        ----------
        prob : pd.Series
            Scenario probabilities.
        scenarios : pd.DataFrame, optional
            If provided, checks that all scenario IDs have probabilities.
            
        Raises
        ------
        ValueError
            If probabilities are invalid.
        """
        if not np.isclose(prob.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got {prob.sum()}")
        
        if (prob < 0).any():
            raise ValueError("Probabilities cannot be negative.")
        
        if scenarios is not None:
            scenario_ids = set(scenarios['Scenario'].unique())
            missing = scenario_ids - set(prob.index)
            if missing:
                raise ValueError(f"Missing probabilities for scenarios: {missing}")
    
    @staticmethod
    def equal_probabilities(n_scenarios: int, prefix: str = "s") -> pd.Series:
        """
        Generate equal scenario probabilities.
        
        Parameters
        ----------
        n_scenarios : int
            Number of scenarios.
        prefix : str, default "S"
            Prefix for scenario IDs.
            
        Returns
        -------
        prob : pd.Series
            Equal probabilities (1/n each).
        """
        scenario_ids = [f"{prefix}{i}" for i in range(n_scenarios)]
        return pd.Series(1.0 / n_scenarios, index=scenario_ids)
    
    def to_long_format(
        self,
        scenarios_dict: Dict[str, pd.DataFrame],
        dates: List[pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Convert scenario dictionary to long-form DataFrame.
        
        Parameters
        ----------
        scenarios_dict : Dict[str, pd.DataFrame]
            Variable name -> DataFrame (dates × scenarios).
        dates : List[pd.Timestamp]
            Dates for the scenarios.
            
        Returns
        -------
        pd.DataFrame
            Long-form with Date, Scenario, and variable columns.
        """
        first_var = list(scenarios_dict.keys())[0]
        scenario_ids = scenarios_dict[first_var].columns.tolist()
        
        records = []
        for date in dates:
            for sid in scenario_ids:
                record = {'Date': date, 'Scenario': sid}
                for var_name, df in scenarios_dict.items():
                    record[var_name] = df.loc[date, sid]
                records.append(record)
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # Placeholder methods for visualization (can be overridden)
    # =========================================================================
    
    def _plot_analysis(self, result: AnalysisResult, data: pd.DataFrame) -> None:
        """Plot analysis results. Override for custom visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Order selection
        ax = axes[0, 0]
        orders = list(result.order_selection.all_scores.keys())
        scores = list(result.order_selection.all_scores.values())
        ax.bar(orders, scores, color='steelblue', alpha=0.7)
        ax.axvline(result.order_selection.selected_order, color='red', linestyle='--', 
                   label=f'Selected: {result.order_selection.selected_order}')
        ax.set_xlabel('Order')
        ax.set_ylabel(result.order_selection.method.upper())
        ax.set_title('Order Selection')
        ax.legend()
        
        # Stationarity results
        ax = axes[0, 1]
        vars_tested = [s.variable for s in result.stationarity]
        pvalues = [s.adf_pvalue for s in result.stationarity]
        colors = ['green' if p < 0.05 else 'red' for p in pvalues]
        ax.barh(vars_tested, pvalues, color=colors, alpha=0.7)
        ax.axvline(0.05, color='black', linestyle='--', label='5% threshold')
        ax.set_xlabel('ADF p-value')
        ax.set_title('Stationarity Tests')
        ax.legend()
        
        # Time series
        ax = axes[1, 0]
        for col in data.columns[:3]:  # Plot first 3
            ax.plot(data.index, data[col], label=col, alpha=0.7)
        ax.set_title('Log Returns')
        ax.legend()
        
        # ACF of first variable
        ax = axes[1, 1]
        from statsmodels.graphics.tsaplots import plot_acf
        if len(data.columns) > 0:
            plot_acf(data[data.columns[0]].dropna(), ax=ax, lags=20)
            ax.set_title(f'ACF: {data.columns[0]}')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_scenarios_impl(
        self,
        result: Union[GenerationResult, ReductionResult],
        show_quantiles: bool,
        figsize: Tuple[int, int],
        actual_data: Optional[pd.DataFrame] = None,
        history_periods: Optional[int] = None,
        variable_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Plot scenarios implementation using Plotly."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        scenarios = result.scenarios
        prob = result.probabilities
        
        # Trim historical data if history_periods is specified
        if self._historical_data is not None and history_periods is not None:
            hist_data = self._historical_data.iloc[-history_periods:]
        else:
            hist_data = self._historical_data
        
        n_vars = len(self._variable_names)
        
        labels = variable_labels or {}
        
        # Create subplots (no titles — y-axis labels carry the variable identity)
        fig = make_subplots(
            rows=n_vars, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
        )
        
        for i, var in enumerate(self._variable_names):
            row = i + 1
            show_legend = (i == 0)  # Only show legend for first subplot
            
            # Compute statistics across scenarios
            if show_quantiles and var in scenarios.columns:
                dates = sorted(scenarios['Date'].unique())
                q05, q25, q50, q75, q95, mean_vals = [], [], [], [], [], []
                for d in dates:
                    vals = scenarios.loc[scenarios['Date'] == d, var].values
                    q05.append(np.percentile(vals, 5))
                    q25.append(np.percentile(vals, 25))
                    q50.append(np.percentile(vals, 50))
                    q75.append(np.percentile(vals, 75))
                    q95.append(np.percentile(vals, 95))
                    mean_vals.append(np.mean(vals))
                
                # 5-95% band
                fig.add_trace(
                    go.Scatter(
                        x=list(dates) + list(reversed(dates)),
                        y=q95 + list(reversed(q05)),
                        fill='toself',
                        fillcolor='rgba(70, 130, 180, 0.1)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='5-95%',
                        legendgroup='q95',
                        showlegend=show_legend,
                        hoverinfo='skip',
                    ),
                    row=row, col=1
                )
                
                # 25-75% band
                fig.add_trace(
                    go.Scatter(
                        x=list(dates) + list(reversed(dates)),
                        y=q75 + list(reversed(q25)),
                        fill='toself',
                        fillcolor='rgba(70, 130, 180, 0.25)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='25-75%',
                        legendgroup='q75',
                        showlegend=show_legend,
                        hoverinfo='skip',
                    ),
                    row=row, col=1
                )
                
                # Median line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=q50,
                        mode='lines',
                        line=dict(color='steelblue', width=2, dash='dash'),
                        name='Median',
                        legendgroup='median',
                        showlegend=show_legend,
                    ),
                    row=row, col=1
                )
                
                # Mean line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=mean_vals,
                        mode='lines',
                        line=dict(color='lightskyblue', width=2),
                        name='Mean',
                        legendgroup='mean',
                        showlegend=show_legend,
                    ),
                    row=row, col=1
                )
            
            # Plot historical
            if hist_data is not None and var in hist_data.columns:
                hist_series = hist_data[var]
                
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_series,
                        mode='lines',
                        line=dict(color='royalblue', width=2),
                        name='Historical',
                        legendgroup='historical',
                        showlegend=show_legend,
                    ),
                    row=row, col=1
                )
                
                # Add min/max annotations for historical
                hist_min_idx = hist_series.idxmin()
                hist_max_idx = hist_series.idxmax()
                
                # Historical min marker — label above to avoid subplot boundary clipping
                fig.add_trace(
                    go.Scatter(
                        x=[hist_min_idx],
                        y=[hist_series[hist_min_idx]],
                        mode='markers+text',
                        marker=dict(color='royalblue', size=8, symbol='triangle-down'),
                        text=[f'  Min: {hist_series[hist_min_idx]:,.0f}'],
                        textposition='middle right',
                        textfont=dict(size=9, color='royalblue'),
                        name='Hist Min',
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=row, col=1
                )
                
                # Historical max marker
                fig.add_trace(
                    go.Scatter(
                        x=[hist_max_idx],
                        y=[hist_series[hist_max_idx]],
                        mode='markers+text',
                        marker=dict(color='royalblue', size=8, symbol='triangle-up'),
                        text=[f'  Max: {hist_series[hist_max_idx]:,.0f}'],
                        textposition='middle right',
                        textfont=dict(size=9, color='royalblue'),
                        name='Hist Max',
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=row, col=1
                )
            
            # Overlay actual/realized values (limited to scenario date range)
            if actual_data is not None and var in actual_data.columns:
                scenario_end_date = scenarios['Date'].max()
                actual_trimmed = actual_data.loc[actual_data.index <= scenario_end_date]
                actual_series = actual_trimmed[var]
                
                fig.add_trace(
                    go.Scatter(
                        x=actual_trimmed.index,
                        y=actual_series,
                        mode='lines+markers',
                        line=dict(color='seagreen', width=2.5),
                        marker=dict(size=5, color='seagreen'),
                        name='Actual',
                        legendgroup='actual',
                        showlegend=show_legend,
                    ),
                    row=row, col=1
                )
                
                # Add min/max annotations for actual
                if len(actual_series) > 1:
                    actual_min_idx = actual_series.idxmin()
                    actual_max_idx = actual_series.idxmax()
                    
                    # Actual min marker
                    fig.add_trace(
                        go.Scatter(
                            x=[actual_min_idx],
                            y=[actual_series[actual_min_idx]],
                            mode='markers+text',
                            marker=dict(color='seagreen', size=8, symbol='triangle-down'),
                            text=[f'  Min: {actual_series[actual_min_idx]:,.0f}'],
                            textposition='middle right',
                            textfont=dict(size=9, color='seagreen'),
                            name='Actual Min',
                            showlegend=False,
                            hoverinfo='skip',
                        ),
                        row=row, col=1
                    )
                    
                    # Actual max marker
                    fig.add_trace(
                        go.Scatter(
                            x=[actual_max_idx],
                            y=[actual_series[actual_max_idx]],
                            mode='markers+text',
                            marker=dict(color='seagreen', size=8, symbol='triangle-up'),
                            text=[f'  Max: {actual_series[actual_max_idx]:,.0f}'],
                            textposition='middle right',
                            textfont=dict(size=9, color='seagreen'),
                            name='Actual Max',
                            showlegend=False,
                            hoverinfo='skip',
                        ),
                        row=row, col=1
                    )
            
            # Update y-axis label (use human-readable label if provided)
            y_label = labels.get(var, var)
            fig.update_yaxes(title_text=y_label, row=row, col=1)
        
        # Update layout
        fig.update_layout(
            height=max(figsize[1] * 80, n_vars * 280),  # taller per subplot
            width=figsize[0] * 60,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        
        # Apply consistent grid to ALL subplots
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#e0e0e0',
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#e0e0e0',
            zeroline=False,
        )
        
        fig.update_xaxes(title_text='Date', row=n_vars, col=1)
        
        fig.show()
    
    def _compute_irf_impl(self, periods: int, orthogonalized: bool, plot: bool) -> Dict[str, Any]:
        """IRF implementation. Override in subclasses that support it."""
        return {}
    
    # =========================================================================
    # Dunder methods
    # =========================================================================
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(model={self.model_name}, {status})"
    
    def summary(self) -> Dict[str, Any]:
        """Get generator summary."""
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'variable_names': self.variable_names,
            'config': self.config.to_dict(),
            'n_historical_obs': len(self._historical_data) if self._historical_data is not None else 0,
            'analysis_result': {
                'recommended_order': self._analysis_result.recommended_order if self._analysis_result else None,
                'all_stationary': self._analysis_result.all_stationary if self._analysis_result else None,
            } if self._analysis_result else None,
        }
