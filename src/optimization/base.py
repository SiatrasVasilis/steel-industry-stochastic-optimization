"""
Base Optimizer Abstract Class

This module defines the abstract interface for all optimization models.
All concrete optimizers should inherit from BaseStochasticModel and implement
the required abstract methods and properties.

Design Philosophy (Option C)
-----------------------------
Each model takes a scenario DataFrame directly in optimize(). There is no
separate prepare_inputs() layer. Instead, each model:
1. Declares what variables it needs via ``required_variables``
2. Provides default column-name aliases via ``default_variable_mapping``
3. The base class applies mapping + validation automatically

Variable Mapping (Three-Layer Priority)
---------------------------------------
When resolving column names, the priority is:
    user mapping  >  model defaults  >  no mapping

This allows scenario generators to produce columns with any naming convention
and have each optimizer automatically adapt.

Classes
-------
BaseStochasticModel
    Abstract base class for all optimization models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
import logging

if TYPE_CHECKING:
    from .results import OptimizationResult, BacktestResult


class BaseStochasticModel(ABC):
    """
    Abstract base class for steel industry optimization models.

    All optimization models should inherit from this class and implement:

    - ``model_name``: Human-readable identifier
    - ``required_variables``: List of column names the model expects
    - ``default_variable_mapping``: Dict mapping common aliases → required names
    - ``optimize()``: Core optimization logic
    - ``backtest()``: Out-of-sample evaluation

    The base class provides:

    - ``_apply_mapping()``: Resolves column names using the three-layer priority
    - ``validate_scenario_inputs()``: Checks required columns exist after mapping
    - ``run()``: Convenience wrapper (validate → optimize in one call)
    - Logging utilities (``set_log_level``, ``_setup_logger``)

    Parameters
    ----------
    solver : str, default "highs"
        Optimization solver to use ("highs", "gurobi", "cplex", "glpk")

    Examples
    --------
    >>> class MyModel(BaseStochasticModel):
    ...     model_name = "MyModel"
    ...     required_variables = ["D", "P", "C"]
    ...     default_variable_mapping = {"demand": "D", "price": "P"}
    ...
    ...     def optimize(self, scenarios, prob, **kwargs):
    ...         ...
    ...     def backtest(self, decisions, actual_data, **kwargs):
    ...         ...
    """

    # Class-level logger configuration
    _log_level: str = "INFO"
    _logger: Optional[logging.Logger] = None

    def __init__(self, solver: str = "highs"):
        """
        Initialize the optimizer.

        Parameters
        ----------
        solver : str, default "highs"
            Optimization solver to use ("highs", "gurobi", "cplex", "glpk")
        """
        self.solver = solver

    # =========================================================================
    # Abstract Properties — Must be defined by subclasses
    # =========================================================================

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name for this model (e.g. ``'SimpleStochastic'``)."""
        ...

    @property
    @abstractmethod
    def required_variables(self) -> List[str]:
        """
        Column names the model expects in the scenario DataFrame.

        These are the *internal* names used by the model's optimization logic
        (e.g. ``['D', 'P', 'C']``). The ``default_variable_mapping`` maps
        common aliases to these names.
        """
        ...

    @property
    @abstractmethod
    def default_variable_mapping(self) -> Dict[str, str]:
        """
        Default mapping from common column aliases to ``required_variables``.

        Keys are alternative column names that might appear in input data,
        values are the corresponding ``required_variables`` name.

        Example::

            {
                'demand': 'D',
                'Demand': 'D',
                'steel_price': 'P',
                'Steel_Price': 'P',
                'scrap_cost': 'C',
                'Scrap_Cost': 'C',
            }
        """
        ...

    # =========================================================================
    # Abstract Methods — Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def optimize(
        self,
        scenarios: pd.DataFrame,
        prob: pd.Series,
        **kwargs,
    ) -> "OptimizationResult":
        """
        Solve the optimization problem.

        Parameters
        ----------
        scenarios : pd.DataFrame
            Long-form DataFrame with scenario data.  Must contain columns
            ``'Date'``, ``'Scenario'``, plus all ``required_variables``
            (after mapping).
        prob : pd.Series
            Scenario probabilities indexed by scenario identifiers.
            Must sum to 1.0.
        **kwargs
            Model-specific parameters (params objects, risk profiles, etc.)

        Returns
        -------
        OptimizationResult
            Optimization output containing decisions, scenario profits,
            risk metrics, and stage-2 results.
        """
        ...

    @abstractmethod
    def backtest(
        self,
        decisions: pd.DataFrame,
        actual_data: pd.DataFrame,
        **kwargs,
    ) -> "BacktestResult":
        """
        Evaluate optimization decisions against realized (actual) data.

        Parameters
        ----------
        decisions : pd.DataFrame
            First-stage decisions from ``optimize()``.
        actual_data : pd.DataFrame
            Historical data with ``required_variables`` columns and
            DatetimeIndex.
        **kwargs
            Model-specific parameters and options.

        Returns
        -------
        BacktestResult
            Backtesting output containing timeline, metrics, and
            optional comparison to the scenario distribution.
        """
        ...

    # =========================================================================
    # Concrete Methods — Variable Mapping & Validation
    # =========================================================================

    def _apply_mapping(
        self,
        df: pd.DataFrame,
        user_mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Rename columns using the three-layer priority system.

        Priority order:
            1. ``user_mapping`` (explicit overrides from the caller)
            2. ``default_variable_mapping`` (model-defined defaults)
            3. No mapping (columns already match ``required_variables``)

        Only columns present in the DataFrame are renamed.  Columns that
        already have the correct name are left untouched.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame whose columns may need renaming.
        user_mapping : dict, optional
            Explicit ``{old_name: new_name}`` overrides.

        Returns
        -------
        df_mapped : pd.DataFrame
            Copy of *df* with columns renamed.  Original is not modified.
        """
        # Build effective mapping: model defaults first, user overrides on top
        effective_mapping: Dict[str, str] = {}
        effective_mapping.update(self.default_variable_mapping)
        if user_mapping:
            effective_mapping.update(user_mapping)

        # Only rename columns that actually exist in the DataFrame
        rename_dict = {
            old: new
            for old, new in effective_mapping.items()
            if old in df.columns and old != new
        }

        if rename_dict:
            return df.rename(columns=rename_dict)
        return df.copy()

    def validate_scenario_inputs(
        self,
        scenarios: pd.DataFrame,
        prob: pd.Series,
    ) -> None:
        """
        Validate that the scenario DataFrame and probabilities are well-formed.

        Checks (in order):
            1. ``'Date'`` and ``'Scenario'`` columns exist.
            2. All ``required_variables`` columns exist.
            3. Probabilities sum to 1.0 (tolerance 1e-6).
            4. Every scenario in the DataFrame has a matching probability.

        Parameters
        ----------
        scenarios : pd.DataFrame
            Long-form scenario data (after mapping has been applied).
        prob : pd.Series
            Scenario probabilities indexed by scenario identifiers.

        Raises
        ------
        ValueError
            On any validation failure.
        """
        # --- structural columns ---
        for col in ("Date", "Scenario"):
            if col not in scenarios.columns:
                raise ValueError(
                    f"scenarios DataFrame missing required column: '{col}'"
                )

        # --- required variables ---
        missing = [
            v for v in self.required_variables if v not in scenarios.columns
        ]
        if missing:
            available = [
                c for c in scenarios.columns if c not in ("Date", "Scenario")
            ]
            raise ValueError(
                f"[{self.model_name}] Missing required variable columns: "
                f"{missing}. Available data columns: {available}. "
                f"Consider providing a `variable_mapping` dict to rename "
                f"columns."
            )

        # --- probabilities sum to 1 ---
        psum = float(prob.sum())
        if abs(psum - 1.0) > 1e-6:
            raise ValueError(
                f"Scenario probabilities must sum to 1.0. Got {psum:.6f}"
            )

        # --- every scenario has a probability ---
        scenario_ids = set(scenarios["Scenario"].astype(str).unique())
        prob_ids = set(prob.index.astype(str))
        missing_probs = scenario_ids - prob_ids
        if missing_probs:
            raise ValueError(
                f"Missing probabilities for scenarios: "
                f"{sorted(missing_probs)[:10]}"
            )

    def run(
        self,
        scenarios: pd.DataFrame,
        prob: pd.Series,
        variable_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "OptimizationResult":
        """
        Convenience method: apply mapping → validate → optimize.

        This is the recommended entry-point when consuming scenarios directly
        from a generator, where column names may not match what the model
        expects.

        Parameters
        ----------
        scenarios : pd.DataFrame
            Long-form scenario data (any column naming convention).
        prob : pd.Series
            Scenario probabilities (must sum to 1.0).
        variable_mapping : dict, optional
            Explicit ``{old_name: new_name}`` overrides.  Merged on top of
            the model's ``default_variable_mapping``.
        **kwargs
            Forwarded to ``optimize()``.

        Returns
        -------
        OptimizationResult
            Output of ``optimize()``.
        """
        mapped = self._apply_mapping(scenarios, user_mapping=variable_mapping)
        self.validate_scenario_inputs(mapped, prob)
        return self.optimize(mapped, prob, **kwargs)

    # =========================================================================
    # Logging Utilities
    # =========================================================================

    @classmethod
    def set_log_level(cls, level: str = "INFO") -> None:
        """
        Set the logging level for all instances of this class.

        Parameters
        ----------
        level : str
            One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        """
        cls._log_level = level.upper()
        cls._logger = None
        print(f"Log level set to: {cls._log_level}")

    @classmethod
    def get_log_level(cls) -> str:
        """Return the current class-level logging level."""
        return cls._log_level

    @staticmethod
    def _setup_logger(
        name: str = __name__,
        level: Optional[str] = None,
    ) -> logging.Logger:
        """
        Create a logger with consistent formatting.

        Parameters
        ----------
        name : str
            Logger name (typically ``__name__``).
        level : str, optional
            Override the class-level log level.

        Returns
        -------
        logging.Logger
        """
        effective_level = level or BaseStochasticModel._log_level

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, effective_level, logging.INFO))
        logger.handlers = []

        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, effective_level, logging.INFO))
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

        return logger

    # =========================================================================
    # Shared Utilities — Risk Metrics & Comparison
    # =========================================================================

    @staticmethod
    def _compute_risk_metrics(
        total_profits: pd.Series,
        probabilities: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute standard risk metrics from scenario-level total profits.

        This is a shared utility used by all subclass ``optimize()``
        implementations to populate ``OptimizationResult.risk_metrics``.

        Parameters
        ----------
        total_profits : pd.Series
            Total (summed over horizon) profit per scenario.
        probabilities : pd.Series
            Scenario probabilities aligned with *total_profits*.

        Returns
        -------
        dict[str, float]
            Keys: Expected_Profit, Profit_Std, VaR_95, VaR_99,
            CVaR_95, CVaR_99, Min_Profit, Max_Profit, Sharpe.
        """
        arr = total_profits.values.astype(float)
        p = probabilities.reindex(total_profits.index).values.astype(float)

        expected = float(np.dot(arr, p))
        std = float(np.sqrt(np.dot(p, (arr - expected) ** 2)))

        sorted_idx = np.argsort(arr)
        sorted_profits = arr[sorted_idx]
        sorted_p = p[sorted_idx]
        cum_p = np.cumsum(sorted_p)

        def _quantile(alpha: float) -> float:
            idx = np.searchsorted(cum_p, alpha)
            idx = min(idx, len(sorted_profits) - 1)
            return float(sorted_profits[idx])

        def _cvar(alpha: float) -> float:
            var = _quantile(alpha)
            mask = arr <= var
            if mask.any():
                tail_p = p[mask]
                tail_v = arr[mask]
                if tail_p.sum() > 0:
                    return float(np.dot(tail_v, tail_p) / tail_p.sum())
            return var

        var95 = _quantile(0.05)
        var99 = _quantile(0.01)
        cvar95 = _cvar(0.05)
        cvar99 = _cvar(0.01)
        sharpe = (expected / std) if std > 1e-10 else 0.0

        return {
            'Expected_Profit': expected,
            'Profit_Std': std,
            'VaR_95': var95,
            'VaR_99': var99,
            'CVaR_95': cvar95,
            'CVaR_99': cvar99,
            'Min_Profit': float(arr.min()),
            'Max_Profit': float(arr.max()),
            'Sharpe': sharpe,
        }

    @staticmethod
    def _compare_to_scenarios(
        realized_total_profit: float,
        scenario_total_profits: pd.Series,
        probabilities: pd.Series,
    ) -> Dict[str, float]:
        """
        Compare realized backtest profit to the scenario distribution.

        Parameters
        ----------
        realized_total_profit : float
            Sum of monthly realized profits.
        scenario_total_profits : pd.Series
            Total profit per scenario from the optimization result.
        probabilities : pd.Series
            Scenario probabilities.

        Returns
        -------
        dict[str, float]
        """
        arr = scenario_total_profits.values.astype(float)
        expected = float(np.dot(arr, probabilities.reindex(scenario_total_profits.index).values))
        percentile = float((arr <= realized_total_profit).mean() * 100)

        return {
            'Realized_Total_Profit': realized_total_profit,
            'Expected_Total_Profit': expected,
            'Realized_vs_Expected': realized_total_profit - expected,
            'Realized_Percentile': percentile,
        }

    # =========================================================================
    # Dunder Methods
    # =========================================================================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(solver='{self.solver}')"
