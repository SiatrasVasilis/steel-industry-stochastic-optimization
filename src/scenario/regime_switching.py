"""
Regime-Switching Scenario Generator
====================================

Implements :class:`BaseScenarioGenerator` using a Markov-Switching VAR model.

The model captures two distinct market regimes (e.g. "normal" vs "stress")
with separate VAR dynamics, mean returns, and volatility structures for each.
Scenario generation samples regime paths from the Markov chain and applies
regime-specific dynamics, producing realistic crash / boom trajectories.

Approach
--------
A direct multivariate Markov-Switching VAR is not available in statsmodels.
We use the following strategy:

1.  **Regime identification**: Fit a univariate Markov-Switching AR model
    on the first principal component of the log-return vector (capturing
    the dominant co-movement) to learn the hidden regime sequence and
    transition matrix.

2.  **Regime-conditional VAR**: Using the most-likely regime path from (1),
    split the multivariate log-return data into regime subsets and estimate
    a separate VAR + covariance for each regime.

3.  **Simulation**: For each scenario, simulate a regime path from the
    Markov chain, then at each step draw innovations from the
    regime-specific VAR dynamics and covariance.

This is a standard approach used in commodity risk management (see e.g.
Ang & Bekaert 2002, Hamilton 1989 applied to multivariate settings).

Workflow
--------
>>> from data.loader import DataLoader
>>> gen = RegimeSwitchingGenerator(n_regimes=2)
>>> gen.fit(loader)
>>> result = gen.generate(n_scenarios=3000, horizon=12)
>>> reduced = gen.reduce(result, n_clusters=200)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .base import (
    BaseScenarioGenerator,
    DiagnosticResult,
    GeneratorConfig,
    OrderSelectionResult,
    ShockDistribution,
)

logger = logging.getLogger(__name__)


class RegimeSwitchingGenerator(BaseScenarioGenerator):
    """
    Markov-Switching VAR scenario generator.

    Captures regime-dependent dynamics (normal vs stress) for multivariate
    commodity return series.  Scenario paths switch between regimes according
    to estimated transition probabilities, producing realistic tail events.

    Parameters
    ----------
    n_regimes : int, default 2
        Number of hidden regimes (2 = normal/stress).
    shock_distribution : str, default ``'t'``
        Innovation distribution within each regime.
    distribution_params : dict, optional
        Parameters for the shock distribution (e.g. ``{'df': 5}``).
    switching_variance : bool, default True
        If True, each regime has its own covariance matrix.
        If False, regimes share a single covariance (only means differ).
    """

    def __init__(
        self,
        n_regimes: int = 2,
        shock_distribution: str = "t",
        distribution_params: Optional[Dict] = None,
        switching_variance: bool = True,
    ):
        config = GeneratorConfig(
            shock_distribution=ShockDistribution(shock_distribution),
            distribution_params=distribution_params or {"df": 5},
        )
        super().__init__(config=config, variable_names=["D", "P", "C"])

        self.n_regimes = n_regimes
        self.switching_variance = switching_variance

        # Fitted state (populated during _fit_model)
        self._transition_matrix: Optional[np.ndarray] = None
        self._regime_labels: Optional[np.ndarray] = None  # per-observation
        self._regime_vars: Dict[int, Any] = {}             # VAR ref per regime (shared)
        self._regime_means: Dict[int, np.ndarray] = {}     # mean shift per regime
        self._regime_covs: Dict[int, np.ndarray] = {}      # Σ per regime
        self._regime_order: Dict[int, int] = {}             # VAR order per regime
        self._regime_fractions: Dict[int, float] = {}      # fraction of time in each
        self._ms_model = None                               # statsmodels MS result
        self._steady_state_probs: Optional[np.ndarray] = None
        self._full_var = None                               # shared full-sample VAR
        self._shared_coefs: list = []                       # shared AR coef matrices
        self._var_order: int = 1                            # VAR order

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "RegimeSwitchingVAR"

    @property
    def supports_irf(self) -> bool:
        return False  # IRF not meaningful across regimes

    @property
    def requires_stationarity(self) -> bool:
        return True  # each regime should be stationary

    # ------------------------------------------------------------------
    # Order selection
    # ------------------------------------------------------------------

    def _select_order(
        self,
        data: pd.DataFrame,
        max_order: int,
        method: str,
    ) -> OrderSelectionResult:
        """
        Select VAR lag order using standard VAR BIC on the full sample.

        The order is applied to each regime-conditional VAR.
        """
        from statsmodels.tsa.api import VAR

        model = VAR(data[self._variable_names])
        lag_sel = model.select_order(maxlags=max_order)

        selected_p = lag_sel.selected_orders.get(method, 1)

        all_scores: Dict[int, float] = {}
        ic_table = lag_sel.ics
        if hasattr(ic_table, "index"):
            method_upper = method.upper()
            if method_upper in ic_table.columns:
                all_scores = ic_table[method_upper].to_dict()

        logger.info(
            "Lag selection (regime-switching): p=%d (%s), tested 1..%d",
            selected_p, method, max_order,
        )

        return OrderSelectionResult(
            selected_order=max(selected_p, 1),
            method=method,
            all_scores=all_scores,
            max_order_tested=max_order,
        )

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit_model(
        self,
        data: pd.DataFrame,
        order: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fit the Markov-Switching Intercept-Heteroskedasticity (MSIH) model.

        Steps:
        1. Fit a **single full-sample VAR** for stable, well-estimated dynamics.
        2. Fit a univariate MS-AR on PC1 of residuals to identify regimes
           and learn the transition matrix.
        3. Compute regime-specific **mean shifts** and **covariance matrices**
           from the full-sample VAR residuals split by regime.

        This avoids the small-sample instability of fitting separate VARs
        per regime, while still capturing regime-dependent volatility
        and intercept shifts.
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
        from statsmodels.tsa.api import VAR
        from sklearn.decomposition import PCA

        vars_data = data[self._variable_names].dropna()
        k = len(self._variable_names)

        # --- Step 1: Fit full-sample VAR for stable dynamics ---
        var_model = VAR(vars_data)
        var_result = var_model.fit(order)
        self._full_var = var_result

        # Store shared AR coefficients (excluding intercept)
        self._shared_coefs = []
        for lag in range(order):
            start = 1 + lag * k
            self._shared_coefs.append(var_result.params.iloc[start: start + k].values)
        self._var_order = order

        residuals = var_result.resid  # (T-p) x k
        logger.info(
            "Full-sample VAR(%d) fitted: %d residuals, stable=%s",
            order, len(residuals), var_result.is_stable(),
        )

        # --- Step 2: MS-AR on PC1 of residuals for regime identification ---
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(residuals.values).ravel()
        pc1_series = pd.Series(pc1, index=residuals.index, name="PC1")

        logger.info(
            "PC1 explains %.1f%% of residual variance — using for regime detection",
            pca.explained_variance_ratio_[0] * 100,
        )

        ms_order = min(order, 2)

        ms_model = MarkovAutoregression(
            pc1_series,
            k_regimes=self.n_regimes,
            order=ms_order,
            switching_ar=False,
            switching_variance=self.switching_variance,
        )

        best_result = None
        best_llf = -np.inf

        for attempt in range(10):
            try:
                res = ms_model.fit(
                    disp=False,
                    search_reps=20 if attempt == 0 else 5,
                )
                if res.llf > best_llf:
                    best_llf = res.llf
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError(
                "Markov-Switching model failed to converge after multiple attempts. "
                "Try using more observations or fewer regimes."
            )

        self._ms_model = best_result

        # Transition matrix
        self._transition_matrix = best_result.regime_transition[:, :, 0].T

        # Most-likely regime path (aligned with residuals, not raw data)
        smoothed = best_result.smoothed_marginal_probabilities
        regime_labels = smoothed.values.argmax(axis=1)

        # Label regimes: low-vol = 0 (normal), high-vol = 1+ (stress)
        regime_vols = {}
        for r in range(self.n_regimes):
            mask = regime_labels == r
            if mask.sum() > 0:
                regime_vols[r] = pc1_series.iloc[: len(regime_labels)][mask].std()
            else:
                regime_vols[r] = 0.0

        sorted_regimes = sorted(regime_vols, key=regime_vols.get)
        relabel_map = {old: new for new, old in enumerate(sorted_regimes)}

        regime_labels = np.array([relabel_map[r] for r in regime_labels])
        P = self._transition_matrix.copy()
        perm = sorted_regimes
        self._transition_matrix = P[np.ix_(perm, perm)]

        self._regime_labels = regime_labels

        # Steady-state probabilities
        self._steady_state_probs = self._compute_steady_state(
            self._transition_matrix
        )

        # --- Step 3: Regime-specific mean shifts and covariances ---
        # Compute from the *residuals* of the full-sample VAR.
        # Apply adaptive Ledoit-Wolf style shrinkage: regime covariances are
        # blended with the full-sample covariance in proportion to the regime's
        # sample size. This prevents small regimes (e.g. 20 obs) from having
        # wildly overestimated volatility.
        aligned_resid = residuals.iloc[: len(regime_labels)]
        full_cov = aligned_resid.cov().values
        shrinkage_kappa = 10  # regularization strength

        for r in range(self.n_regimes):
            mask = regime_labels == r
            n_r = mask.sum()
            frac = n_r / len(regime_labels)
            self._regime_fractions[r] = frac

            label = "Normal" if r == 0 else f"Stress-{r}"

            if n_r >= k + 1:
                regime_resid = aligned_resid[mask]
                self._regime_means[r] = regime_resid.mean().values
                raw_cov = regime_resid.cov().values
                # Adaptive shrinkage: w = n_r / (n_r + kappa)
                w = n_r / (n_r + shrinkage_kappa)
                self._regime_covs[r] = w * raw_cov + (1 - w) * full_cov
            else:
                logger.warning(
                    "Regime %d has only %d obs — using full-sample covariance.",
                    r, n_r,
                )
                self._regime_means[r] = np.zeros(k)
                self._regime_covs[r] = full_cov.copy()

            # Store VAR reference for diagnostics
            self._regime_vars[r] = var_result
            self._regime_order[r] = order

            logger.info(
                "  Regime %d (%s): %.1f%% of obs, shrinkage_w=%.2f, "
                "P(stay)=%.2f, avg_vol=%.4f, steady-state=%.1f%%",
                r, label, frac * 100, n_r / (n_r + shrinkage_kappa),
                self._transition_matrix[r, r],
                np.sqrt(np.diag(self._regime_covs[r])).mean(),
                self._steady_state_probs[r] * 100,
            )

        logger.info("[✓] Regime-Switching (MSIH) VAR fitted successfully!")

        return {
            "n_regimes": self.n_regimes,
            "transition_matrix": self._transition_matrix,
            "regime_fractions": self._regime_fractions,
            "steady_state_probs": self._steady_state_probs,
            "order": order,
            "residuals": residuals,
            "ms_model": best_result,
        }

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def _generate_scenarios(
        self,
        n_scenarios: int,
        horizon: int,
        start_date: pd.Timestamp,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simulate MSIH forward trajectories.

        Uses **shared VAR dynamics** (AR coefficients from full-sample VAR)
        with **regime-specific mean shifts and covariance**.

        For each scenario:
        1. Sample a starting regime from the steady-state distribution.
        2. At each step, draw the next regime from the transition matrix.
        3. Compute VAR forecast using shared AR coefficients.
        4. Add regime-specific mean shift + correlated shock from
           regime-specific covariance.
        """
        from scipy.linalg import cholesky

        rng = np.random.default_rng(seed)
        k = len(self._variable_names)
        dist = self.config.shock_distribution.value
        dist_params = self.config.distribution_params

        # Cholesky factors per regime
        chol_factors = {}
        for r in range(self.n_regimes):
            cov = self._regime_covs[r]
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals <= 0):
                cov = cov + np.eye(k) * (abs(eigvals.min()) + 1e-8)
            chol_factors[r] = cholesky(cov, lower=True)

        # Shared VAR intercept (full-sample)
        full_intercept = self._full_var.params.iloc[0].values
        order = self._var_order

        # Initial conditions from last observations of training data
        last_returns = self._log_returns[self._variable_names].iloc[-order:].values

        P = self._transition_matrix
        pi0 = self._steady_state_probs

        logger.info(
            "Generating %d MSIH scenarios "
            "(horizon=%d, regimes=%d, dist=%s)",
            n_scenarios, horizon, self.n_regimes, dist,
        )

        def _generate_shock(regime: int) -> np.ndarray:
            """Single k-dimensional correlated shock for a given regime."""
            L = chol_factors[regime]
            if dist == "normal":
                eps = rng.standard_normal(k)
            elif dist == "t":
                df = dist_params.get("df", 5)
                eps = rng.standard_normal(k) * np.sqrt(df / rng.chisquare(df))
            elif dist == "laplace":
                scale = dist_params.get("scale", 1.0 / np.sqrt(2))
                eps = rng.laplace(scale=scale, size=k)
            else:
                eps = rng.standard_normal(k)
            return L @ eps

        all_paths = np.empty((n_scenarios, horizon, k))

        for s in range(n_scenarios):
            regime = rng.choice(self.n_regimes, p=pi0)
            y_history = last_returns.copy()

            for t in range(horizon):
                if t > 0:
                    regime = rng.choice(self.n_regimes, p=P[regime])

                # Shared VAR dynamics: y_t = intercept + A1*y_{t-1} + ...
                y_forecast = full_intercept.copy()
                for lag_idx, A in enumerate(self._shared_coefs):
                    if lag_idx < len(y_history):
                        y_forecast = y_forecast + A @ y_history[-(lag_idx + 1)]

                # Regime-specific mean shift + shock
                y_new = y_forecast + self._regime_means[regime] + _generate_shock(regime)
                all_paths[s, t, :] = y_new

                # Update history
                if order > 1:
                    y_history = np.vstack([y_history[1:], y_new.reshape(1, -1)])
                else:
                    y_history = y_new.reshape(1, -1)

        # Format to long-form DataFrame
        future_dates = pd.date_range(
            start=(start_date + pd.offsets.MonthBegin(0)).normalize(),
            periods=horizon,
            freq="MS",
        )

        records: list[dict] = []
        for s in range(n_scenarios):
            sid = f"s{s}"
            for t in range(horizon):
                row = {"Date": future_dates[t], "Scenario": sid}
                for vi, vn in enumerate(self._variable_names):
                    row[vn] = all_paths[s, t, vi]
                records.append(row)

        scenarios_df = pd.DataFrame(records)
        scenarios_df["Date"] = pd.to_datetime(scenarios_df["Date"])
        scenarios_df["Scenario"] = scenarios_df["Scenario"].astype("string")
        scenarios_df.sort_values(["Scenario", "Date"], inplace=True)
        scenarios_df.reset_index(drop=True, inplace=True)

        probabilities = self.equal_probabilities(n_scenarios)

        logger.info(
            "Regime-switching simulation complete -- %d scenarios x %d periods",
            n_scenarios, horizon,
        )

        return scenarios_df, probabilities

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _get_model_diagnostics(self) -> List[DiagnosticResult]:
        """Regime-switching specific diagnostics."""
        diagnostics: List[DiagnosticResult] = []

        if self._transition_matrix is None:
            return diagnostics

        P = self._transition_matrix

        # 1. Transition matrix validity
        row_sums = P.sum(axis=1)
        valid = np.allclose(row_sums, 1.0, atol=1e-6)
        diagnostics.append(DiagnosticResult(
            test_name="Transition Matrix Validity",
            passed=valid,
            details={"row_sums": row_sums.tolist()},
        ))

        # 2. Regime persistence
        for r in range(self.n_regimes):
            persistence = P[r, r]
            label = "Normal" if r == 0 else f"Stress-{r}"
            diagnostics.append(DiagnosticResult(
                test_name=f"Regime Persistence ({label})",
                statistic=float(persistence),
                passed=0.3 < persistence < 0.999,
                details={
                    "regime": r,
                    "expected_duration_months": 1.0 / (1.0 - persistence + 1e-10),
                    "fraction": self._regime_fractions.get(r, 0),
                },
                recommendation=(
                    f"Expected duration: {1.0 / (1.0 - persistence + 1e-10):.1f} months"
                ),
            ))

        # 3. Regime separation (volatility ratio)
        if len(self._regime_covs) >= 2:
            vol_0 = np.sqrt(np.diag(self._regime_covs[0])).mean()
            vol_1 = np.sqrt(np.diag(self._regime_covs[1])).mean()
            vol_ratio = max(vol_0, vol_1) / (min(vol_0, vol_1) + 1e-10)
            diagnostics.append(DiagnosticResult(
                test_name="Volatility Ratio (Stress/Normal)",
                statistic=float(vol_ratio),
                passed=vol_ratio > 1.3,
                recommendation=(
                    f"Stress regime is {vol_ratio:.1f}x more volatile. "
                    f"{'Good separation.' if vol_ratio > 1.5 else 'Weak — regimes may be similar.'}"
                ),
            ))

        # 4. Full-sample VAR stability
        if self._full_var is not None:
            try:
                stable = self._full_var.is_stable()
                diagnostics.append(DiagnosticResult(
                    test_name="Full-Sample VAR Stability",
                    passed=stable,
                ))
            except Exception:
                pass

        # 5. MS model log-likelihood
        if self._ms_model is not None:
            diagnostics.append(DiagnosticResult(
                test_name="MS Model Log-Likelihood",
                statistic=float(self._ms_model.llf),
                passed=True,
                details={
                    "aic": float(self._ms_model.aic),
                    "bic": float(self._ms_model.bic),
                },
            ))

        return diagnostics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_steady_state(P: np.ndarray) -> np.ndarray:
        """Compute the steady-state distribution of a transition matrix."""
        n = P.shape[0]
        # Solve π P = π  ⟺  (P^T - I) π = 0  with  Σπ = 1
        A = np.vstack([P.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.maximum(pi, 0)
        pi /= pi.sum()
        return pi

    def regime_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of regime characteristics.

        Columns: regime, label, fraction, persistence, expected_duration,
        avg_vol, mean_return_* (per variable).
        """
        rows = []
        for r in range(self.n_regimes):
            label = "Normal" if r == 0 else f"Stress-{r}"
            persistence = self._transition_matrix[r, r]
            row = {
                "regime": r,
                "label": label,
                "fraction": self._regime_fractions.get(r, 0),
                "persistence": persistence,
                "expected_duration_months": 1.0 / (1.0 - persistence + 1e-10),
                "avg_volatility": np.sqrt(np.diag(self._regime_covs[r])).mean()
                if r in self._regime_covs else None,
            }
            # Per-variable mean returns
            if r in self._regime_means:
                for i, vn in enumerate(self._variable_names):
                    row[f"mean_{vn}"] = self._regime_means[r][i]
            rows.append(row)
        return pd.DataFrame(rows)
