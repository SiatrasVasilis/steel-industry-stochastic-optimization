"""
VAR-based Scenario Generator
=============================

Implements :class:`BaseScenarioGenerator` using a Vector Autoregression (VAR)
model from *statsmodels*.  All core logic lives directly inside the
abstract-hook implementations -- no legacy static helper methods.

Workflow
--------
>>> from data.loader import DataLoader
>>> gen = VarModelScenarioGenerator(shock_distribution="t",
...                                  distribution_params={"df": 5})
>>> gen.fit(DataLoader(...), order=2)
>>> result = gen.generate(n_scenarios=1000, horizon=12)
>>> reduced = gen.reduce(result, n_clusters=50)
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


# =========================================================================
# VAR Scenario Generator
# =========================================================================

class VarModelScenarioGenerator(BaseScenarioGenerator):
    """
    VAR model-based scenario generator.

    Generates multivariate return scenarios via a VAR(*p*) model fitted to
    log-return data, with support for five shock distributions (normal, *t*,
    skewed-*t*, Laplace, uniform).  Level reconstruction is handled by the
    base class.

    Parameters
    ----------
    shock_distribution : str, default ``'normal'``
        Innovation distribution -- one of ``'normal'``, ``'t'``,
        ``'skewed_t'``, ``'laplace'``, ``'uniform'``.
    distribution_params : dict, optional
        Distribution-specific parameters (e.g. ``{'df': 5}`` for *t*).

    Notes
    -----
    For real-price conversion, use ``DataLoader.convert_to_real_prices()``
    before fitting the generator.
    """

    def __init__(
        self,
        shock_distribution: str = "normal",
        distribution_params: Optional[Dict] = None,
    ):
        config = GeneratorConfig(
            shock_distribution=ShockDistribution(shock_distribution),
            distribution_params=distribution_params or {},
        )
        super().__init__(config=config, variable_names=["D", "P", "C"])

        # VAR-specific state (populated during fit)
        self._var_model = None  # statsmodels VARResults
        self._fit_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "VAR"

    @property
    def supports_irf(self) -> bool:
        return True

    @property
    def requires_stationarity(self) -> bool:
        return True

    @property
    def var_model(self):
        """Fitted ``VARResults`` object, or *None* before :meth:`fit`."""
        return self._var_model

    @property
    def fit_params(self) -> Dict:
        return self._fit_params.copy()

    # =================================================================
    # Abstract-method implementations
    # =================================================================

    # -----------------------------------------------------------------
    # Order / lag selection
    # -----------------------------------------------------------------

    def _select_order(
        self,
        data: pd.DataFrame,
        max_order: int,
        method: str,
    ) -> OrderSelectionResult:
        """Select VAR lag order via information criteria."""
        from statsmodels.tsa.api import VAR

        model = VAR(data[self._variable_names])
        lag_sel = model.select_order(maxlags=max_order)

        selected_p = lag_sel.selected_orders.get(method)
        if selected_p is None:
            raise ValueError(
                f"Method '{method}' not in results. "
                f"Available: {list(lag_sel.selected_orders.keys())}"
            )

        # Collect IC scores for every lag tested
        all_scores: Dict[int, float] = {}
        ic_table = lag_sel.ics
        if hasattr(ic_table, "index"):
            method_upper = method.upper() if method != "fpe" else "FPE"
            if method_upper in ic_table.columns:
                all_scores = ic_table[method_upper].to_dict()

        logger.info(
            "Lag selection: p=%d (%s), tested lags 1..%d",
            selected_p, method, max_order,
        )

        return OrderSelectionResult(
            selected_order=selected_p,
            method=method,
            all_scores=all_scores,
            max_order_tested=max_order,
            additional_info={"selected_orders": dict(lag_sel.selected_orders)},
        )

    # -----------------------------------------------------------------
    # Model fitting
    # -----------------------------------------------------------------

    def _fit_model(
        self,
        data: pd.DataFrame,
        order: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fit a VAR(*order*) model to log-return data."""
        from statsmodels.tsa.api import VAR

        model = VAR(data[self._variable_names])
        var_results = model.fit(order)
        self._var_model = var_results

        self._fit_params = {
            "order": order,
            "nobs": var_results.nobs,
        }

        logger.info(
            "VAR(%d) fitted -- nobs=%d, AIC=%.4f, BIC=%.4f",
            order, var_results.nobs, var_results.aic, var_results.bic,
        )

        return {
            "var_model": var_results,
            "k_ar": var_results.k_ar,
            "nobs": var_results.nobs,
            "aic": var_results.aic,
            "bic": var_results.bic,
            "params": var_results.params,
            "order": order,
            "residuals": var_results.resid,  # For fit diagnostics
        }

    # -----------------------------------------------------------------
    # Scenario generation (returns only -- base handles levels / real)
    # -----------------------------------------------------------------

    def _generate_scenarios(
        self,
        n_scenarios: int,
        horizon: int,
        start_date: pd.Timestamp,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simulate *n_scenarios* forward trajectories of log returns.

        Returns long-form ``(scenarios_df, probabilities)`` where the base
        class subsequently applies ``_convert_to_levels``.
        """
        from scipy.linalg import cholesky
        from scipy import stats as sp_stats

        if self._var_model is None:
            raise RuntimeError("VAR model not fitted. Call fit() first.")

        var_model = self._var_model
        p = var_model.k_ar
        k = var_model.neqs
        Sigma = var_model.sigma_u.values
        dist = self.config.shock_distribution.value
        dist_params = self.config.distribution_params

        # Cholesky decomposition for correlated innovations
        chol = cholesky(Sigma, lower=True)

        # Initial conditions (last p observations from training data)
        endog = var_model.endog
        y_init = (endog.iloc[-p:].values if hasattr(endog, "iloc")
                  else endog[-p:].copy())

        rng = np.random.default_rng(seed)

        # Variable names from fitted model (or fall back to config)
        var_names = (list(endog.columns) if hasattr(endog, "columns")
                     else list(self._variable_names))

        logger.info(
            "Generating %d scenarios (horizon=%d, dist=%s, seed=%s)",
            n_scenarios, horizon, dist, seed,
        )

        # ---- shock generator ----------------------------------------

        def _generate_shock() -> np.ndarray:
            """Return a single *k*-dimensional correlated shock."""
            if dist == "normal":
                eps = rng.standard_normal(k)

            elif dist == "t":
                df = dist_params.get("df", 5)
                eps = rng.standard_normal(k) * np.sqrt(df / rng.chisquare(df))

            elif dist == "skewed_t":
                df = dist_params.get("df", 5)
                skewness = dist_params.get("skewness", [0.0] * k)
                eps = np.zeros(k)
                for i in range(k):
                    sn = sp_stats.skewnorm.rvs(
                        a=skewness[i],
                        scale=np.sqrt((df - 2) / df) if df > 2 else 1.0,
                        random_state=rng,
                    )
                    t_comp = sp_stats.t.rvs(df, random_state=rng)
                    eps[i] = 0.7 * sn + 0.3 * t_comp

            elif dist == "laplace":
                scale = dist_params.get("scale", 1.0 / np.sqrt(2))
                eps = rng.laplace(scale=scale, size=k)

            elif dist == "uniform":
                bounds = dist_params.get("bounds", 3.0)
                eps = rng.uniform(-bounds, bounds, size=k)

            else:
                raise ValueError(f"Unsupported shock distribution: {dist}")

            return chol @ eps

        # ---- forward simulation --------------------------------------

        all_paths = np.empty((n_scenarios, horizon, k))

        for s in range(n_scenarios):
            y_history = y_init.copy()
            for t in range(horizon):
                y_forecast = var_model.forecast(y_history, steps=1)[0]
                y_new = y_forecast + _generate_shock()
                all_paths[s, t, :] = y_new

                if p > 1:
                    y_history = np.vstack([y_history[1:], y_new.reshape(1, -1)])
                else:
                    y_history = y_new.reshape(1, -1)

        # ---- format to long-form DataFrame ---------------------------

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
                for vi, vn in enumerate(var_names):
                    row[vn] = all_paths[s, t, vi]
                records.append(row)

        scenarios_df = pd.DataFrame(records)
        scenarios_df["Date"] = pd.to_datetime(scenarios_df["Date"])
        scenarios_df["Scenario"] = scenarios_df["Scenario"].astype("string")
        scenarios_df.sort_values(["Scenario", "Date"], inplace=True)
        scenarios_df.reset_index(drop=True, inplace=True)

        # Equal probabilities
        probabilities = self.equal_probabilities(n_scenarios)

        logger.info(
            "Simulation complete -- %d scenarios x %d periods, shape=%s",
            n_scenarios, horizon, scenarios_df.shape,
        )

        return scenarios_df, probabilities

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def _get_model_diagnostics(self) -> List[DiagnosticResult]:
        """
        Run VAR post-estimation diagnostics.

        Tests performed:
        1. Stability (eigenvalues inside unit circle)
        2. Durbin-Watson (serial correlation proxy per variable)
        3. Ljung-Box (serial correlation in residuals)
        4. Jarque-Bera normality (per variable)
        5. Residual correlation matrix summary
        """
        diagnostics: List[DiagnosticResult] = []
        if self._var_model is None:
            return diagnostics

        vm = self._var_model
        resid = vm.resid

        # 1. Stability
        try:
            stable = vm.is_stable()
            diagnostics.append(DiagnosticResult(
                test_name="VAR Stability",
                passed=stable,
                details={
                    "message": ("All eigenvalues inside unit circle"
                                if stable else "Unstable VAR system"),
                },
            ))
        except Exception:
            pass

        # 2. Durbin-Watson
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(resid)
            for i, col in enumerate(self._variable_names):
                diagnostics.append(DiagnosticResult(
                    test_name=f"Durbin-Watson ({col})",
                    statistic=float(dw[i]),
                    passed=1.5 < dw[i] < 2.5,
                    details={"variable": col},
                ))
        except Exception:
            pass

        # 3. Ljung-Box (lag 10)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            for col in self._variable_names:
                if col not in resid.columns:
                    continue
                lb = acorr_ljungbox(resid[col], lags=[10], return_df=True)
                lb_pval = float(lb["lb_pvalue"].iloc[0])
                diagnostics.append(DiagnosticResult(
                    test_name=f"Ljung-Box ({col})",
                    statistic=float(lb["lb_stat"].iloc[0]),
                    pvalue=lb_pval,
                    passed=lb_pval > 0.05,
                    details={"variable": col, "lag": 10},
                ))
        except Exception:
            pass

        # 4. Jarque-Bera normality
        try:
            from scipy.stats import jarque_bera
            for col in self._variable_names:
                if col not in resid.columns:
                    continue
                jb_stat, jb_pval = jarque_bera(resid[col].dropna())
                diagnostics.append(DiagnosticResult(
                    test_name=f"Jarque-Bera ({col})",
                    statistic=float(jb_stat),
                    pvalue=float(jb_pval),
                    passed=jb_pval > 0.05,
                    details={"variable": col},
                ))
        except Exception:
            pass

        # 5. Residual correlation matrix
        try:
            corr = resid.corr()
            diagnostics.append(DiagnosticResult(
                test_name="Residual Correlation",
                passed=True,
                details={"correlation_matrix": corr.to_dict()},
            ))
        except Exception:
            pass

        return diagnostics

    # =================================================================
    # Residual / shock-distribution analysis
    # =================================================================

    def analyze_residuals(self) -> Dict[str, Any]:
        """
        Analyse VAR residuals and recommend a shock distribution.

        Examines skewness, kurtosis and formal normality tests
        (Jarque-Bera, Shapiro-Wilk) for each variable's residuals and
        returns per-variable + overall recommendations.

        Returns
        -------
        dict
            ``{'individual': {var: {...}}, 'overall_distribution': str,
            'overall_params': dict}``

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        from scipy import stats as sp_stats

        if self._var_model is None:
            raise RuntimeError("Model not fitted -- call fit() first.")

        residuals = self._var_model.resid
        recommendations: Dict[str, Dict] = {}

        for col in residuals.columns:
            r = residuals[col].dropna()
            skew = float(r.skew())
            kurt = float(r.kurtosis())  # excess kurtosis (normal ~ 0)
            jb_stat, jb_pval = sp_stats.jarque_bera(r)
            sw_stat, sw_pval = sp_stats.shapiro(
                r.values[:5000] if len(r) > 5000 else r.values
            )

            logger.debug(
                "%s residuals -- skew=%.3f, kurt=%.3f, JB p=%.4f, SW p=%.4f",
                col, skew, kurt, jb_pval, sw_pval,
            )

            # Decision tree
            if jb_pval > 0.05 and abs(skew) < 0.5 and -0.5 < kurt < 0.5:
                rec = {"distribution": "normal", "params": {},
                       "reason": "passes normality tests"}
            elif kurt > 2.0 and abs(skew) < 1.0:
                est_df = max(3, min(10, 4 + 6 / max(kurt, 0.1)))
                rec = {"distribution": "t", "params": {"df": round(est_df)},
                       "reason": f"heavy tails (kurt={kurt:.1f})"}
            elif abs(skew) > 0.75 and kurt > 1.0:
                est_df = max(3, min(8, 4 + 6 / max(kurt, 0.1)))
                rec = {"distribution": "skewed_t",
                       "params": {"df": est_df},
                       "reason": f"skewed ({skew:.2f}) + heavy tails"}
            elif kurt > 1.0 and abs(skew) < 0.5:
                rec = {"distribution": "laplace",
                       "params": {"scale": float(r.std() / np.sqrt(2))},
                       "reason": "heavy symmetric tails, sharp peak"}
            else:
                rec = {"distribution": "t", "params": {"df": 6},
                       "reason": "non-normal -- conservative fallback"}

            recommendations[col] = {**rec, "skew": skew, "kurtosis": kurt,
                                     "jb_pvalue": float(jb_pval),
                                     "sw_pvalue": float(sw_pval)}

        # Overall recommendation (majority vote / compromise)
        dist_votes = [r["distribution"] for r in recommendations.values()]
        if len(set(dist_votes)) == 1:
            overall_dist = dist_votes[0]
        else:
            overall_dist = "t"  # safe default when mixed

        # Average df if applicable
        df_vals = [r["params"]["df"] for r in recommendations.values()
                   if "df" in r.get("params", {})]
        overall_params: Dict[str, Any] = {}
        if df_vals:
            overall_params["df"] = round(np.mean(df_vals))

        logger.info(
            "Residual analysis -- recommendation: %s %s",
            overall_dist, overall_params,
        )

        return {
            "individual": recommendations,
            "overall_distribution": overall_dist,
            "overall_params": overall_params,
        }

    # =================================================================
    # IRF (optional, base checks supports_irf before calling)
    # =================================================================

    def _compute_irf_impl(
        self, periods: int, orthogonalized: bool, plot: bool,
    ) -> Dict[str, Any]:
        """Compute impulse-response functions from the fitted VAR."""
        if self._var_model is None:
            raise RuntimeError("Model not fitted.")

        irf = self._var_model.irf(periods)

        if plot:
            irf.plot(orth=orthogonalized)

        return {
            "irf_object": irf,
            "periods": periods,
            "orthogonalized": orthogonalized,
        }

    # =================================================================
    # Plotting (delegate to base; provide a lightweight override hook)
    # =================================================================

    def _plot_analysis(self, result, data: pd.DataFrame) -> None:
        """Plot analysis outputs (stationarity + order selection)."""
        import matplotlib.pyplot as plt

        n_vars = len(self._variable_names)
        fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
        if n_vars == 1:
            axes = [axes]

        for ax, col in zip(axes, self._variable_names):
            if col in data.columns:
                ax.plot(data.index, data[col], linewidth=0.8)
                ax.set_title(f"{col} log-returns")
                ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"VAR Analysis - recommended order: {result.recommended_order}",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
