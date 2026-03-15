"""
Benchmark Models for Comparison

This module implements rule-based planning models that serve as benchmarks
for evaluating the value of stochastic optimization. These models represent
traditional ERP/MRP + SAP IBP style planning approaches.

Classes
-------
SafetyStockParams
    Policy parameters for the safety stock model.
SafetyStockModel
    Traditional planning with safety stock buffers and static procurement allocation.

The benchmark models inherit from BaseStochasticModel to reuse:
- Variable mapping and validation
- Risk metric computation 
- Result containers (OptimizationResult, BacktestResult)
- Plotting capabilities

This allows direct comparison of stochastic vs. deterministic approaches
using identical metrics and visualizations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING

from .base import BaseStochasticModel
from .results import OptimizationResult, BacktestResult
from .stochastic import ModelParameters

if TYPE_CHECKING:
    from ..params.risk import RiskProfile


@dataclass
class SafetyStockParams:
    """
    Policy parameters for traditional safety-stock based planning.
    
    This represents how SAP IBP / ERP systems typically approach planning:
    - Point forecast with safety stock buffer for uncertainty
    - Static procurement allocation rules (fixed % to each channel)
    - Level-load or chase production strategy
    
    Attributes
    ----------
    service_level : float
        Target service level (fill rate) for safety stock calculation.
        Default 0.95 = 95% probability of meeting demand from stock.
    review_period : int
        Planning review cycle in periods (months). Affects safety stock.
        Default 1 (monthly review).
    fixed_pct : float
        Percentage of expected RM requirement allocated to fixed contracts.
        Default 0.60 (60% fixed).
    framework_pct : float
        Percentage of demand variability hedged via framework contracts.
        Default 0.25 (25% framework to cover fluctuations).
    production_smoothing : bool
        If True, use level-load production (produce at average rate).
        If False, use chase strategy (follow demand forecast).
        Default True (level-load).
    safety_stock_fg : bool
        Whether to maintain finished goods safety stock. Default True.
    safety_stock_rm : bool
        Whether to maintain raw material safety stock. Default True.
    safety_stock_periods : float
        Number of periods of demand to hold as safety stock (alternative
        to service_level based calculation). If None, uses service_level.
        Default None.
    """
    # Service level targets
    service_level: float = 0.95
    review_period: int = 1
    
    # Procurement allocation (should sum to <= 1.0; remainder is spot)
    fixed_pct: float = 0.60
    framework_pct: float = 0.25
    
    # Production strategy
    production_smoothing: bool = True
    
    # Safety stock configuration
    safety_stock_fg: bool = True
    safety_stock_rm: bool = True
    safety_stock_periods: Optional[float] = None
    
    def __post_init__(self):
        """Validate policy parameters."""
        if not 0 < self.service_level < 1:
            raise ValueError(f"service_level must be in (0, 1), got {self.service_level}")
        if self.fixed_pct + self.framework_pct > 1.0:
            raise ValueError(
                f"fixed_pct ({self.fixed_pct}) + framework_pct ({self.framework_pct}) "
                f"exceeds 1.0"
            )
        if self.fixed_pct < 0 or self.framework_pct < 0:
            raise ValueError("Allocation percentages must be non-negative")


class SafetyStockModel(BaseStochasticModel):
    """
    Traditional Safety Stock Planning Model.
    
    This model represents ERP/MRP + SAP IBP style planning:
    
    1. **Point Forecast**: Use expected (mean) values from scenarios
    2. **Safety Stock**: Buffer based on demand variability and service level
    3. **Static Allocation**: Fixed % split across procurement channels
    4. **Production**: Level-load (smoothing) or chase (follow forecast)
    
    The model doesn't optimize — it applies deterministic rules. However,
    it evaluates these rule-based decisions across ALL scenarios to compute
    risk metrics, enabling fair comparison with stochastic optimization.
    
    Value of Stochastic Solution (VSS)
    ----------------------------------
    The difference in expected profit between the stochastic model and this
    benchmark quantifies the value of explicit uncertainty modeling::
    
        VSS = E[Profit_stochastic] - E[Profit_safety_stock]
    
    Required Variables
    ------------------
    D      : Demand [Tn FG]
    c_spot : Spot RM unit cost [€/Tn RM]
    p_sell : Selling price [€/Tn FG]
    
    Examples
    --------
    >>> from optimization import SafetyStockModel
    >>> from optimization.benchmark import SafetyStockParams
    >>> 
    >>> model = SafetyStockModel()
    >>> policy = SafetyStockParams(service_level=0.95, fixed_pct=0.6)
    >>> result = model.run(scenarios, prob, params=params, policy=policy)
    >>> print(f"Expected Profit: €{result.risk_metrics['Expected_Profit']:,.0f}")
    """
    
    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------
    
    @property
    def model_name(self) -> str:
        return "SafetyStockPolicy"
    
    @property
    def required_variables(self) -> List[str]:
        return ["D", "c_spot", "p_sell"]
    
    @property
    def default_variable_mapping(self) -> Dict[str, str]:
        """Common aliases for the three required variables."""
        return {
            "demand": "D",
            "Demand": "D",
            "demand_tons": "D",
            "spot_cost": "c_spot",
            "Spot_Cost": "c_spot",
            "scrap_cost": "c_spot",
            "Scrap_Cost": "c_spot",
            "C": "c_spot",
            "sell_price": "p_sell",
            "Sell_Price": "p_sell",
            "steel_price": "p_sell",
            "Steel_Price": "p_sell",
            "P": "p_sell",
        }
    
    def __init__(self):
        """Initialize the SafetyStockModel (no solver needed)."""
        super().__init__(solver=None)
        self._last_result: Optional[OptimizationResult] = None
    
    @staticmethod
    def _get_param_value(param: Union[Dict, float], key, default: float = 0.0) -> float:
        """Helper to extract parameter value (supports dict or scalar)."""
        if isinstance(param, dict):
            return param.get(key, default)
        return float(param)
    
    # ------------------------------------------------------------------
    # Core Methods
    # ------------------------------------------------------------------
    
    def optimize(
        self,
        scenarios: pd.DataFrame,
        prob: pd.Series,
        params: ModelParameters = None,
        policy: SafetyStockParams = None,
        risk_profile: Optional["RiskProfile"] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Apply safety stock policy and evaluate across all scenarios.
        
        This method:
        1. Computes point forecasts (mean D, c_spot, p_sell per period)
        2. Calculates safety stock targets based on demand variability
        3. Applies static procurement allocation rules
        4. Evaluates the resulting decisions across ALL scenarios
        5. Computes risk metrics for comparison with stochastic models
        
        Parameters
        ----------
        scenarios : pd.DataFrame
            Long-form DataFrame with columns ['Date', 'Scenario', 'D', 'c_spot', 'p_sell'].
        prob : pd.Series
            Scenario probabilities indexed by scenario ID.
        params : ModelParameters
            Model parameters (same as AdvancedTacticalPlanningModel).
        policy : SafetyStockParams
            Safety stock policy configuration.
        risk_profile : RiskProfile, optional
            Ignored (included for API compatibility).
        **kwargs
            Additional arguments (for compatibility).
            
        Returns
        -------
        OptimizationResult
            Same structure as stochastic models, enabling direct comparison.
        """
        # Allow params/policy via kwargs
        if params is None:
            params = kwargs.get("params")
        if policy is None:
            policy = kwargs.get("policy", SafetyStockParams())
        
        if params is None:
            raise ValueError("params (ModelParameters) is required")
        
        logger = self._setup_logger()
        logger.info(f"[{self.model_name}] Applying safety stock policy...")
        
        # Extract scenario structure
        dates = sorted(scenarios['Date'].unique())
        scenario_ids = scenarios['Scenario'].unique()
        n_scenarios = len(scenario_ids)
        T = len(dates)
        
        # ---------------------------------------------------------------------
        # STEP 1: Compute point forecasts and demand statistics
        # ---------------------------------------------------------------------
        logger.debug("  Computing point forecasts...")
        
        forecasts = {}
        demand_stats = {}
        
        for t, date in enumerate(dates):
            date_data = scenarios[scenarios['Date'] == date]
            
            # Weighted mean forecasts
            weights = prob.reindex(date_data['Scenario']).values
            weights = weights / weights.sum()  # Normalize
            
            forecasts[date] = {
                'D_mean': float(np.average(date_data['D'].values, weights=weights)),
                'c_spot_mean': float(np.average(date_data['c_spot'].values, weights=weights)),
                'p_sell_mean': float(np.average(date_data['p_sell'].values, weights=weights)),
            }
            
            # Demand statistics for safety stock
            demand_stats[date] = {
                'D_std': float(date_data['D'].std()),
                'D_min': float(date_data['D'].min()),
                'D_max': float(date_data['D'].max()),
            }
        
        # Overall demand variability (for safety stock)
        all_D = scenarios['D'].values
        D_std_overall = float(all_D.std())
        D_mean_overall = float(all_D.mean())
        
        # ---------------------------------------------------------------------
        # STEP 2: Calculate safety stock targets
        # ---------------------------------------------------------------------
        logger.debug("  Calculating safety stock targets...")
        
        if policy.safety_stock_periods is not None:
            # Simple rule: X periods of average demand
            SS_FG = policy.safety_stock_periods * D_mean_overall
        else:
            # Service level based: SS = z * σ * √L
            z = stats.norm.ppf(policy.service_level)
            SS_FG = z * D_std_overall * np.sqrt(policy.review_period)
        
        SS_RM = params.alpha * SS_FG if policy.safety_stock_rm else 0.0
        SS_FG = SS_FG if policy.safety_stock_fg else 0.0
        
        logger.debug(f"    Safety Stock FG: {SS_FG:,.0f} Tn")
        logger.debug(f"    Safety Stock RM: {SS_RM:,.0f} Tn")
        
        # ---------------------------------------------------------------------
        # STEP 3: Determine first-stage decisions (rule-based)
        # ---------------------------------------------------------------------
        logger.debug("  Computing procurement and production decisions...")
        
        decisions_data = []
        
        for t, date in enumerate(dates):
            fc = forecasts[date]
            D_mean = fc['D_mean']
            D_std = demand_stats[date]['D_std']
            
            # RM requirement = alpha * expected production
            RM_base_need = params.alpha * D_mean
            
            # Static allocation: fixed contracts
            x_fix = min(
                policy.fixed_pct * RM_base_need,
                self._get_param_value(params.x_fix_max, date, float('inf'))
            )
            
            # Framework reservation: hedge demand variability
            x_opt = min(
                policy.framework_pct * params.alpha * D_std,
                self._get_param_value(params.x_opt_max, date, float('inf'))
            )
            
            # Production decisions
            # Safety stock build-up: add fraction of SS target to production
            # to gradually build the buffer over the horizon
            ss_build_rate = SS_FG / T if T > 0 else 0.0
            
            if policy.production_smoothing:
                # Level-load: produce at average rate + SS build-up
                P_base = min(
                    D_mean_overall + ss_build_rate,
                    self._get_param_value(params.Cap_base, date, float('inf'))
                )
            else:
                # Chase: follow demand forecast + SS build-up
                P_base = min(
                    D_mean + ss_build_rate,
                    self._get_param_value(params.Cap_base, date, float('inf'))
                )
            
            decisions_data.append({
                'Date': date,
                'x_fix': x_fix,
                'x_opt': x_opt,
                'P_base': P_base,
                'D_forecast': D_mean,
                'c_spot_forecast': fc['c_spot_mean'],
                'p_sell_forecast': fc['p_sell_mean'],
            })
        
        decisions = pd.DataFrame(decisions_data).set_index('Date')
        
        # ---------------------------------------------------------------------
        # STEP 4: Evaluate decisions across ALL scenarios (recourse simulation)
        # ---------------------------------------------------------------------
        logger.debug("  Evaluating across scenarios...")
        
        scenario_profits = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        
        # Stage 2 result accumulators
        stage2_y_opt = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        stage2_x_spot = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        stage2_P_flex = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        stage2_I_rm = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        stage2_I_fg = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        stage2_S = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        stage2_U = pd.DataFrame(index=dates, columns=scenario_ids, dtype=float)
        
        for s in scenario_ids:
            scenario_data = scenarios[scenarios['Scenario'] == s].set_index('Date')
            
            # Track inventory across periods
            I_rm_prev = params.I_rm0 + SS_RM / T  # Start building buffer
            I_fg_prev = params.I_fg0 + SS_FG / T
            
            for t, date in enumerate(dates):
                # Realized scenario values
                D_actual = float(scenario_data.loc[date, 'D'])
                c_spot_actual = float(scenario_data.loc[date, 'c_spot'])
                p_sell_actual = float(scenario_data.loc[date, 'p_sell'])
                
                # First-stage decisions
                x_fix_t = float(decisions.loc[date, 'x_fix'])
                x_opt_t = float(decisions.loc[date, 'x_opt'])
                P_base_t = float(decisions.loc[date, 'P_base'])
                
                # Framework exercise price
                basis = self._get_param_value(params.basis_opt, date, 0.0)
                floor_val = self._get_param_value(params.floor_opt, date, 0.0)
                cap_val = self._get_param_value(params.cap_opt, date, float('inf'))
                c_opt_t = min(max(c_spot_actual + basis, floor_val), cap_val)
                
                # Recourse decisions (simple heuristics)
                # Exercise framework if it's cheaper than spot
                if c_opt_t <= c_spot_actual:
                    y_opt_t = x_opt_t  # Exercise full reservation
                else:
                    y_opt_t = 0.0  # Don't exercise
                
                # RM contracted inflows
                RM_contracted = x_fix_t + y_opt_t
                
                # Determine full production plan first, then buy spot to cover
                # Base production: planned level (not yet constrained by RM)
                Cap_flex = self._get_param_value(params.Cap_flex, date, 0.0)
                
                # Flex production decision: produce if demand exceeds base output
                # OR if we need to rebuild FG safety stock
                FG_available_est = I_fg_prev + P_base_t
                demand_gap = max(0, D_actual - FG_available_est)
                
                # Also consider FG safety stock replenishment
                FG_below_target = max(0, SS_FG - (FG_available_est - D_actual))
                ss_replenish_fg = FG_below_target * 0.3  # Gradual rebuild (30% per period)
                
                P_flex_t = min(demand_gap + ss_replenish_fg, Cap_flex)
                
                # Total RM needed for planned production
                total_RM_need = params.alpha * (P_base_t + P_flex_t)
                RM_available = I_rm_prev + RM_contracted
                
                # Spot purchases to cover production gap + safety stock maintenance
                RM_shortfall = max(0, total_RM_need - RM_available)
                target_RM = SS_RM
                RM_after_production = RM_available + RM_shortfall - total_RM_need
                RM_below_target = max(0, target_RM - RM_after_production)
                x_spot_t = RM_shortfall + RM_below_target * 0.5  # Gradual replenishment
                
                # With spot purchased, check actual RM available
                RM_total_available = RM_available + x_spot_t
                actual_P_base = min(P_base_t, RM_total_available / params.alpha)
                P_flex_t = min(P_flex_t, max(0, RM_total_available / params.alpha - actual_P_base))
                
                # Update RM inventory
                RM_used = params.alpha * (actual_P_base + P_flex_t)
                I_rm_t = max(0, I_rm_prev + RM_contracted + x_spot_t - RM_used)
                
                # FG inventory dynamics
                FG_produced = actual_P_base + P_flex_t
                FG_available_total = I_fg_prev + FG_produced
                sales_t = min(D_actual, FG_available_total)
                shortage_t = max(0, D_actual - FG_available_total)
                I_fg_t = max(0, FG_available_total - sales_t)
                
                # Costs
                c_fix = self._get_param_value(params.c_fix, date, 0.0)
                c_base = self._get_param_value(params.c_base, date, 0.0)
                c_flex = self._get_param_value(params.c_flex, date, 0.0)
                f_opt = self._get_param_value(params.f_opt, date, 0.0)
                h_rm = self._get_param_value(params.h_rm, date, 0.0)
                h_fg = self._get_param_value(params.h_fg, date, 0.0)
                if params.pen_short_pct is not None:
                    pen_short = params.pen_short_pct * p_sell_actual
                else:
                    pen_short = self._get_param_value(params.pen_short, date, 0.0)

                fixed_cost = c_fix * x_fix_t
                framework_fee = f_opt * x_opt_t
                framework_cost = c_opt_t * y_opt_t
                spot_cost = c_spot_actual * x_spot_t
                base_prod_cost = c_base * actual_P_base
                flex_prod_cost = c_flex * P_flex_t
                rm_holding = h_rm * I_rm_t
                fg_holding = h_fg * I_fg_t
                shortage_penalty = pen_short * shortage_t

                total_cost = (fixed_cost + framework_fee + framework_cost + spot_cost +
                              base_prod_cost + flex_prod_cost + rm_holding + fg_holding +
                              shortage_penalty)

                # Revenue and profit
                revenue = p_sell_actual * sales_t
                profit = revenue - total_cost
                
                # Store results
                scenario_profits.loc[date, s] = profit
                stage2_y_opt.loc[date, s] = y_opt_t
                stage2_x_spot.loc[date, s] = x_spot_t
                stage2_P_flex.loc[date, s] = P_flex_t
                stage2_I_rm.loc[date, s] = I_rm_t
                stage2_I_fg.loc[date, s] = I_fg_t
                stage2_S.loc[date, s] = sales_t
                stage2_U.loc[date, s] = shortage_t
                
                # Update inventory state
                I_rm_prev = I_rm_t
                I_fg_prev = I_fg_t
        
        # ---------------------------------------------------------------------
        # STEP 5: Compute risk metrics
        # ---------------------------------------------------------------------
        logger.debug("  Computing risk metrics...")
        
        total_profits = scenario_profits.astype(float).sum(axis=0)
        risk_metrics = self._compute_risk_metrics(total_profits, prob)
        
        # Compute objective value (expected profit, no risk adjustment)
        objective_value = risk_metrics['Expected_Profit']
        
        # ---------------------------------------------------------------------
        # STEP 6: Build cost components for plotting
        # ---------------------------------------------------------------------
        # Average costs across scenarios for each category
        cost_components = {}
        
        # Recompute average costs for plotting
        for cost_name, compute_func in [
            ('Fixed_Procurement', lambda d, s: self._get_param_value(params.c_fix, d, 0) * decisions.loc[d, 'x_fix']),
            ('Reservation_Fee', lambda d, s: self._get_param_value(params.f_opt, d, 0) * decisions.loc[d, 'x_opt']),
            ('Base_Production', lambda d, s: self._get_param_value(params.c_base, d, 0) * decisions.loc[d, 'P_base']),
        ]:
            total = sum(compute_func(d, None) for d in dates)
            cost_components[cost_name] = total
        
        # Scenario-dependent costs (average)
        for cost_name, stage2_df, param_getter in [
            ('Framework', stage2_y_opt, lambda d, s: min(max(
                scenarios[(scenarios['Date'] == d) & (scenarios['Scenario'] == s)]['c_spot'].iloc[0] +
                self._get_param_value(params.basis_opt, d, 0),
                self._get_param_value(params.floor_opt, d, 0)
            ), self._get_param_value(params.cap_opt, d, float('inf')))),
            ('Spot', stage2_x_spot, lambda d, s: scenarios[(scenarios['Date'] == d) & (scenarios['Scenario'] == s)]['c_spot'].iloc[0]),
            ('Flex_Production', stage2_P_flex, lambda d, s: self._get_param_value(params.c_flex, d, 0)),
            ('RM_Holding', stage2_I_rm, lambda d, s: self._get_param_value(params.h_rm, d, 0)),
            ('FG_Holding', stage2_I_fg, lambda d, s: self._get_param_value(params.h_fg, d, 0)),
            ('Shortage_Penalty', stage2_U, lambda d, s: self._get_param_value(params.pen_short, d, 0)),
        ]:
            total = 0.0
            for s in scenario_ids:
                for d in dates:
                    qty = float(stage2_df.loc[d, s])
                    unit_cost = param_getter(d, s)
                    total += qty * unit_cost * float(prob.get(s, 1/n_scenarios))
            cost_components[cost_name] = total
        
        # ---------------------------------------------------------------------
        # STEP 7: Build result
        # ---------------------------------------------------------------------
        logger.info(f"[✓] Safety stock policy applied. Expected Profit: €{objective_value:,.0f}")
        
        stage2_results = {
            'y_opt': stage2_y_opt,
            'x_spot': stage2_x_spot,
            'P_flex': stage2_P_flex,
            'I_rm': stage2_I_rm,
            'I_fg': stage2_I_fg,
            'S': stage2_S,
            'U': stage2_U,
        }
        
        metadata = {
            'model': self.model_name,
            'policy': {
                'service_level': policy.service_level,
                'fixed_pct': policy.fixed_pct,
                'framework_pct': policy.framework_pct,
                'production_smoothing': policy.production_smoothing,
                'SS_FG': SS_FG,
                'SS_RM': SS_RM,
            },
            'n_scenarios': n_scenarios,
            'horizon': T,
            'cost_components': cost_components,
        }
        
        result = OptimizationResult(
            decisions=decisions,
            scenario_profits=scenario_profits,
            probabilities=prob,
            objective_value=objective_value,
            risk_metrics=risk_metrics,
            stage2_results=stage2_results,
            metadata=metadata,
        )
        
        self._last_result = result
        return result
    
    def backtest(
        self,
        decisions: "pd.DataFrame | OptimizationResult" = None,
        actual_data: pd.DataFrame = None,
        params: ModelParameters = None,
        policy: SafetyStockParams = None,
        **kwargs,
    ) -> BacktestResult:
        """
        Evaluate safety stock policy decisions against realized data.
        
        Parameters
        ----------
        decisions : pd.DataFrame or OptimizationResult
            First-stage decisions or the full OptimizationResult.
        actual_data : pd.DataFrame
            Historical data with columns ['D', 'c_spot', 'p_sell']
            and DatetimeIndex.
        params : ModelParameters
            Model parameters.
        policy : SafetyStockParams, optional
            Policy parameters (used for metadata).
        **kwargs
            Additional arguments.
            
        Returns
        -------
        BacktestResult
            Container with timeline, metrics, and optional scenario comparison.
        """
        if params is None:
            params = kwargs.get("params")
        if policy is None:
            policy = kwargs.get("policy", SafetyStockParams())
        if params is None:
            raise ValueError("params (ModelParameters) is required")
        
        # Accept OptimizationResult directly
        opt_result: Optional[OptimizationResult] = None
        if isinstance(decisions, OptimizationResult):
            opt_result = decisions
            decisions = opt_result.decisions
        elif decisions is None and self._last_result is not None:
            opt_result = self._last_result
            decisions = opt_result.decisions
        
        logger = self._setup_logger()
        logger.info(f"[{self.model_name}] Running backtest...")
        
        # Input validation
        required_decision_cols = ['x_fix', 'x_opt', 'P_base']
        missing_decision_cols = [c for c in required_decision_cols if c not in decisions.columns]
        if missing_decision_cols:
            raise ValueError(f"decisions DataFrame missing: {missing_decision_cols}")
        
        required_data_cols = ['D', 'c_spot', 'p_sell']
        missing_data_cols = [c for c in required_data_cols if c not in actual_data.columns]
        if missing_data_cols:
            raise ValueError(f"actual_data missing: {missing_data_cols}")
        
        decisions = decisions.copy()
        actual_future_data = actual_data.copy()
        
        if not isinstance(decisions.index, pd.DatetimeIndex):
            decisions.index = pd.to_datetime(decisions.index)
        if not isinstance(actual_future_data.index, pd.DatetimeIndex):
            actual_future_data.index = pd.to_datetime(actual_future_data.index)
        
        common_dates = decisions.index.intersection(actual_future_data.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between decisions and actual_data")
        
        decisions_period = decisions.loc[common_dates].sort_index()
        actual_period = actual_future_data.loc[common_dates].sort_index()
        
        # Track inventory state
        I_rm_prev = params.I_rm0
        I_fg_prev = params.I_fg0
        
        monthly_results = []
        
        for date in common_dates:
            # First-stage decisions
            x_fix_t = float(decisions_period.loc[date, 'x_fix'])
            x_opt_t = float(decisions_period.loc[date, 'x_opt'])
            P_base_t = float(decisions_period.loc[date, 'P_base'])
            
            # Realized values
            D_t = float(actual_period.loc[date, 'D'])
            c_spot_t = float(actual_period.loc[date, 'c_spot'])
            p_sell_t = float(actual_period.loc[date, 'p_sell'])
            
            # Framework exercise price
            basis = self._get_param_value(params.basis_opt, date, 0.0)
            floor_val = self._get_param_value(params.floor_opt, date, 0.0)
            cap_val = self._get_param_value(params.cap_opt, date, float('inf'))
            c_opt_t = min(max(c_spot_t + basis, floor_val), cap_val)
            
            # Exercise decision
            if c_opt_t <= c_spot_t:
                y_opt_t = x_opt_t
            else:
                y_opt_t = 0.0
            
            # RM contracted inflows
            RM_contracted = x_fix_t + y_opt_t
            
            # Determine full production plan first, then buy spot to cover
            Cap_flex = self._get_param_value(params.Cap_flex, date, 0.0)
            
            # Flex: produce extra if demand exceeds base output + FG inventory
            FG_available_est = I_fg_prev + P_base_t
            demand_gap = max(0, D_t - FG_available_est)
            P_flex_t = min(demand_gap, Cap_flex)
            
            # Total RM needed for planned production
            total_RM_need = params.alpha * (P_base_t + P_flex_t)
            RM_available = I_rm_prev + RM_contracted
            
            # Spot purchases to cover RM gap
            x_spot_t = max(0, total_RM_need - RM_available)
            
            # With spot purchased, check actual RM available
            RM_total_available = RM_available + x_spot_t
            actual_P_base = min(P_base_t, RM_total_available / params.alpha)
            P_flex_t = min(P_flex_t, max(0, RM_total_available / params.alpha - actual_P_base))
            
            # Inventory updates
            RM_used = params.alpha * (actual_P_base + P_flex_t)
            I_rm_t = max(0, I_rm_prev + RM_contracted + x_spot_t - RM_used)
            
            FG_produced = actual_P_base + P_flex_t
            FG_available_total = I_fg_prev + FG_produced
            sales_t = min(D_t, FG_available_total)
            shortage_t = max(0, D_t - FG_available_total)
            I_fg_t = max(0, FG_available_total - sales_t)
            
            # Costs
            c_fix = self._get_param_value(params.c_fix, date, 0.0)
            c_base = self._get_param_value(params.c_base, date, 0.0)
            c_flex = self._get_param_value(params.c_flex, date, 0.0)
            f_opt = self._get_param_value(params.f_opt, date, 0.0)
            h_rm = self._get_param_value(params.h_rm, date, 0.0)
            h_fg = self._get_param_value(params.h_fg, date, 0.0)
            if params.pen_short_pct is not None:
                pen_short = params.pen_short_pct * p_sell_t
            else:
                pen_short = self._get_param_value(params.pen_short, date, 0.0)

            fixed_cost = c_fix * x_fix_t
            framework_fee = f_opt * x_opt_t
            framework_cost = c_opt_t * y_opt_t
            spot_cost = c_spot_t * x_spot_t
            base_prod_cost = c_base * actual_P_base
            flex_prod_cost = c_flex * P_flex_t
            rm_holding = h_rm * I_rm_t
            fg_holding = h_fg * I_fg_t
            shortage_penalty = pen_short * shortage_t

            total_cost = (fixed_cost + framework_fee + framework_cost + spot_cost +
                          base_prod_cost + flex_prod_cost + rm_holding + fg_holding +
                          shortage_penalty)

            revenue = p_sell_t * sales_t
            profit = revenue - total_cost
            
            monthly_results.append({
                'Date': date,
                'Demand_Actual': D_t,
                'c_spot_Actual': c_spot_t,
                'p_sell_Actual': p_sell_t,
                'x_fix': x_fix_t,
                'x_opt': x_opt_t,
                'y_opt': y_opt_t,
                'x_spot': x_spot_t,
                'P_base': actual_P_base,
                'P_flex': P_flex_t,
                'Production': actual_P_base + P_flex_t,
                'I_rm': I_rm_t,
                'I_fg': I_fg_t,
                'Sales': sales_t,
                'Shortage': shortage_t,
                'Fixed_Procurement_Costs': fixed_cost,
                'Reservation_Fee_Costs': framework_fee,
                'Framework_Costs': framework_cost,
                'Spot_Costs': spot_cost,
                'Base_Production_Costs': base_prod_cost,
                'Flex_Production_Costs': flex_prod_cost,
                'RM_Holding_Costs': rm_holding,
                'FG_Holding_Costs': fg_holding,
                'Shortage_Penalty_Costs': shortage_penalty,
                'Total_Costs': total_cost,
                'Revenue': revenue,
                'Profit': profit,
                'Capacity_Utilization': (
                    (actual_P_base + P_flex_t) /
                    (self._get_param_value(params.Cap_base, date, 1) + Cap_flex)
                    if (self._get_param_value(params.Cap_base, date, 1) + Cap_flex) > 0
                    else 0),
                'Demand_Satisfaction': sales_t / D_t if D_t > 0 else 1.0,
            })
            
            I_rm_prev = I_rm_t
            I_fg_prev = I_fg_t
        
        timeline = pd.DataFrame(monthly_results).set_index('Date')
        
        # Metrics
        total_profit = float(timeline['Profit'].sum())
        total_revenue = float(timeline['Revenue'].sum())
        total_costs = float(timeline['Total_Costs'].sum())
        avg_fill_rate = float((timeline['Sales'] / timeline['Demand_Actual'].replace(0, np.nan)).mean())
        total_shortage = float(timeline['Shortage'].sum())
        
        # Scenario comparison
        scenario_comparison = None
        if opt_result is not None:
            scenario_total_profits = opt_result.scenario_profits.sum(axis=0)
            scenario_comparison = self._compare_to_scenarios(
                total_profit,
                scenario_total_profits,
                opt_result.probabilities,
            )
        
        metrics = {
            'Total_Profit': total_profit,
            'Total_Revenue': total_revenue,
            'Total_Costs': total_costs,
            'Profit_Margin': total_profit / total_revenue if total_revenue > 0 else 0,
            'Avg_Fill_Rate': avg_fill_rate,
            'Total_Shortage': total_shortage,
            'Periods': len(timeline),
        }
        
        logger.info(f"[✓] Backtest complete. Total Profit: €{total_profit:,.0f}")
        
        return BacktestResult(
            timeline=timeline,
            decisions=decisions_period,
            metrics=metrics,
            scenario_comparison=scenario_comparison,
        )
