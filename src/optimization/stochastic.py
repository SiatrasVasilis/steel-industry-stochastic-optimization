"""
Advanced Two-Stage Stochastic Tactical Planning Model for Steel Industry

This module implements a comprehensive two-stage stochastic program for tactical planning
in steel production, featuring:
- Multiple procurement instruments (fixed contracts, framework/call-off options, spot purchases)
- Two-tier capacity model (base + flex production)
- Dual inventory buffers (raw material + finished goods)
- Yield conversion factor
- Scenario-dependent pricing with bounded framework exercise prices

Mathematical formulation based on: procurement_model_form.md
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import logging
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

from .base import BaseStochasticModel
from .results import OptimizationResult, BacktestResult

if TYPE_CHECKING:
    from ..params.risk import RiskProfile

try:
    from ..params.risk import RiskProfile as _RiskProfile
except ImportError:
    _RiskProfile = None


@dataclass
class ModelParameters:
    """
    Container for all deterministic model parameters.
    
    Attributes
    ----------
    # Procurement & Contracts
    c_fix : dict or float
        Fixed contract RM unit cost [€/Tn RM] - can be per period {t: cost} or scalar
    x_fix_max : dict or float
        Max fixed contract volume [Tn RM] per period
    x_opt_max : dict or float
        Max reserved call-off volume [Tn RM] per period
    f_opt : dict or float
        Reservation fee per reserved ton [€/Tn reserved] (can be 0)
    
    # Framework pricing parameters
    basis_opt : dict or float
        Premium/discount vs spot for call-off (can be 0)
    floor_opt : dict or float
        Lower bound for call-off exercise price
    cap_opt : dict or float
        Upper bound for call-off exercise price
    
    # Capacity
    Cap_base : dict or float
        Base capacity [Tn FG] per period
    Cap_flex : dict or float
        Flex capacity [Tn FG] per period
    c_base : dict or float
        Base production variable cost [€/Tn FG]
    c_flex : dict or float
        Flex production variable cost [€/Tn FG] (>= c_base)
    
    # Inventory & Conversion
    alpha : float
        RM required per ton FG production [Tn RM / Tn FG]
    h_rm : dict or float
        RM holding cost [€/Tn RM-period]
    h_fg : dict or float
        FG holding cost [€/Tn FG-period]
    I_rm0 : float
        Initial RM inventory [Tn RM]
    I_fg0 : float
        Initial FG inventory [Tn FG]
    
    # Service
    pen_short : dict or float
        Unmet demand penalty [€/Tn FG]
    
    # Optional constraints
    I_rm_max : dict or float or None
        Max RM inventory capacity (None = unlimited)
    I_fg_max : dict or float or None
        Max FG inventory capacity (None = unlimited)
    spot_max : dict or None
        Max spot availability per (t, w) (None = unlimited)
    """
    # Procurement & Contracts
    c_fix: Union[Dict, float] = 125.0
    x_fix_max: Union[Dict, float] = 40_000.0
    x_opt_max: Union[Dict, float] = 20_000.0
    f_opt: Union[Dict, float] = 5.0
    
    # Framework pricing
    basis_opt: Union[Dict, float] = -3.0
    floor_opt: Union[Dict, float] = 100.0
    cap_opt: Union[Dict, float] = 200.0
    
    # Capacity
    Cap_base: Union[Dict, float] = 42_000.0
    Cap_flex: Union[Dict, float] = 10_000.0
    c_base: Union[Dict, float] = 50.0
    c_flex: Union[Dict, float] = 80.0
    
    # Inventory & Conversion
    alpha: float = 1.05
    h_rm: Union[Dict, float] = 1.8
    h_fg: Union[Dict, float] = 5.0
    I_rm0: float = 10_000.0
    I_fg0: float = 30_000.0
    
    # Service
    pen_short: Union[Dict, float] = 600.0
    pen_short_pct: Optional[float] = None  # If set, penalty = pen_short_pct * p_sell (overrides pen_short)

    # Optional constraints
    I_rm_max: Optional[Union[Dict, float]] = None
    I_fg_max: Optional[Union[Dict, float]] = None
    spot_max: Optional[Dict] = None


class StochasticOptimizationModel(BaseStochasticModel):
    """
    Two-Stage Stochastic Tactical Planning Model for Steel Industry.

    Implements a comprehensive stochastic programming model with:
    - Stage 1 (here-and-now): Fixed contracts, framework reservations, base production
    - Stage 2 (recourse): Framework exercise, spot purchases, flex production, inventories

    The model maximizes expected profit under demand, price, and cost uncertainty.

    Inherits from BaseStochasticModel and implements the optimize() and
    backtest() methods.

    Required Variables
    ------------------
    D      : Demand [Tn FG]
    c_spot : Spot RM unit cost [€/Tn RM]
    p_sell : Selling price [€/Tn FG]

    Parameters
    ----------
    solver : str, default "highs"
        Optimization solver to use ("highs", "gurobi", "cplex", "glpk")
    """

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "AdvancedTacticalPlanning"

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
            "sell_price": "p_sell",
            "Sell_Price": "p_sell",
            "steel_price": "p_sell",
            "Steel_Price": "p_sell",
        }

    def __init__(self, solver: str = "highs"):
        """Initialize the StochasticOptimizationModel."""
        super().__init__(solver=solver)
        self._last_result: Optional[OptimizationResult] = None
        self._pyomo_model: Optional[Any] = None

    @staticmethod
    def _get_param_value(param: Union[Dict, float], key, default: float = 0.0) -> float:
        """Helper to extract parameter value (supports dict or scalar)."""
        if isinstance(param, dict):
            return param.get(key, default)
        return float(param)
    
    @staticmethod
    def compute_framework_exercise_price(c_spot: Dict, basis_opt: Union[Dict, float],
                                         floor_opt: Union[Dict, float], 
                                         cap_opt: Union[Dict, float],
                                         T: list, W: list) -> Dict:
        """
        Compute the scenario-dependent framework exercise price.
        
        c_opt[t,w] = min(max(c_spot[t,w] + basis_opt[t], floor_opt[t]), cap_opt[t])
        
        Parameters
        ----------
        c_spot : dict
            Spot prices indexed by (t, w)
        basis_opt, floor_opt, cap_opt : dict or float
            Framework pricing parameters
        T : list
            Time periods
        W : list
            Scenarios
            
        Returns
        -------
        dict
            Framework exercise prices indexed by (t, w)
        """
        c_opt = {}
        for t in T:
            basis = StochasticOptimizationModel._get_param_value(basis_opt, t, 0.0)
            floor_val = StochasticOptimizationModel._get_param_value(floor_opt, t, 0.0)
            cap_val = StochasticOptimizationModel._get_param_value(cap_opt, t, float('inf'))
            
            for w in W:
                spot = c_spot[(t, w)]
                # Apply pricing rule: min(max(spot + basis, floor), cap)
                price = min(max(spot + basis, floor_val), cap_val)
                c_opt[(t, w)] = price
        
        return c_opt
    
    def optimize(
        self,
        scenarios: pd.DataFrame,
        prob: pd.Series,
        params: ModelParameters = None,
        risk_profile: Optional["RiskProfile"] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Build and solve the two-stage stochastic tactical planning model.
        
        This method implements the full formulation from procurement_model_form.md with:
        - Three procurement instruments (fixed, framework/call-off, spot)
        - Two-tier production (base + flex capacity)
        - Dual inventory buffers (RM + FG)
        - Yield conversion factor (alpha)
        - Bounded framework exercise pricing
        
        Parameters
        ----------
        scenarios : pd.DataFrame
            Long-form DataFrame with scenario data containing columns:
            ``'Date'``, ``'Scenario'``, ``'D'``, ``'c_spot'``, ``'p_sell'``.
        prob : pd.Series
            Scenario probabilities indexed by scenario ID. Must sum to 1.0.
        params : ModelParameters
            All deterministic model parameters (see ModelParameters class).
        risk_profile : RiskProfile, optional
            Risk preferences (risk_aversion, cvar_alpha).
            Default: risk-neutral (λ=0).
        **kwargs
            Accepts ``params`` and ``risk_profile`` as keyword arguments
            for compatibility with the ``run()`` interface.
            
        Returns
        -------
        OptimizationResult
            Contains decisions, scenario profits, risk metrics, and
            stage-2 recourse results.
            
        Mathematical Formulation
        ------------------------
        **Objective:** Maximize expected profit
        
        max Σ_w p_w Σ_t [
            p_sell[t,w] * S[t,w]
            - c_fix[t] * x_fix[t]
            - f_opt[t] * x_opt[t]
            - c_opt[t,w] * y_opt[t,w]
            - c_spot[t,w] * x_spot[t,w]
            - c_base[t] * P_base[t]
            - c_flex[t] * P_flex[t,w]
            - h_rm[t] * I_rm[t,w]
            - h_fg[t] * I_fg[t,w]
            - pen_short[t] * U[t,w]
        ]
        
        **Key Constraints:**
        1. Contract bounds: x_fix[t] <= x_fix_max[t], x_opt[t] <= x_opt_max[t]
        2. Option exercise: y_opt[t,w] <= x_opt[t]
        3. Capacity: P_base[t] <= Cap_base[t], P_flex[t,w] <= Cap_flex[t]
        4. RM balance: I_rm[t,w] = I_rm[t-1,w] + inflows - alpha*P[t,w]
        5. FG balance: I_fg[t,w] = I_fg[t-1,w] + P[t,w] - S[t,w]
        6. Demand: S[t,w] + U[t,w] = D[t,w]
        """
        # Allow params/risk_profile via **kwargs (from run() forwarding)
        if params is None:
            params = kwargs.get("params")
        if params is None:
            raise ValueError("params (ModelParameters) is required")
        if risk_profile is None:
            risk_profile = kwargs.get("risk_profile")

        # Extract risk parameters
        if risk_profile is not None:
            risk_aversion = risk_profile.risk_aversion
            cvar_alpha = risk_profile.cvar_alpha
        else:
            risk_aversion = 0.0
            cvar_alpha = 0.05

        logger = self._setup_logger()
        
        logger.debug("=" * 60)
        logger.debug("ADVANCED TWO-STAGE STOCHASTIC TACTICAL PLANNING")
        logger.debug("=" * 60)
        
        # ================================================================
        # INPUT VALIDATION
        # ================================================================
        logger.debug("Validating inputs...")
        
        self.validate_scenario_inputs(scenarios, prob)
        
        df = scenarios.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Scenario"] = df["Scenario"].astype(str)
        
        prob = prob.copy()
        prob.index = prob.index.astype(str)
        
        T = sorted(df["Date"].unique())
        W = sorted(df["Scenario"].unique())
        
        missing_probs = [w for w in W if w not in prob.index]
        if missing_probs:
            raise ValueError(f"Missing probabilities for scenarios: {missing_probs[:10]}")
        
        logger.info(f"[✓] Input validation completed!")
        logger.debug(f"    Time periods: {len(T)}")
        logger.debug(f"    Scenarios: {len(W)}")
        
        # ================================================================
        # BUILD PARAMETER DICTIONARIES
        # ================================================================
        logger.debug("Building parameter dictionaries...")
        
        df2 = df.set_index(["Date", "Scenario"]).sort_index()
        
        # Scenario-indexed parameters
        D = {}       # Demand [Tn FG]
        c_spot = {}  # Spot RM cost [€/Tn RM]
        p_sell = {}  # Selling price [€/Tn FG]
        
        for t in T:
            for w in W:
                try:
                    row = df2.loc[(t, w)]
                    D[(t, w)] = float(row["D"])
                    c_spot[(t, w)] = float(row["c_spot"])
                    p_sell[(t, w)] = float(row["p_sell"])
                except KeyError as e:
                    raise ValueError(f"Missing data for (Date={t}, Scenario={w})") from e
        
        p = {w: float(prob.loc[w]) for w in W}
        
        # Compute framework exercise price (scenario-dependent)
        c_opt = self.compute_framework_exercise_price(
            c_spot, params.basis_opt, params.floor_opt, params.cap_opt, T, W
        )
        
        logger.info(f"[✓] Parameter dictionaries built!")
        logger.debug(f"    Average demand: {np.mean(list(D.values())):.1f} Tn")
        logger.debug(f"    Average spot price: €{np.mean(list(c_spot.values())):.1f}/Tn")
        logger.debug(f"    Average selling price: €{np.mean(list(p_sell.values())):.1f}/Tn")
        
        # ================================================================
        # BUILD PYOMO MODEL
        # ================================================================
        logger.debug("Building optimization model...")
        
        m = pyo.ConcreteModel(name="StochasticOptimizationModel")
        
        # ----- SETS -----
        m.T = pyo.Set(initialize=T, ordered=True, doc="Time periods")
        m.W = pyo.Set(initialize=W, ordered=True, doc="Scenarios")
        
        # ----- FIRST-STAGE VARIABLES (here-and-now) -----
        m.x_fix = pyo.Var(m.T, domain=pyo.NonNegativeReals, 
                         doc="Fixed RM procurement [Tn RM]")
        m.x_opt = pyo.Var(m.T, domain=pyo.NonNegativeReals, 
                         doc="Reserved framework RM volume [Tn RM]")
        m.P_base = pyo.Var(m.T, domain=pyo.NonNegativeReals, 
                          doc="Base production [Tn FG]")
        
        # ----- SECOND-STAGE VARIABLES (recourse) -----
        m.y_opt = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                         doc="Exercised framework RM volume [Tn RM]")
        m.x_spot = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                          doc="Spot RM purchases [Tn RM]")
        m.P_flex = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                          doc="Flex production [Tn FG]")
        m.I_rm = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                        doc="RM inventory end of period [Tn RM]")
        m.I_fg = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                        doc="FG inventory end of period [Tn FG]")
        m.S = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                     doc="Sales served [Tn FG]")
        m.U = pyo.Var(m.T, m.W, domain=pyo.NonNegativeReals, 
                     doc="Unmet demand [Tn FG]")
        
        # Helper: Total production
        def total_production_rule(m, t, w):
            return m.P_base[t] + m.P_flex[t, w]
        m.P_total = pyo.Expression(m.T, m.W, rule=total_production_rule,
                                   doc="Total production = Base + Flex")
        
        # CVaR variables (if risk-averse)
        if risk_aversion > 0:
            m.VaR = pyo.Var(domain=pyo.Reals, doc="Value-at-Risk threshold")
            m.z = pyo.Var(m.W, domain=pyo.NonNegativeReals, doc="CVaR shortfall")
        
        logger.debug(f"    Variables defined")
        
        # ================================================================
        # CONSTRAINTS
        # ================================================================
        logger.debug("Adding constraints...")
        
        # Helper function for parameter extraction
        def get_param(param, t, default=0.0):
            return self._get_param_value(param, t, default)
        
        # ----- (1) Contract bounds (first stage) -----
        def fixed_contract_bound_rule(m, t):
            return m.x_fix[t] <= get_param(params.x_fix_max, t, float('inf'))
        m.FixedContractBound = pyo.Constraint(m.T, rule=fixed_contract_bound_rule,
                                              doc="Fixed contract volume bound")
        
        def framework_reservation_bound_rule(m, t):
            return m.x_opt[t] <= get_param(params.x_opt_max, t, float('inf'))
        m.FrameworkReservationBound = pyo.Constraint(m.T, rule=framework_reservation_bound_rule,
                                                     doc="Framework reservation bound")
        
        # ----- (2) Option exercise rule -----
        def option_exercise_rule(m, t, w):
            return m.y_opt[t, w] <= m.x_opt[t]
        m.OptionExerciseRule = pyo.Constraint(m.T, m.W, rule=option_exercise_rule,
                                              doc="Can only exercise what was reserved")
        
        # ----- (3) Capacity constraints -----
        def base_capacity_rule(m, t):
            return m.P_base[t] <= get_param(params.Cap_base, t, float('inf'))
        m.BaseCapacityConstraint = pyo.Constraint(m.T, rule=base_capacity_rule,
                                                  doc="Base production capacity limit")
        
        def flex_capacity_rule(m, t, w):
            return m.P_flex[t, w] <= get_param(params.Cap_flex, t, float('inf'))
        m.FlexCapacityConstraint = pyo.Constraint(m.T, m.W, rule=flex_capacity_rule,
                                                  doc="Flex production capacity limit")
        
        # ----- (4) RM inventory balance -----
        T_list = list(m.T)
        
        def rm_inventory_balance_rule(m, t, w):
            t_idx = T_list.index(t)
            
            # Inflows
            inflow = m.x_fix[t] + m.y_opt[t, w] + m.x_spot[t, w]
            
            # Outflow (consumption)
            outflow = params.alpha * m.P_total[t, w]
            
            if t_idx == 0:
                # First period: use initial inventory
                return m.I_rm[t, w] == params.I_rm0 + inflow - outflow
            else:
                # Subsequent periods: use previous period inventory
                t_prev = T_list[t_idx - 1]
                return m.I_rm[t, w] == m.I_rm[t_prev, w] + inflow - outflow
        
        m.RMInventoryBalance = pyo.Constraint(m.T, m.W, rule=rm_inventory_balance_rule,
                                              doc="Raw material inventory balance")
        
        # ----- (5) FG inventory balance -----
        def fg_inventory_balance_rule(m, t, w):
            t_idx = T_list.index(t)
            
            if t_idx == 0:
                return m.I_fg[t, w] == params.I_fg0 + m.P_total[t, w] - m.S[t, w]
            else:
                t_prev = T_list[t_idx - 1]
                return m.I_fg[t, w] == m.I_fg[t_prev, w] + m.P_total[t, w] - m.S[t, w]
        
        m.FGInventoryBalance = pyo.Constraint(m.T, m.W, rule=fg_inventory_balance_rule,
                                              doc="Finished goods inventory balance")
        
        # ----- (6) Demand accounting -----
        def demand_accounting_rule(m, t, w):
            return m.S[t, w] + m.U[t, w] == D[(t, w)]
        m.DemandAccounting = pyo.Constraint(m.T, m.W, rule=demand_accounting_rule,
                                            doc="Sales + Unmet = Demand")
        
        def sales_upper_bound_rule(m, t, w):
            return m.S[t, w] <= D[(t, w)]
        m.SalesUpperBound = pyo.Constraint(m.T, m.W, rule=sales_upper_bound_rule,
                                           doc="Sales cannot exceed demand")
        
        # ----- (7) Optional: Inventory capacity constraints -----
        if params.I_rm_max is not None:
            def rm_inventory_cap_rule(m, t, w):
                return m.I_rm[t, w] <= get_param(params.I_rm_max, t, float('inf'))
            m.RMInventoryCap = pyo.Constraint(m.T, m.W, rule=rm_inventory_cap_rule,
                                              doc="RM inventory capacity limit")
        
        if params.I_fg_max is not None:
            def fg_inventory_cap_rule(m, t, w):
                return m.I_fg[t, w] <= get_param(params.I_fg_max, t, float('inf'))
            m.FGInventoryCap = pyo.Constraint(m.T, m.W, rule=fg_inventory_cap_rule,
                                              doc="FG inventory capacity limit")
        
        if params.spot_max is not None:
            def spot_availability_rule(m, t, w):
                max_spot = params.spot_max.get((t, w), float('inf'))
                return m.x_spot[t, w] <= max_spot
            m.SpotAvailability = pyo.Constraint(m.T, m.W, rule=spot_availability_rule,
                                                doc="Spot market availability limit")
        
        logger.info(f"[✓] Constraints added!")
        
        # ================================================================
        # OBJECTIVE FUNCTION
        # ================================================================
        logger.debug("Building objective function...")
        
        # Profit expression for each scenario
        def scenario_profit_rule(m, w):
            profit = 0.0
            for t in m.T:
                # Revenue
                revenue = p_sell[(t, w)] * m.S[t, w]
                
                # First-stage costs (allocated to each scenario for CVaR)
                fixed_cost = get_param(params.c_fix, t) * m.x_fix[t]
                reservation_fee = get_param(params.f_opt, t) * m.x_opt[t]
                base_prod_cost = get_param(params.c_base, t) * m.P_base[t]
                
                # Second-stage costs
                framework_cost = c_opt[(t, w)] * m.y_opt[t, w]
                spot_cost = c_spot[(t, w)] * m.x_spot[t, w]
                flex_prod_cost = get_param(params.c_flex, t) * m.P_flex[t, w]
                rm_holding_cost = get_param(params.h_rm, t) * m.I_rm[t, w]
                fg_holding_cost = get_param(params.h_fg, t) * m.I_fg[t, w]
                if params.pen_short_pct is not None:
                    shortage_penalty = params.pen_short_pct * p_sell[(t, w)] * m.U[t, w]
                else:
                    shortage_penalty = get_param(params.pen_short, t) * m.U[t, w]

                profit += (revenue
                          - fixed_cost - reservation_fee - base_prod_cost
                          - framework_cost - spot_cost - flex_prod_cost
                          - rm_holding_cost - fg_holding_cost - shortage_penalty)
            
            return profit
        
        m.ScenarioProfit = pyo.Expression(m.W, rule=scenario_profit_rule,
                                          doc="Profit by scenario")
        
        if risk_aversion > 0:
            # CVaR shortfall constraint
            def cvar_shortfall_rule(m, w):
                return m.z[w] >= m.VaR - m.ScenarioProfit[w]
            m.CVaRShortfall = pyo.Constraint(m.W, rule=cvar_shortfall_rule,
                                             doc="CVaR shortfall constraint")
            
            # Risk-adjusted objective
            def objective_rule(m):
                expected_profit = sum(p[w] * m.ScenarioProfit[w] for w in m.W)
                cvar_term = m.VaR - (1.0 / cvar_alpha) * sum(p[w] * m.z[w] for w in m.W)
                return (1.0 - risk_aversion) * expected_profit + risk_aversion * cvar_term
            
            m.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize,
                                        doc="Maximize risk-adjusted expected profit")
            logger.debug(f"    Risk-averse objective: λ={risk_aversion}, α={cvar_alpha}")
        else:
            # Risk-neutral objective
            def objective_rule(m):
                return sum(p[w] * m.ScenarioProfit[w] for w in m.W)
            
            m.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize,
                                        doc="Maximize expected profit")
            logger.debug(f"    Risk-neutral objective")
        
        logger.info(f"[✓] Objective function built!")
        
        # ================================================================
        # SOLVE MODEL
        # ================================================================
        logger.debug(f"Solving with {self.solver} solver...")
        
        try:
            solver_obj = pyo.SolverFactory(self.solver)
            if not solver_obj.available():
                raise RuntimeError(f"Solver '{self.solver}' is not available")
            
            results = solver_obj.solve(m, tee=False)
            
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                logger.info(f"[✓] Optimization completed successfully!")
                obj_value = pyo.value(m.Objective)
                logger.debug(f"    Optimal objective: €{obj_value:,.0f}")
            elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
                raise RuntimeError("Model is infeasible. Check parameters and constraints.")
            elif results.solver.termination_condition == pyo.TerminationCondition.unbounded:
                raise RuntimeError("Model is unbounded. Check objective and constraints.")
            else:
                logger.warning(f"⚠ Solver terminated: {results.solver.termination_condition}")
                
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")
        
        # ================================================================
        # EXTRACT RESULTS
        # ================================================================
        logger.debug("Extracting optimal decisions...")
        
        # Stage 1 decisions
        x_fix_vals = {t: pyo.value(m.x_fix[t]) for t in m.T}
        x_opt_vals = {t: pyo.value(m.x_opt[t]) for t in m.T}
        P_base_vals = {t: pyo.value(m.P_base[t]) for t in m.T}
        
        decisions = pd.DataFrame({
            'x_fix': x_fix_vals,
            'x_opt': x_opt_vals,
            'P_base': P_base_vals
        })
        decisions.index = pd.Index(sorted(T), name='Date')
        
        obj_value = pyo.value(m.Objective)
        
        logger.info(f"[✓] Results extracted!")
        logger.debug(f"    Total fixed procurement: {sum(x_fix_vals.values()):.1f} Tn")
        logger.debug(f"    Total framework reservation: {sum(x_opt_vals.values()):.1f} Tn")
        logger.debug(f"    Average base production: {np.mean(list(P_base_vals.values())):.1f} Tn/period")
        
        # Stage 2 variables by scenario (T × S DataFrames)
        logger.debug("Extracting full recourse results...")
        
        y_opt_df = pd.DataFrame({
            w: {t: pyo.value(m.y_opt[t, w]) for t in m.T} for w in m.W
        })
        y_opt_df.index = decisions.index
        
        x_spot_df = pd.DataFrame({
            w: {t: pyo.value(m.x_spot[t, w]) for t in m.T} for w in m.W
        })
        x_spot_df.index = decisions.index
        
        P_flex_df = pd.DataFrame({
            w: {t: pyo.value(m.P_flex[t, w]) for t in m.T} for w in m.W
        })
        P_flex_df.index = decisions.index
        
        I_rm_df = pd.DataFrame({
            w: {t: pyo.value(m.I_rm[t, w]) for t in m.T} for w in m.W
        })
        I_rm_df.index = decisions.index
        
        I_fg_df = pd.DataFrame({
            w: {t: pyo.value(m.I_fg[t, w]) for t in m.T} for w in m.W
        })
        I_fg_df.index = decisions.index
        
        S_df = pd.DataFrame({
            w: {t: pyo.value(m.S[t, w]) for t in m.T} for w in m.W
        })
        S_df.index = decisions.index
        
        U_df = pd.DataFrame({
            w: {t: pyo.value(m.U[t, w]) for t in m.T} for w in m.W
        })
        U_df.index = decisions.index
        
        # Build scenario profits DataFrame (T × S) — per-period per-scenario
        scenario_profits_df = pd.DataFrame(index=decisions.index, columns=W, dtype=float)
        for w in W:
            for t in T:
                revenue = p_sell[(t, w)] * pyo.value(m.S[t, w])
                fixed_cost = get_param(params.c_fix, t) * pyo.value(m.x_fix[t])
                reservation_fee = get_param(params.f_opt, t) * pyo.value(m.x_opt[t])
                base_prod_cost = get_param(params.c_base, t) * pyo.value(m.P_base[t])
                framework_cost = c_opt[(t, w)] * pyo.value(m.y_opt[t, w])
                spot_cost = c_spot[(t, w)] * pyo.value(m.x_spot[t, w])
                flex_prod_cost = get_param(params.c_flex, t) * pyo.value(m.P_flex[t, w])
                rm_holding = get_param(params.h_rm, t) * pyo.value(m.I_rm[t, w])
                fg_holding = get_param(params.h_fg, t) * pyo.value(m.I_fg[t, w])
                shortage = get_param(params.pen_short, t) * pyo.value(m.U[t, w])
                
                period_profit = (revenue
                                 - fixed_cost - reservation_fee - base_prod_cost
                                 - framework_cost - spot_cost - flex_prod_cost
                                 - rm_holding - fg_holding - shortage)
                scenario_profits_df.loc[t, w] = period_profit
        
        # Stage 2 results dict
        stage2 = {
            'y_opt': y_opt_df,
            'x_spot': x_spot_df,
            'P_flex': P_flex_df,
            'I_rm': I_rm_df,
            'I_fg': I_fg_df,
            'S': S_df,
            'U': U_df,
        }
        
        # Risk metrics (using base class utility)
        total_profits = scenario_profits_df.sum(axis=0)
        risk_metrics = self._compute_risk_metrics(total_profits, prob)
        
        # Service level
        demand_df = pd.DataFrame({w: {t: D[(t, w)] for t in T} for w in W})
        demand_df.index = decisions.index
        service_level = (S_df.sum() / demand_df.sum())
        expected_service = float((service_level * pd.Series(p)).sum())

        # Compute average cost components (per-period) for plotting
        cost_components = pd.DataFrame(index=sorted(T))
        # First-stage costs (same across scenarios)
        cost_components["Fixed_Procurement"] = pd.Series({t: get_param(params.c_fix, t) * pyo.value(m.x_fix[t]) for t in T})
        cost_components["Reservation_Fee"] = pd.Series({t: get_param(params.f_opt, t) * pyo.value(m.x_opt[t]) for t in T})
        cost_components["Base_Production"] = pd.Series({t: get_param(params.c_base, t) * pyo.value(m.P_base[t]) for t in T})

        # Scenario-dependent (second-stage) costs: average across scenarios
        framework_df = pd.DataFrame({w: {t: c_opt[(t, w)] * pyo.value(m.y_opt[t, w]) for t in T} for w in W})
        cost_components["Framework"] = framework_df.mean(axis=1)

        spot_df = pd.DataFrame({w: {t: c_spot[(t, w)] * pyo.value(m.x_spot[t, w]) for t in T} for w in W})
        cost_components["Spot"] = spot_df.mean(axis=1)

        flex_df = pd.DataFrame({w: {t: get_param(params.c_flex, t) * pyo.value(m.P_flex[t, w]) for t in T} for w in W})
        cost_components["Flex_Production"] = flex_df.mean(axis=1)

        rm_hold_df = pd.DataFrame({w: {t: get_param(params.h_rm, t) * pyo.value(m.I_rm[t, w]) for t in T} for w in W})
        cost_components["RM_Holding"] = rm_hold_df.mean(axis=1)

        fg_hold_df = pd.DataFrame({w: {t: get_param(params.h_fg, t) * pyo.value(m.I_fg[t, w]) for t in T} for w in W})
        cost_components["FG_Holding"] = fg_hold_df.mean(axis=1)

        if params.pen_short_pct is not None:
            shortage_df = pd.DataFrame({w: {t: params.pen_short_pct * p_sell[(t, w)] * pyo.value(m.U[t, w]) for t in T} for w in W})
        else:
            shortage_df = pd.DataFrame({w: {t: get_param(params.pen_short, t) * pyo.value(m.U[t, w]) for t in T} for w in W})
        cost_components["Shortage_Penalty"] = shortage_df.mean(axis=1)

        # Build result object
        result = OptimizationResult(
            decisions=decisions,
            scenario_profits=scenario_profits_df,
            probabilities=prob,
            objective_value=obj_value,
            risk_metrics=risk_metrics,
            stage2_results=stage2,
            metadata={
                'model': self.model_name,
                'solver': self.solver,
                'params': {f.name: getattr(params, f.name) for f in params.__dataclass_fields__.values()},
                'risk_profile': risk_profile.to_dict() if risk_profile is not None else None,
                'risk_aversion': risk_aversion,
                'cvar_alpha': cvar_alpha,
                'n_scenarios': len(W),
                'horizon': len(T),
                'expected_service_level': expected_service,
                'demand': demand_df,
                'c_opt': pd.DataFrame({w: {t: c_opt[(t, w)] for t in T} for w in W}),
                'cost_components': cost_components.sum().to_dict(),  # Total costs per category
            },
        )
        
        self._last_result = result
        self._pyomo_model = m
        
        logger.info(f"[✓] Full results extracted!")
        logger.debug(f"    Expected profit: €{obj_value:,.0f}")
        logger.debug(f"    Expected service level: {expected_service:.1%}")
        
        return result

    # ====================================================================
    # BACKTEST
    # ====================================================================

    def backtest(
        self,
        decisions: "pd.DataFrame | OptimizationResult" = None,
        actual_data: pd.DataFrame = None,
        params: ModelParameters = None,
        **kwargs,
    ) -> BacktestResult:
        """
        Simulate the performance of Stage-1 decisions against realized data.

        For each month the method solves a single-period recourse problem
        that mirrors Stage 2 of the stochastic programme: given the fixed
        first-stage decisions (``x_fix``, ``x_opt``, ``P_base``), it
        optimises framework exercise, spot purchases, flex production,
        inventory, and sales.

        Parameters
        ----------
        decisions : pd.DataFrame or OptimizationResult
            First-stage decisions (``x_fix``, ``x_opt``, ``P_base``) or
            the full ``OptimizationResult`` (decisions are extracted
            automatically; scenario comparison is also computed).
        actual_data : pd.DataFrame
            Historical data with columns ``['D', 'c_spot', 'p_sell']``
            and DatetimeIndex.
        params : ModelParameters
            All deterministic model parameters.
        **kwargs
            Additional keyword arguments (for base class compatibility).

        Returns
        -------
        BacktestResult
            Container with timeline, metrics, comparison to scenarios,
            ``summary()``, ``plot()``, ``detailed_report()``.
        """
        # Allow params via **kwargs
        if params is None:
            params = kwargs.get("params")
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

        logger.debug("=" * 60)
        logger.debug("ADVANCED BACKTESTING SIMULATION")
        logger.debug("=" * 60)

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

        # Helper
        def get_param(param, t, default=0.0):
            return self._get_param_value(param, t, default)

        # Track inventory state across periods
        I_rm_prev = params.I_rm0
        I_fg_prev = params.I_fg0

        monthly_results = []
        T_list = list(common_dates)

        for date in T_list:
            x_fix_t = float(decisions_period.loc[date, 'x_fix'])
            x_opt_t = float(decisions_period.loc[date, 'x_opt'])
            P_base_t = float(decisions_period.loc[date, 'P_base'])
            D_t = float(actual_period.loc[date, 'D'])
            c_spot_t = float(actual_period.loc[date, 'c_spot'])
            p_sell_t = float(actual_period.loc[date, 'p_sell'])

            # Framework exercise price for this period (deterministic realised)
            basis = get_param(params.basis_opt, date, 0.0)
            floor_val = get_param(params.floor_opt, date, 0.0)
            cap_val = get_param(params.cap_opt, date, float('inf'))
            c_opt_t = min(max(c_spot_t + basis, floor_val), cap_val)

            try:
                m = pyo.ConcreteModel()

                # Stage 2 recourse variables
                m.y_opt = pyo.Var(domain=pyo.NonNegativeReals, doc="Exercised framework volume")
                m.x_spot = pyo.Var(domain=pyo.NonNegativeReals, doc="Spot purchases")
                m.P_flex = pyo.Var(domain=pyo.NonNegativeReals, doc="Flex production")
                m.I_rm = pyo.Var(domain=pyo.NonNegativeReals, doc="End RM inventory")
                m.I_fg = pyo.Var(domain=pyo.NonNegativeReals, doc="End FG inventory")
                m.S = pyo.Var(domain=pyo.NonNegativeReals, doc="Sales")
                m.U = pyo.Var(domain=pyo.NonNegativeReals, doc="Unmet demand")

                # Option exercise bound
                m.exercise_bound = pyo.Constraint(expr=m.y_opt <= x_opt_t)

                # Flex capacity bound
                m.flex_cap = pyo.Constraint(
                    expr=m.P_flex <= get_param(params.Cap_flex, date, float('inf'))
                )

                # Base capacity check (P_base is first-stage — enforce as a pre-check)
                cap_base_val = get_param(params.Cap_base, date, float('inf'))
                if P_base_t > cap_base_val:
                    logger.warning(
                        f"First-stage decision P_base={P_base_t} exceeds Cap_base={cap_base_val} for {date}. "
                        "Proceeding without adding a trivial constraint."
                    )

                # RM inventory balance
                total_production = P_base_t + m.P_flex
                inflow = x_fix_t + m.y_opt + m.x_spot
                m.rm_balance = pyo.Constraint(
                    expr=m.I_rm == I_rm_prev + inflow - params.alpha * total_production
                )

                # FG inventory balance
                m.fg_balance = pyo.Constraint(
                    expr=m.I_fg == I_fg_prev + total_production - m.S
                )

                # Demand accounting
                m.demand_acct = pyo.Constraint(expr=m.S + m.U == D_t)
                m.sales_bound = pyo.Constraint(expr=m.S <= D_t)

                # Optional constraints
                if params.I_rm_max is not None:
                    m.rm_cap = pyo.Constraint(
                        expr=m.I_rm <= get_param(params.I_rm_max, date, float('inf'))
                    )
                if params.I_fg_max is not None:
                    m.fg_cap = pyo.Constraint(
                        expr=m.I_fg <= get_param(params.I_fg_max, date, float('inf'))
                    )

                # Objective: maximise single-period profit
                def profit_rule(m):
                    revenue = p_sell_t * m.S
                    fixed_cost = get_param(params.c_fix, date) * x_fix_t
                    reservation_fee = get_param(params.f_opt, date) * x_opt_t
                    base_prod_cost = get_param(params.c_base, date) * P_base_t
                    framework_cost = c_opt_t * m.y_opt
                    spot_cost = c_spot_t * m.x_spot
                    flex_prod_cost = get_param(params.c_flex, date) * m.P_flex
                    rm_holding = get_param(params.h_rm, date) * m.I_rm
                    fg_holding = get_param(params.h_fg, date) * m.I_fg
                    if params.pen_short_pct is not None:
                        shortage = params.pen_short_pct * p_sell_t * m.U
                    else:
                        shortage = get_param(params.pen_short, date) * m.U
                    return (revenue
                            - fixed_cost - reservation_fee - base_prod_cost
                            - framework_cost - spot_cost - flex_prod_cost
                            - rm_holding - fg_holding - shortage)

                m.profit = pyo.Objective(rule=profit_rule, sense=pyo.maximize)

                solver_obj = pyo.SolverFactory(self.solver)
                results = solver_obj.solve(m, tee=False)

                if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    profit = pyo.value(m.profit)
                    y_opt_v = pyo.value(m.y_opt)
                    x_spot_v = pyo.value(m.x_spot)
                    P_flex_v = pyo.value(m.P_flex)
                    I_rm_v = pyo.value(m.I_rm)
                    I_fg_v = pyo.value(m.I_fg)
                    S_v = pyo.value(m.S)
                    U_v = pyo.value(m.U)
                else:
                    logger.warning(f"Solver non-optimal for {date}: "
                                   f"{results.solver.termination_condition}")
                    profit = -1e6
                    y_opt_v = x_spot_v = P_flex_v = 0.0
                    I_rm_v = I_rm_prev
                    I_fg_v = I_fg_prev
                    S_v = 0.0
                    U_v = D_t

            except Exception as e:
                logger.error(f"Simulation failed for {date}: {e}")
                profit = -1e6
                y_opt_v = x_spot_v = P_flex_v = 0.0
                I_rm_v = I_rm_prev
                I_fg_v = I_fg_prev
                S_v = 0.0
                U_v = D_t

            # Update rolling inventory for next period
            I_rm_prev = I_rm_v
            I_fg_prev = I_fg_v

            total_prod = P_base_t + P_flex_v

            monthly_results.append({
                'Date': date,
                'Profit': profit,
                'Revenue': p_sell_t * S_v,
                'Fixed_Procurement_Costs': get_param(params.c_fix, date) * x_fix_t,
                'Reservation_Fee_Costs': get_param(params.f_opt, date) * x_opt_t,
                'Framework_Costs': c_opt_t * y_opt_v,
                'Spot_Costs': c_spot_t * x_spot_v,
                'Base_Production_Costs': get_param(params.c_base, date) * P_base_t,
                'Flex_Production_Costs': get_param(params.c_flex, date) * P_flex_v,
                'RM_Holding_Costs': get_param(params.h_rm, date) * I_rm_v,
                'FG_Holding_Costs': get_param(params.h_fg, date) * I_fg_v,
                'Shortage_Penalty_Costs': (params.pen_short_pct * p_sell_t if params.pen_short_pct is not None else get_param(params.pen_short, date)) * U_v,
                # Operational
                'x_fix': x_fix_t,
                'x_opt': x_opt_t,
                'y_opt': y_opt_v,
                'x_spot': x_spot_v,
                'P_base': P_base_t,
                'P_flex': P_flex_v,
                'Total_Production': total_prod,
                'Production': total_prod,
                'I_rm': I_rm_v,
                'I_fg': I_fg_v,
                'Sales': S_v,
                'Unmet_Demand': U_v,
                # Market
                'Demand_Actual': D_t,
                'c_spot_Actual': c_spot_t,
                'p_sell_Actual': p_sell_t,
                'c_opt_Actual': c_opt_t,
                # Ratios
                'Capacity_Utilization': (total_prod /
                    (get_param(params.Cap_base, date, 1) + get_param(params.Cap_flex, date, 0))
                    if (get_param(params.Cap_base, date, 1) + get_param(params.Cap_flex, date, 0)) > 0
                    else 0),
                'Demand_Satisfaction': S_v / D_t if D_t > 0 else 1.0,
            })

        # Build timeline DataFrame
        timeline = pd.DataFrame(monthly_results).set_index('Date')

        # Scalar metrics
        cost_cols = [c for c in timeline.columns if c.endswith('_Costs')]
        total_costs = timeline[cost_cols].sum().sum() if cost_cols else 0
        
        metrics = {
            'Total_Profit': timeline['Profit'].sum(),
            'Avg_Monthly_Profit': timeline['Profit'].mean(),
            'Profit_Volatility': timeline['Profit'].std(),
            'Total_Revenue': timeline['Revenue'].sum(),
            'Total_Costs': total_costs,
            'Avg_Capacity_Utilization': timeline['Capacity_Utilization'].mean(),
            'Avg_Fill_Rate': timeline['Demand_Satisfaction'].mean(),
            'Total_Shortage': timeline['Unmet_Demand'].sum(),
            'Periods': len(common_dates),
        }

        # Scenario comparison (if optimization result is available)
        scenario_comparison = {}
        if opt_result is not None:
            scenario_comparison = self._compare_to_scenarios(
                realized_total_profit=timeline['Profit'].sum(),
                scenario_total_profits=opt_result.total_profits,
                probabilities=opt_result.probabilities,
            )

        bt = BacktestResult(
            timeline=timeline,
            decisions=decisions_period,
            metrics=metrics,
            scenario_comparison=scenario_comparison,
            metadata={
                'model': self.model_name,
                'solver': self.solver,
                'params': {f.name: getattr(params, f.name)
                           for f in params.__dataclass_fields__.values()},
                'n_periods': len(common_dates),
            },
        )

        logger.info(f"[✓] Backtesting simulation completed!")
        return bt
