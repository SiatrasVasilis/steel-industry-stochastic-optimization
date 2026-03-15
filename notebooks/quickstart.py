# %% notebooks/quickstart.py
# Quick-start walkthrough: single planning window end-to-end.
#
# Covers:
#   1. Load FRED market data
#   2. Fit Regime-Switching VAR and generate scenarios
#   3. Solve the stochastic procurement + capacity optimisation
#   4. Compare against the Safety Stock benchmark
#   5. Print a results summary
#
# Runtime: ~2-3 minutes (scenario generation + LP solve)
#
# Usage:
#   .venv/Scripts/python.exe notebooks/quickstart.py
#   or run cell-by-cell in VSCode (Ctrl+Enter per cell)

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path('.').resolve()
sys.path.insert(0, str(_here.parent / 'src'))

import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

# %% -- 1. Load data ----------------------------------------------------------
# Requires FRED_API_KEY in your .env file (copy .env.example -> .env).
# Downloads: WPU101704 (steel price), WPU1012 (scrap cost), IPG3311A2S (demand).

from data.loader import DataLoader

PLANNING_DATE = '2019-12-01'   # treat this as "today"
REAL_DATA     = {'P': 520.0, 'C': 130.0, 'D': 50_000.0}  # anchor to real units

loader = (
    DataLoader()
    .load_from_fred(plot_data=False)
    .subset(n_observations=180, last_observation=PLANNING_DATE)
    .convert_to_real_prices(anchor_date=PLANNING_DATE, real_data=REAL_DATA)
    .compute_log_returns()
)

print(f"Training window: {loader.data.index[0].date()} to {loader.data.index[-1].date()}")
print(f"Observations   : {len(loader.data)}")

# %% -- 2. Scenario generation -------------------------------------------------
# RegimeSwitchingGenerator fits a 2-regime Markov-Switching VAR.
# Regime 0 = normal market, Regime 1 = stress / crisis.

from scenario import RegimeSwitchingGenerator

generator = RegimeSwitchingGenerator(n_regimes=2)
generator.fit(loader)
print(generator.regime_summary())

raw     = generator.generate(n_scenarios=3_000, horizon=12, seed=42)
reduced = generator.reduce(raw, n_clusters=300)

print(f"\nScenarios after reduction: {len(reduced.scenarios['Scenario'].unique())}")

# %% -- 3. Optimise ------------------------------------------------------------
# Two-stage stochastic LP with CVaR risk measure.
# lambda=0.0 -> pure expected-profit maximisation
# lambda=0.3 -> moderate downside protection (recommended starting point)
# lambda=1.0 -> full CVaR minimisation

from optimization import StochasticOptimizationModel
from optimization.stochastic import ModelParameters
from params import RiskProfile

scrap_price = REAL_DATA['C']

params = ModelParameters(
    c_fix     = scrap_price * 0.96,   # fixed contract price
    f_opt     = scrap_price * 0.04,   # framework reservation fee
    basis_opt = scrap_price * -0.02,  # basis adjustment
    floor_opt = scrap_price * 0.75,   # floor exercise price
    cap_opt   = scrap_price * 1.50,   # cap exercise price
    x_fix_max = 40_000.0,
    x_opt_max = 20_000.0,
    Cap_base  = 42_000.0,
    Cap_flex  = 10_000.0,
    c_base    = 50.0,
    c_flex    = 80.0,
    alpha     = 1.05,
    h_rm      = 1.8,
    h_fg      = 5.0,
    pen_short = scrap_price * 1.5,
    I_rm0     = 5_000.0,
    I_fg0     = 5_000.0,
)

model  = StochasticOptimizationModel(solver='highs')
result = model.run(
    scenarios=reduced.scenarios,
    prob=reduced.probabilities,
    params=params,
    risk_profile=RiskProfile(risk_aversion=0.3, cvar_alpha=0.05),
    variable_mapping={'C': 'c_spot', 'P': 'p_sell'},
)

print("\n-- Stochastic Optimiser ----------------------------------")
print(f"  E[Profit]  : EUR {result.risk_metrics['Expected_Profit']:>12,.0f}")
print(f"  CVaR (5%)  : EUR {result.risk_metrics['CVaR_95']:>12,.0f}")
print(f"  Sharpe     :      {result.risk_metrics['Sharpe']:>11.3f}")

# %% -- 4. Safety Stock benchmark ----------------------------------------------
# Replicate the same procurement decision using the traditional ERP/MRP approach.

from optimization import SafetyStockModel, SafetyStockParams

ss_policy = SafetyStockParams(
    service_level=0.95, fixed_pct=0.60,
    framework_pct=0.25, production_smoothing=True,
)
ss_model  = SafetyStockModel()
ss_result = ss_model.run(
    scenarios=reduced.scenarios,
    prob=reduced.probabilities,
    params=params,
    policy=ss_policy,
    variable_mapping={'C': 'c_spot', 'P': 'p_sell'},
)

print("\n-- Safety Stock Benchmark --------------------------------")
print(f"  E[Profit]  : EUR {ss_result.risk_metrics['Expected_Profit']:>12,.0f}")
print(f"  CVaR (5%)  : EUR {ss_result.risk_metrics['CVaR_95']:>12,.0f}")
print(f"  Sharpe     :      {ss_result.risk_metrics['Sharpe']:>11.3f}")

# %% -- 5. Value of Stochastic Solution ----------------------------------------
vss     = result.risk_metrics['Expected_Profit'] - ss_result.risk_metrics['Expected_Profit']
vss_pct = vss / abs(ss_result.risk_metrics['Expected_Profit']) * 100

print("\n-- Value of Stochastic Solution (VSS) -------------------")
print(f"  VSS        : EUR {vss:>12,.0f}  ({vss_pct:+.2f}%)")
print("\nStage-1 decisions (month 1):")
print(f"  Fixed procurement  : {result.decisions['x_fix'][1]:>8,.0f} Tn")
print(f"  Framework reserved : {result.decisions['x_opt'][1]:>8,.0f} Tn")
print(f"  Base capacity      : {result.decisions['P_base'][1]:>8,.0f} Tn")

# %% -- 6. Backtest against realised data --------------------------------------
# Evaluate the committed Stage-1 decisions against what actually happened.

actual_data = loader.get_future_data(last_observation=PLANNING_DATE)
future_data = actual_data.rename(columns={'C': 'c_spot', 'P': 'p_sell'})

stoch_bt = model.backtest(decisions=result, actual_data=future_data, params=params)
ss_bt    = ss_model.backtest(decisions=ss_result, actual_data=future_data,
                              params=params, policy=ss_policy)

stoch_profit = stoch_bt.timeline['Profit'].sum()
ss_profit    = ss_bt.timeline['Profit'].sum()

print("\n-- Out-of-sample Backtest (12 months) -------------------")
print(f"  Stochastic profit  : EUR {stoch_profit:>12,.0f}")
print(f"  Benchmark profit   : EUR {ss_profit:>12,.0f}")
print(f"  Realised VSS       : EUR {stoch_profit - ss_profit:>12,.0f}")
print(f"\nMonthly timeline (Stochastic):")
print(stoch_bt.timeline[['Profit', 'Demand_Satisfaction', 'Capacity_Utilization']].round(2).to_string())
