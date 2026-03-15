# Project Context

## What This Is

A **two-stage stochastic programming** framework for steel procurement and capacity planning. It replaces traditional ERP/MRP safety-stock planning with a scenario-aware optimizer that evaluates thousands of plausible futures and finds the plan that maximizes expected profit while explicitly protecting against downside risk.

## The Business Problem

A steel manufacturer must plan procurement and production over a 12-month rolling horizon under three correlated uncertainties:
- **Steel selling price (P)** — revenue driver
- **Scrap metal cost (C)** — primary raw material input cost
- **Customer demand (D)** — volume uncertainty

Decisions are split into two stages:

**Stage 1 (commit now, before uncertainty resolves):**
- Base production capacity commitment (fixed cost regardless of use)
- Fixed procurement contracts (obligatory volume at known price)
- Framework (call-off) reservation (right but not obligation, bounded exercise price, small reservation fee)

**Stage 2 (adapt later, after observing actual prices/demand):**
- Framework exercise (call off reserved volume)
- Spot purchases (unlimited, market-rate fallback)
- Flexible production (overtime/subcontracting, more expensive)
- Inventory management (RM and FG buffers)
- Unmet demand accepted with penalty

## Project Structure

```
src/
  data/
    loader.py          — DataLoader: FRED API data loading, alignment, real-price conversion, log returns
  scenario/
    base.py            — BaseScenarioGenerator, GeneratorConfig, StressConfig, result dataclasses
    var.py             — VarModelScenarioGenerator: VAR-based scenario generation
    regime_switching.py — RegimeSwitchingGenerator: Markov-Switching VAR (2+ regimes)
  optimization/
    advanced.py        — AdvancedTacticalPlanningModel (two-stage stochastic LP), ModelParameters
    benchmark.py       — SafetyStockModel, SafetyStockParams (rule-based ERP/MRP benchmark)
    results.py         — OptimizationResult, BacktestResult dataclasses
    base.py            — Base optimization class
    simple.py          — Simplified model (legacy)
  params/
    risk.py            — RiskProfile dataclass (risk_aversion λ, cvar_alpha)
    simple.py          — Legacy params
  utils/               — Utility functions
docs/
  problem_formulation.md — Business context, contract types, production structure
  math_formulation.md    — Full mathematical formulation (VAR, stochastic program, CVaR, benchmark)
  package_documentation.md — API reference with examples
notebooks/             — Backtesting notebooks and scripts
results/               — Output directory for backtest results
```

## Pipeline Flow

```
DataLoader → ScenarioGenerator.fit() → .generate() → .reduce() → OptimizationModel.run() → .backtest()
```

1. **Data loading**: `DataLoader().load_from_fred().subset(180, last_observation=date).convert_to_real_prices(anchor_date, real_data).compute_log_returns()`
2. **Scenario generation**: Fit VAR or Regime-Switching model on log returns, simulate forward paths, reduce via K-medoids
3. **Optimization**: Solve two-stage stochastic LP (HiGHS solver) with CVaR risk measure
4. **Benchmarking**: Compare against SafetyStockModel under identical scenarios
5. **Backtesting**: Evaluate optimized decisions against out-of-sample realized data

## Key API Details

### DataLoader (`src/data/loader.py`)
- `load_from_fred(plot_data=False)` — downloads steel price (WPU101704), scrap cost (WPU1012), demand (IPG3311A2S)
- `subset(n_observations, last_observation)` — select training window
- `convert_to_real_prices(anchor_date, real_data)` — scale indices to physical units (€/Tn, Tn/month). `real_data` is `{'P': float, 'C': float, 'D': float}`. Also scales `raw_data` so `get_future_data()` returns real-unit values
- `compute_log_returns()` → populates `log_returns_data`
- `get_future_data(last_observation)` → returns data AFTER the date from `raw_data` (already in real units after conversion)
- FRED API key: from `FRED_API_KEY` env var or constructor param

### Scenario Generators
- **VarModelScenarioGenerator**: `fit(loader)`, `generate(n_scenarios, horizon, seed)`, `reduce(result, n_clusters, stress)`
- **RegimeSwitchingGenerator(n_regimes=2)**: Same API. Markov-Switching VAR with normal/stress regimes
- Scenario columns: `['Date', 'Scenario', 'D', 'P', 'C']`
- `StressConfig(pct=0.05, variable_stress={'P': 'lower', 'C': 'upper'})` — preserve tail scenarios during reduction

### Optimization
- `AdvancedTacticalPlanningModel(solver='highs')`
  - `run(scenarios, prob, params, risk_profile, variable_mapping={'C': 'c_spot', 'P': 'p_sell'})` → `OptimizationResult`
  - `backtest(decisions, actual_data, params)` → `BacktestResult`
  - **Important**: `variable_mapping={'C': 'c_spot', 'P': 'p_sell'}` is needed when scenario columns are `D, P, C` (the generator default)
- `ModelParameters(c_fix, x_fix_max, x_opt_max, ..., I_rm0, I_fg0, ...)` — all deterministic inputs
- `RiskProfile(risk_aversion=0.0, cvar_alpha=0.05)` — λ=0 is risk-neutral, λ=0.3 is moderate, λ=0.5+ is conservative

### Benchmark
- `SafetyStockModel().run(scenarios, prob, params, policy)` → `OptimizationResult`
- `SafetyStockParams(service_level=0.95, fixed_pct=0.60, framework_pct=0.25, production_smoothing=True)`
- `backtest(decisions, actual_data, params, policy)` → `BacktestResult`
- SS timeline uses `'Shortage'` column; stochastic model uses `'Unmet_Demand'`

### Result Objects
- `BacktestResult.timeline` — monthly DataFrame with: `Profit`, `Revenue`, `I_rm`, `I_fg`, `Demand_Satisfaction`, `Capacity_Utilization`, `Spot_Costs`, `Fixed_Procurement_Costs`, `Framework_Costs`, `Base_Production_Costs`, `Flex_Production_Costs`, `RM_Holding_Costs`, `FG_Holding_Costs`
- `OptimizationResult.risk_metrics` — dict with `Expected_Profit`, `CVaR_5pct`, `Fill_Rate`, etc.
- `OptimizationResult.decisions` — dict with `x_fix`, `x_opt`, `P_base` keyed by period

## Rolling Replan Backtesting

The backtest operates on a rolling-replan basis:
1. Plan 12 months ahead using scenarios generated from historical data up to a given date
2. Execute the first 6 months of the plan against realized data
3. Carry forward the end-of-window inventory state (`I_rm`, `I_fg`) and price anchor to the next planning window
4. Repeat across the full backtest period (e.g., 2007-12 to 2022-06, stepping every 6 months)

Key state carried between windows:
- `I_rm0_carry = stoch_bt.timeline['I_rm'].iloc[EXECUTE_MONTHS - 1]`
- `I_fg0_carry = stoch_bt.timeline['I_fg'].iloc[EXECUTE_MONTHS - 1]`
- `original_real_data` updated from actuals at the execution boundary

VSS (Value of Stochastic Solution) = stochastic profit - safety stock profit, measured per window and cumulatively.

## Data Sources

All from FRED (Federal Reserve Economic Data):
| Variable | Series ID | Description |
|----------|-----------|-------------|
| P (price) | WPU101704 | PPI: Hot Rolled Bars, Plates & Structural Shapes |
| C (cost) | WPU1012 | PPI: Iron and Steel Scrap |
| D (demand) | IPG3311A2S | Industrial Production: Steel Products |

## Tech Stack

- Python 3.8+, Pyomo (LP modeling), HiGHS (solver), statsmodels (VAR), scikit-learn-extra (K-medoids)
- matplotlib + plotly for visualization
- FRED API via `fredapi` package
- `.env` file for `FRED_API_KEY`

## Important Conventions

- Notebooks use `# %%` cell format (VSCode interactive Python) rather than `.ipynb` for easier editing
- Path setup in notebooks: use `Path(__file__).resolve().parent` for script mode with fallback for interactive mode
- Results are saved to `results/` directory as CSV and pickle files
- The `variable_mapping` parameter is required when passing generator output (columns `D, P, C`) to the optimizer (expects `D, c_spot, p_sell`)
