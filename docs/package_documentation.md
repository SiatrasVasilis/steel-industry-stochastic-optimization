# Package Documentation

This document is the API reference for the `steel-industry-stochastic-optimization` package. It covers every public class and method with parameter descriptions, return types, and working code examples.

---

## Table of Contents

1. [DataLoader](#1-dataloader)
2. [VarModelScenarioGenerator](#2-varmodelscenariogenerator)
3. [RegimeSwitchingGenerator](#3-regimeswitchinggenerator)
4. [StressConfig](#4-stressconfig)
5. [ModelParameters](#5-modelparameters)
6. [RiskProfile](#6-riskprofile)
7. [AdvancedTacticalPlanningModel](#7-advancedtacticalplanningmodel)
8. [SafetyStockModel and SafetyStockParams](#8-safetystockmodel-and-safetystockparams)
9. [Result Objects](#9-result-objects)
10. [End-to-End Example](#10-end-to-end-example)
11. [Rolling-Replan Workflow](#11-rolling-replan-workflow)

---

## 1. DataLoader

```python
from src.data.loader import DataLoader
```

Loads, aligns, and transforms time series data for model fitting. Supports FRED API, CSV files, and direct DataFrame input.

### Constructor

```python
DataLoader(
    fred_api_key: str | None = None,
    variable_labels: dict[str, str] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fred_api_key` | `str` | `None` | FRED API key. Alternatively set env var `FRED_API_KEY`. |
| `variable_labels` | `dict` | `{'P': 'Steel Price', 'C': 'Scrap Cost', 'D': 'Demand'}` | Display labels for plots. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `raw_data` | `pd.DataFrame \| None` | Data as loaded, before any transformation. |
| `data` | `pd.DataFrame \| None` | Processed data at monthly frequency (levels). |
| `log_returns_data` | `pd.DataFrame \| None` | Log returns of `data`. |

### Methods

---

#### `load_from_fred()`

```python
loader.load_from_fred(
    series_ids: dict[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    plot_data: bool = True,
) -> DataLoader
```

Downloads time series from FRED, aligns to monthly frequency, and forward-fills missing values.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `series_ids` | `{'P': 'WPU101704', 'C': 'WPU1012', 'D': 'IPG3311A2S'}` | Variable name → FRED series ID mapping. |
| `start_date` | `None` | ISO date string, e.g. `'2000-01-01'`. `None` = all available. |
| `end_date` | `None` | ISO date string. `None` = most recent. |
| `plot_data` | `True` | Plot the loaded levels on load. |

Returns `self` for chaining.

---

#### `load_from_csv()`

```python
loader.load_from_csv(
    filepath: str,
    date_column: str = "Date",
    **read_kwargs,
) -> DataLoader
```

Loads a CSV file with at least a date column and variable columns (`P`, `C`, `D` or as mapped).

---

#### `load_from_dataframe()`

```python
loader.load_from_dataframe(df: pd.DataFrame) -> DataLoader
```

Accepts a pre-built DataFrame with a `DatetimeIndex` and columns matching the variable names.

---

#### `subset()`

```python
loader.subset(
    n_observations: int,
    last_observation: str | None = None,
) -> DataLoader
```

Selects the most recent `n_observations` monthly rows, optionally ending at `last_observation` (ISO date string). Used to define the fitting window.

---

#### `convert_to_real_prices()`

```python
loader.convert_to_real_prices(
    anchor_date: str,
    real_data: dict[str, float],
    plot_data: bool = False,
) -> DataLoader
```

Scales the FRED index series to physical units by anchoring each variable to a known real-world value at a specific date. All other observations are scaled proportionally, preserving relative movements. Log returns are invariant to this scaling; the conversion is for interpretability and correct level reconstruction during scenario simulation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `anchor_date` | `str` | ISO date string, e.g. `'2019-12-01'`. Must exist in the data (nearest date used if missing). |
| `real_data` | `dict` | Known real values at `anchor_date`. E.g. `{'P': 520.0, 'C': 130.0, 'D': 50000.0}` — €/Tn for prices, Tn/month for demand. |
| `plot_data` | `bool` | Plot the converted series. Default `False`. |

---

#### `compute_log_returns()`

```python
loader.compute_log_returns() -> DataLoader
```

Computes $r_{i,t} = \log(p_{i,t} / p_{i,t-1})$ for all variables. Stores results in `log_returns_data`.

---

#### `get_log_returns()`

```python
loader.get_log_returns() -> pd.DataFrame
```

Returns `log_returns_data`. Raises `RuntimeError` if `compute_log_returns()` has not been called.

---

#### `get_future_data()`

```python
loader.get_future_data(
    last_observation: str,
) -> pd.DataFrame
```

Returns all rows from `raw_data` strictly after `last_observation`, in real-unit levels (after `convert_to_real_prices` has been called). Used in backtesting to retrieve the out-of-sample realized data against which optimized decisions are evaluated.

| Parameter | Type | Description |
|-----------|------|-------------|
| `last_observation` | `str` | ISO date string cutoff. All data with index > this date is returned. |

---

### Method Chaining

`DataLoader` supports a fluid interface:

```python
loader = (
    DataLoader(fred_api_key="your_key")
    .load_from_fred()
    .subset(n_observations=180, last_observation="2019-12-01")
    .convert_to_real_prices({'P': 650.0, 'C': 310.0, 'D': 45000.0})
    .compute_log_returns()
)

log_returns = loader.get_log_returns()
```

---

## 2. VarModelScenarioGenerator

```python
from src.scenario.var import VarModelScenarioGenerator
```

Generates forward scenario paths using a fitted VAR model. Inherits from `BaseScenarioGenerator`.

### Constructor

```python
VarModelScenarioGenerator(
    shock_distribution: str = "normal",
    distribution_params: dict | None = None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shock_distribution` | `"normal"` | Innovation distribution. One of `"normal"`, `"t"`, `"skewed_t"`, `"laplace"`, `"uniform"`. |
| `distribution_params` | `{}` | Distribution hyperparameters. For `"t"`: `{"df": 5}`. For `"skewed_t"`: `{"df": 5, "skew": -0.3}`. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `is_fitted` | `bool` | Whether `fit()` has been called successfully. |
| `var_model` | `VARResults \| None` | The fitted `statsmodels` VAR object. Exposes `.aic`, `.bic`, `.params`, `.resid`. |
| `fit_params` | `dict` | Summary of fitting: `{'order': p, 'nobs': N, ...}`. |

### Methods

---

#### `analyze()`

```python
generator.analyze(
    loader: DataLoader,
    max_order: int = 12,
    method: str = "bic",
) -> AnalysisResult
```

Runs pre-fitting diagnostics: stationarity tests (ADF + KPSS) and lag order selection. Does not fit the model.

Returns an `AnalysisResult` with:
- `.recommended_order` — suggested VAR lag
- `.order_selection` — IC scores for all tested lags
- `.stationarity` — per-variable ADF/KPSS results

---

#### `fit()`

```python
generator.fit(
    loader: DataLoader,
    order: int | None = None,
    max_order: int = 12,
    method: str = "bic",
) -> VarModelScenarioGenerator
```

Fits the VAR model to log-return data from `loader`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loader` | — | Fitted `DataLoader` with log returns computed. |
| `order` | `None` | Lag order $p$. `None` triggers automatic BIC selection over `1..max_order`. |
| `max_order` | `12` | Maximum lag to test during BIC selection. |
| `method` | `"bic"` | Information criterion for lag selection: `"aic"`, `"bic"`, `"hqic"`. |

Returns `self`.

---

#### `generate()`

```python
generator.generate(
    n_scenarios: int = 3000,
    horizon: int = 12,
    seed: int | None = 42,
) -> GenerationResult
```

Simulates `n_scenarios` forward trajectories of length `horizon` months.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_scenarios` | `3000` | Number of Monte Carlo paths to simulate. |
| `horizon` | `12` | Number of months to project forward. |
| `seed` | `42` | Random seed for reproducibility. `None` = non-deterministic. |

Returns a `GenerationResult` with:
- `.scenarios` — long-form DataFrame: columns `['Date', 'Scenario', 'D', 'P', 'C']`
- `.probabilities` — `pd.Series` of equal weights ($1 / S$), indexed by scenario ID
- `.n_scenarios`, `.horizon`, `.start_date`, `.end_date`

---

#### `reduce()`

```python
generator.reduce(
    result: GenerationResult,
    n_clusters: int = 200,
    stress: StressConfig | None = None,
) -> ReductionResult
```

Reduces the full scenario set to `n_clusters` representative scenarios using K-medoids.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `result` | — | Output of `generate()`. |
| `n_clusters` | `200` | Target number of reduced scenarios. |
| `stress` | `None` | `StressConfig` specifying tail scenarios to preserve. If `None`, pure K-medoids. |

Returns a `ReductionResult` with:
- `.scenarios` — reduced scenario DataFrame (same format as input)
- `.probabilities` — redistributed probabilities summing to 1
- `.reduction_ratio` — fraction of scenarios removed
- `.stress_scenarios_included` — number of stress scenarios in the reduced set

---

#### `fit_diagnostics()`

```python
generator.fit_diagnostics() -> FitDiagnosticsReport
```

Returns a comprehensive report of VAR fit quality: Ljung-Box autocorrelation tests, Jarque-Bera normality tests, stability check, and residual distribution fit comparison (Normal vs Student-*t* by BIC).

```python
report = gen.fit_diagnostics()
print(report)        # formatted text summary
report.plot()        # 4-panel dashboard per variable
df = report.summary()  # machine-readable DataFrame
```

---

#### `plot_scenarios()`

```python
generator.plot_scenarios(
    result: GenerationResult | ReductionResult,
    variable_labels: dict[str, str] | None = None,
    figsize: tuple[int, int] = (14, 6),
    n_sample: int | None = None,
    show_historical: bool = True,
    show_actual: pd.DataFrame | None = None,
) -> plotly.graph_objects.Figure
```

Returns an interactive Plotly figure with one panel per variable. Shows:
- Thin grey lines for individual scenario paths
- Percentile fan bands (10th–90th, 25th–75th)
- Min/max markers
- Historical data (optional)
- Actual out-of-sample realization (optional, for backtest validation)

---

### Usage Example

```python
from src.data.loader import DataLoader
from src.scenario.var import VarModelScenarioGenerator
from src.scenario.base import StressConfig

loader = DataLoader(fred_api_key="your_key")
loader.load_from_fred().subset(180, last_observation="2022-06-01")
loader.convert_to_real_prices({'P': 780.0, 'C': 390.0, 'D': 50000.0})
loader.compute_log_returns()

gen = VarModelScenarioGenerator(shock_distribution="t", distribution_params={"df": 5})

# Optional pre-fit analysis
analysis = gen.analyze(loader)
print(f"Recommended lag order: {analysis.recommended_order}")

# Fit and generate
gen.fit(loader, order=analysis.recommended_order)
result = gen.generate(n_scenarios=3000, horizon=12, seed=42)

# Check fit quality
report = gen.fit_diagnostics()
print(report)

# Reduce with stress preservation
stress = StressConfig(
    pct=0.05,
    variable_stress={"p_sell": "lower", "c_spot": "upper"},
)
reduced = gen.reduce(result, n_clusters=200, stress=stress)

fig = gen.plot_scenarios(reduced)
fig.show()
```

---

## 3. RegimeSwitchingGenerator

```python
from src.scenario.regime_switching import RegimeSwitchingGenerator
```

Generates forward scenario paths using a Markov-Switching VAR model. Captures two or more distinct market regimes (e.g., normal and stress) with separate volatility structures and mean returns. Scenarios switch between regimes according to estimated transition probabilities, naturally producing realistic crash trajectories and fat-tailed joint distributions. This is the **recommended generator** for rolling backtests.

### Constructor

```python
RegimeSwitchingGenerator(
    n_regimes: int = 2,
    shock_distribution: str = "t",
    distribution_params: dict | None = None,
    switching_variance: bool = True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_regimes` | `2` | Number of hidden market regimes. `2` = normal + stress (recommended). |
| `shock_distribution` | `"t"` | Innovation distribution within each regime. Same options as `VarModelScenarioGenerator`. |
| `distribution_params` | `{"df": 5}` | Distribution hyperparameters. |
| `switching_variance` | `True` | If `True`, each regime has its own covariance matrix. If `False`, only mean returns differ across regimes. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `is_fitted` | `bool` | Whether `fit()` has been called successfully. |
| `n_regimes` | `int` | Number of regimes as specified. |
| `_transition_matrix` | `np.ndarray` | $R \times R$ Markov transition matrix. `_transition_matrix[i, j]` = probability of moving from regime $i$ to regime $j$. |
| `_steady_state_probs` | `np.ndarray` | Unconditional regime probabilities (stationary distribution). |
| `_regime_fractions` | `dict` | Empirical fraction of training observations in each regime. |
| `_regime_covs` | `dict` | Shrinkage-regularised covariance matrix per regime. |
| `_regime_means` | `dict` | Mean return shift per regime (relative to full-sample VAR intercept). |

### Methods

`RegimeSwitchingGenerator` shares the identical public API as `VarModelScenarioGenerator`:

- `fit(loader, order=None, max_order=12, method="bic")` — fits the three-step MSIH model (see mathematical formulation Section 1.8)
- `generate(n_scenarios, horizon, seed)` → `GenerationResult`
- `reduce(result, n_clusters, stress)` → `ReductionResult`
- `fit_diagnostics()` → `FitDiagnosticsReport`
- `plot_scenarios(result, ...)` → `plotly.Figure`

#### Additional Method: `regime_summary()`

```python
generator.regime_summary() -> pd.DataFrame
```

Returns a DataFrame summarising the fitted regime characteristics. Columns: `regime`, `label`, `fraction`, `persistence`, `expected_duration_months`, `avg_volatility`, `mean_D`, `mean_P`, `mean_C`.

```python
gen = RegimeSwitchingGenerator(n_regimes=2)
gen.fit(loader)
print(gen.regime_summary())
#    regime   label  fraction  persistence  expected_duration_months  avg_volatility
# 0       0  Normal      0.84         0.95                      20.0          0.0312
# 1       1  Stress      0.16         0.72                       3.6          0.0891
```

### Usage Example

```python
from src.data.loader import DataLoader
from src.scenario.regime_switching import RegimeSwitchingGenerator
from src.scenario.base import StressConfig

loader = (
    DataLoader()
    .load_from_fred(plot_data=False)
    .subset(n_observations=180, last_observation="2022-06-01")
    .convert_to_real_prices(anchor_date="2022-06-01", real_data={"P": 850.0, "C": 380.0, "D": 50000.0})
    .compute_log_returns()
)

gen = RegimeSwitchingGenerator(n_regimes=2)
gen.fit(loader)

# Inspect regime structure
print(gen.regime_summary())

# Generate and reduce
result  = gen.generate(n_scenarios=3000, horizon=12, seed=42)
reduced = gen.reduce(
    result,
    n_clusters=300,
    stress=StressConfig(pct=0.05, variable_stress={"P": "lower", "C": "upper"}),
)

print(f"Reduced to {reduced.n_scenarios} scenarios "
      f"({reduced.stress_scenarios_included} stress-preserved)")
```

---

## 4. StressConfig

```python
from src.scenario.base import StressConfig
```

A dataclass that specifies which extreme scenarios must survive the K-medoids reduction step. Three mutually exclusive targeting modes: variable-tail, composite score, or manual force-include.

### Constructor

```python
@dataclass
StressConfig(
    pct: float = 0.05,
    variable_stress: dict[str, str] | None = None,
    composite_weights: dict[str, float] | None = None,
    composite_direction: str = "both",
    force_include: list | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pct` | `float` | `0.05` | Fraction of the **reduced** set reserved for stress scenarios. `n_stress = ceil(n_clusters × pct)`. Recommended range: 0.03–0.10. |
| `variable_stress` | `dict` | `None` | Per-variable tail selection. Key = column name, value = `'lower'`, `'upper'`, or `'both'`. |
| `composite_weights` | `dict` | `None` | Weighted score across variables. Positive weight = higher is better (e.g. revenue); negative = higher is worse (e.g. cost). Mutually exclusive with `variable_stress`. |
| `composite_direction` | `str` | `'both'` | Which tail of the composite score to protect: `'lower'`, `'upper'`, or `'both'`. |
| `force_include` | `list` | `None` | Scenario IDs to always include regardless of clustering. May be combined with either mode above. |

### Validation

- `pct` must be in `[0, 1]`.
- `variable_stress` and `composite_weights` are mutually exclusive; specifying both raises `ValueError`.
- Each direction in `variable_stress` must be one of `{'upper', 'lower', 'both'}`.

### Modes

**Mode 1 — Variable-tail** (recommended for procurement models)

Identifies the worst steel price and worst scrap cost scenarios independently. The mean value across the entire horizon is used to rank scenarios per variable.

```python
stress = StressConfig(
    pct=0.05,
    variable_stress={
        "p_sell": "lower",   # protect low-price scenarios (bad revenue)
        "c_spot": "upper",   # protect high-cost scenarios (bad procurement)
    },
)
```

**Mode 2 — Composite score**

Projects scenarios onto a single profit-proxy axis before selecting tails.

```python
stress = StressConfig(
    pct=0.05,
    composite_weights={"p_sell": 1.0, "c_spot": -1.0},
    composite_direction="lower",  # protect worst margin scenarios
)
```

**Mode 3 — Force include**

Guarantees specific scenario IDs survive (e.g., a COVID-like benchmark scenario you constructed manually).

```python
stress = StressConfig(
    pct=0.03,
    variable_stress={"c_spot": "upper"},
    force_include=[42, 137, 891],
)
```

**No stress (default)**

Pass `stress=None` to `reduce()` for pure K-medoids without any tail preservation.

---

## 5. ModelParameters

```python
from src.optimization.advanced import ModelParameters
```

A dataclass holding all deterministic inputs to the optimization model. All parameters can be provided as scalars (same value for all periods) or as dictionaries keyed by period index (for time-varying specifications).

### Constructor

```python
@dataclass
ModelParameters(
    # Required
    c_fix,
    x_fix_max,
    x_opt_max,
    # Optional with defaults
    f_opt = 0.0,
    basis_opt = 0.0,
    floor_opt = 0.0,
    cap_opt = inf,
    Cap_base = 10000.0,
    Cap_flex = 2000.0,
    c_base = 50.0,
    c_flex = 80.0,
    alpha = 1.2,
    h_rm = 2.0,
    h_fg = 5.0,
    I_rm0 = 0.0,
    I_fg0 = 0.0,
    pen_short = 500.0,
    I_rm_max = None,
    I_fg_max = None,
    spot_max = None,
)
```

### Parameter Reference

#### Procurement and Contracts

| Parameter | Unit | Description |
|-----------|------|-------------|
| `c_fix` | €/Tn RM | Fixed contract scrap cost |
| `x_fix_max` | Tn RM | Maximum fixed contract volume per period |
| `x_opt_max` | Tn RM | Maximum framework reservation per period |
| `f_opt` | €/Tn reserved | Framework reservation fee (0 = no upfront fee) |

#### Framework Pricing

| Parameter | Unit | Description |
|-----------|------|-------------|
| `basis_opt` | €/Tn | Call-off price = clip(spot + basis, floor, cap) |
| `floor_opt` | €/Tn | Minimum call-off exercise price |
| `cap_opt` | €/Tn | Maximum call-off exercise price |

#### Production Capacity

| Parameter | Unit | Description |
|-----------|------|-------------|
| `Cap_base` | Tn FG | Maximum base production per period |
| `Cap_flex` | Tn FG | Maximum flex production per period |
| `c_base` | €/Tn FG | Base production variable cost |
| `c_flex` | €/Tn FG | Flex production variable cost (must be ≥ `c_base`) |

#### Inventory and Conversion

| Parameter | Unit | Description |
|-----------|------|-------------|
| `alpha` | Tn RM / Tn FG | RM yield: tons of scrap per ton of steel produced |
| `h_rm` | €/Tn RM per period | Raw material holding cost |
| `h_fg` | €/Tn FG per period | Finished goods holding cost |
| `I_rm0` | Tn RM | Initial RM inventory |
| `I_fg0` | Tn FG | Initial FG inventory |

#### Service

| Parameter | Unit | Description |
|-----------|------|-------------|
| `pen_short` | €/Tn FG | Unmet demand penalty (shortage cost) |

#### Optional Capacity Constraints

| Parameter | Unit | Description |
|-----------|------|-------------|
| `I_rm_max` | Tn RM | Maximum RM storage capacity (`None` = unlimited) |
| `I_fg_max` | Tn FG | Maximum FG storage capacity (`None` = unlimited) |
| `spot_max` | dict `(t,w)→float` | Scenario-specific spot availability cap (`None` = unlimited) |

### Time-Varying Parameters

Any scalar parameter can be replaced by a `dict` keyed by the period's `pd.Timestamp`:

```python
from src.optimization.advanced import ModelParameters

# Time-varying capacity (seasonality: higher in Q4)
import pandas as pd
dates = pd.date_range("2023-01-01", periods=12, freq="MS")

cap_base_monthly = {t: 9000 if t.month in [1, 2] else 11000 for t in dates}

params = ModelParameters(
    c_fix=320.0,
    x_fix_max=8000.0,
    x_opt_max=3000.0,
    Cap_base=cap_base_monthly,
    Cap_flex=2500.0,
    c_base=55.0,
    c_flex=90.0,
    alpha=1.15,
    pen_short=600.0,
)
```

---

## 6. RiskProfile

```python
from src.params.risk import RiskProfile
```

Controls the trade-off between expected profit maximization and downside risk protection via CVaR.

### Constructor

```python
@dataclass
RiskProfile(
    risk_aversion: float = 0.0,
    cvar_alpha: float = 0.05,
)
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| `risk_aversion` | $[0, 1]$ | $\lambda$ — weight on CVaR in the objective: $\max\ (1-\lambda)\,\mathbb{E}[Z] + \lambda\,\text{CVaR}_\alpha(Z)$. `0` = risk-neutral, `1` = pure risk minimization. |
| `cvar_alpha` | $(0, 1)$ | Tail probability: `0.05` means CVaR is the expected profit in the worst 5% of scenarios. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_risk_neutral` | `bool` | `True` iff `risk_aversion == 0.0`. |
| `is_fully_risk_averse` | `bool` | `True` iff `risk_aversion == 1.0`. |

### Pre-built Profiles

```python
# Risk-neutral (default)
risk = RiskProfile.risk_neutral()         # λ=0.0, α=0.05

# Moderate downside protection
risk = RiskProfile.conservative()         # λ=0.3, α=0.05

# Light risk adjustment
risk = RiskProfile.aggressive()           # λ=0.1, α=0.10

# Heavy downside protection
risk = RiskProfile.defensive()            # λ=0.5, α=0.05
```

### Serialization

```python
# JSON export and import
risk.to_json("configs/risk_profile.json")
risk2 = RiskProfile.from_json("configs/risk_profile.json")

# Dictionary round-trip
d = risk.to_dict()
risk3 = RiskProfile.from_dict(d)
```

---

## 7. AdvancedTacticalPlanningModel

```python
from src.optimization.advanced import AdvancedTacticalPlanningModel
```

Solves the full two-stage stochastic program. Requires three scenario variables: `D` (demand), `c_spot` (scrap cost), `p_sell` (selling price).

### Constructor

```python
AdvancedTacticalPlanningModel(solver: str = "highs")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `solver` | `"highs"` | LP solver. Supported: `"highs"`, `"gurobi"`, `"cplex"`, `"glpk"`. |

### Methods

---

#### `run()`

```python
model.run(
    scenarios: pd.DataFrame,
    prob: pd.Series,
    params: ModelParameters,
    risk_profile: RiskProfile | None = None,
    variable_mapping: dict[str, str] | None = None,
) -> OptimizationResult
```

The primary interface. Accepts scenarios with any column names and maps them to the required `D`, `c_spot`, `p_sell` via the built-in alias dictionary or a custom `variable_mapping`.

| Parameter | Description |
|-----------|-------------|
| `scenarios` | Long-form DataFrame: must contain `Date`, `Scenario`, and three variable columns. |
| `prob` | `pd.Series` indexed by scenario ID with probabilities summing to 1. |
| `params` | `ModelParameters` instance. |
| `risk_profile` | `RiskProfile` (optional). `None` = risk-neutral. |
| `variable_mapping` | Override column renaming, e.g. `{'steel_price': 'p_sell'}`. If `None`, uses built-in aliases. |

---

#### `backtest()`

```python
model.backtest(
    scenarios: pd.DataFrame,
    prob: pd.Series,
    params: ModelParameters,
    actual_data: pd.DataFrame,
    risk_profile: RiskProfile | None = None,
) -> BacktestResult
```

Runs the optimizer on `scenarios` to get first-stage decisions, then evaluates those decisions against `actual_data` (out-of-sample realized values).

Returns a `BacktestResult` with:
- `.timeline` — monthly DataFrame with all decision and cost columns
- `.risk_metrics` — dict: `Expected_Profit`, `CVaR_5pct`, `Fill_Rate`, `VSS`, etc.
- `.decisions` — first-stage decision dict (`x_fix`, `x_opt`, `P_base` by period)

---

#### `optimize()` (low-level)

```python
model.optimize(
    scenarios: pd.DataFrame,
    prob: pd.Series,
    params: ModelParameters,
    risk_profile: RiskProfile | None = None,
) -> OptimizationResult
```

Same as `run()` but requires columns to already be named `D`, `c_spot`, `p_sell`. Use `run()` for automatic column mapping.

---

### Required Column Names and Aliases

`run()` accepts any of the following column names and maps them internally:

| Internal name | Accepted aliases |
|--------------|-----------------|
| `D` | `demand`, `Demand`, `demand_tons` |
| `c_spot` | `spot_cost`, `Spot_Cost`, `scrap_cost`, `Scrap_Cost`, `C` |
| `p_sell` | `sell_price`, `Sell_Price`, `steel_price`, `Steel_Price`, `P` |

---

### Usage Example

```python
from src.optimization.advanced import AdvancedTacticalPlanningModel, ModelParameters
from src.params.risk import RiskProfile

params = ModelParameters(
    c_fix=320.0,
    x_fix_max=8000.0,
    x_opt_max=4000.0,
    f_opt=2.0,
    basis_opt=-10.0,
    floor_opt=280.0,
    cap_opt=400.0,
    Cap_base=10000.0,
    Cap_flex=2000.0,
    c_base=55.0,
    c_flex=85.0,
    alpha=1.2,
    h_rm=2.5,
    h_fg=5.0,
    I_rm0=500.0,
    I_fg0=0.0,
    pen_short=500.0,
)

risk = RiskProfile(risk_aversion=0.25, cvar_alpha=0.05)
model = AdvancedTacticalPlanningModel(solver="highs")

# scenarios and prob come from VarModelScenarioGenerator.reduce()
result = model.run(
    reduced.scenarios,
    reduced.probabilities,
    params=params,
    risk_profile=risk,
)

print(f"Expected Profit : €{result.risk_metrics['Expected_Profit']:>12,.0f}")
print(f"CVaR (5%)       : €{result.risk_metrics['CVaR_5pct']:>12,.0f}")
print(f"Fill Rate       :  {result.risk_metrics['Fill_Rate']:>10.1%}")

# Inspect first-stage decisions
for t, v in result.decisions['x_fix'].items():
    print(f"  {t:%Y-%m}  fixed={v:,.0f} Tn")
```

---

## 8. SafetyStockModel and SafetyStockParams

```python
from src.optimization.benchmark import SafetyStockModel, SafetyStockParams
```

A rule-based benchmark model representing traditional ERP/MRP planning. Used to compute the **Value of Stochastic Solution (VSS)**.

### SafetyStockParams

```python
@dataclass
SafetyStockParams(
    service_level: float = 0.95,
    review_period: int = 1,
    fixed_pct: float = 0.60,
    framework_pct: float = 0.25,
    production_smoothing: bool = True,
    safety_stock_fg: bool = True,
    safety_stock_rm: bool = True,
    safety_stock_periods: float | None = None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `service_level` | `0.95` | Target fill-rate. Used to size safety stock via normal quantile. |
| `review_period` | `1` | Planning cycle in months. Affects safety stock formula. |
| `fixed_pct` | `0.60` | Share of expected RM requirement sourced from fixed contracts. |
| `framework_pct` | `0.25` | Share of demand variability hedged via framework contracts. Remainder = spot. |
| `production_smoothing` | `True` | `True` = level-load production. `False` = chase demand forecast. |
| `safety_stock_fg` | `True` | Maintain finished goods safety stock. |
| `safety_stock_rm` | `True` | Maintain raw material safety stock. |
| `safety_stock_periods` | `None` | Override: hold this many periods of demand as safety stock instead of using `service_level`. |

### SafetyStockModel

```python
SafetyStockModel()
```

No constructor arguments (no solver needed — rule-based, not an optimization).

#### `run()`

```python
model.run(
    scenarios: pd.DataFrame,
    prob: pd.Series,
    params: ModelParameters,
    policy: SafetyStockParams | None = None,
) -> OptimizationResult
```

Applies the deterministic rules using the scenario mean as the "point forecast". Then evaluates the resulting fixed decisions across *all* scenarios to compute expected profit and risk metrics — enabling fair comparison with the stochastic optimizer.

---

### VSS Calculation

```python
from src.optimization.advanced import AdvancedTacticalPlanningModel, ModelParameters
from src.optimization.benchmark import SafetyStockModel, SafetyStockParams

# Solve both models on the same scenario set
adv_model = AdvancedTacticalPlanningModel()
bm_model  = SafetyStockModel()

adv_result = adv_model.run(scenarios, prob, params=params, risk_profile=risk)
bm_result  = bm_model.run(scenarios, prob, params=params, policy=SafetyStockParams())

vss = (adv_result.risk_metrics['Expected_Profit']
       - bm_result.risk_metrics['Expected_Profit'])

print(f"VSS: €{vss:,.0f} per planning cycle")
```

---

## 9. Result Objects

### OptimizationResult

Returned by `model.run()` and `model.optimize()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `risk_metrics` | `dict` | Keys: `Expected_Profit`, `CVaR_5pct`, `VaR_5pct`, `Fill_Rate`, `Profit_Std`, `Min_Profit`, `Max_Profit`. |
| `decisions` | `dict` | First-stage decisions. Keys: `x_fix`, `x_opt`, `P_base` — each a `{period: value}` dict. |
| `timeline` | `pd.DataFrame` | Per-scenario, per-period results with columns: `Fixed_Costs`, `Spot_Costs`, `Framework_Costs`, `Production_Costs`, `Holding_Costs`, `Shortage_Costs`, `Total_Revenue`, `Total_Costs`, `Profit`. |
| `scenario_profits` | `pd.Series` | Scenario-level total profit, indexed by scenario ID. |

---

### BacktestResult

Returned by `model.backtest()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `timeline` | `pd.DataFrame` | Monthly actuals evaluated with the optimized first-stage decisions. |
| `risk_metrics` | `dict` | Same keys as `OptimizationResult.risk_metrics` but computed on out-of-sample actuals. |
| `decisions` | `dict` | Same first-stage decisions as chosen at planning time. |
| `fill_rate` | `float` | Realized demand fill rate over the backtest horizon. |

---

### GenerationResult and ReductionResult

Returned by `generator.generate()` and `generator.reduce()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `scenarios` | `pd.DataFrame` | Long-form: `['Date', 'Scenario', 'D', 'P', 'C']` |
| `probabilities` | `pd.Series` | Weights indexed by scenario ID, summing to 1. |
| `n_scenarios` | `int` | Number of scenarios in this set. |
| `horizon` | `int` | Forecast horizon in periods. |
| `stress_scenarios_included` | `int` | (`ReductionResult` only) Count of stress-preserved scenarios. |
| `reduction_ratio` | `float` | (`ReductionResult` only) Fraction of original scenarios removed. |

---

## 10. End-to-End Example

The following script demonstrates a complete workflow: load data, generate scenarios, optimize, benchmark, and report results.

```python
import pandas as pd
from src.data.loader import DataLoader
from src.scenario.var import VarModelScenarioGenerator
from src.scenario.regime_switching import RegimeSwitchingGenerator
from src.scenario.base import StressConfig
from src.optimization.advanced import AdvancedTacticalPlanningModel, ModelParameters
from src.optimization.benchmark import SafetyStockModel, SafetyStockParams
from src.params.risk import RiskProfile

# ── Configuration ─────────────────────────────────────────────────────────────
FRED_API_KEY  = "your_key_here"
LAST_OBS      = "2019-12-01"
HORIZON       = 12
N_GEN         = 3000
N_CLUSTERS    = 200
SOLVER        = "highs"
SEED          = 42

# ── 1. Load data ──────────────────────────────────────────────────────────────
loader = DataLoader(fred_api_key=FRED_API_KEY)
loader.load_from_fred(plot_data=False)
loader.subset(n_observations=180, last_observation=LAST_OBS)
loader.convert_to_real_prices({'P': 650.0, 'C': 310.0, 'D': 45000.0})
loader.compute_log_returns()

# ── 2. Fit Regime-Switching VAR and generate scenarios ────────────────────────
from src.scenario.regime_switching import RegimeSwitchingGenerator

gen = RegimeSwitchingGenerator(n_regimes=2)  # normal + stress regimes
gen.fit(loader)                              # BIC lag selection + MS-AR regime identification
print(gen.regime_summary())                  # inspect fitted regime structure

result = gen.generate(n_scenarios=N_GEN, horizon=HORIZON, seed=SEED)

# ── 3. Reduce ─────────────────────────────────────────────────────────────────
stress   = StressConfig(pct=0.05, variable_stress={"p_sell": "lower", "c_spot": "upper"})
reduced  = gen.reduce(result, n_clusters=N_CLUSTERS, stress=stress)

# ── 4. Define business parameters ─────────────────────────────────────────────
params = ModelParameters(
    c_fix=320.0,    x_fix_max=8000.0,   x_opt_max=4000.0,
    f_opt=2.0,      basis_opt=-10.0,    floor_opt=280.0,    cap_opt=420.0,
    Cap_base=10000.0,  Cap_flex=2000.0,
    c_base=55.0,    c_flex=85.0,
    alpha=1.2,      h_rm=2.5,   h_fg=5.0,
    I_rm0=500.0,    pen_short=500.0,
)

# ── 5. Stochastic optimization ────────────────────────────────────────────────
risk     = RiskProfile.conservative()   # λ=0.3, α=0.05
adv_mdl  = AdvancedTacticalPlanningModel(solver=SOLVER)
adv_res  = adv_mdl.run(reduced.scenarios, reduced.probabilities,
                        params=params, risk_profile=risk)

# ── 6. Benchmark ──────────────────────────────────────────────────────────────
bm_mdl   = SafetyStockModel()
bm_res   = bm_mdl.run(reduced.scenarios, reduced.probabilities,
                       params=params, policy=SafetyStockParams())

# ── 7. Report ─────────────────────────────────────────────────────────────────
vss = adv_res.risk_metrics['Expected_Profit'] - bm_res.risk_metrics['Expected_Profit']

print("┌─────────────────────────────────────────────┐")
print("│           Optimization Results               │")
print("├─────────────────────────────────────────────┤")
print(f"│ Expected Profit  : €{adv_res.risk_metrics['Expected_Profit']:>12,.0f}  vs  €{bm_res.risk_metrics['Expected_Profit']:>12,.0f} │")
print(f"│ CVaR (5%)        : €{adv_res.risk_metrics['CVaR_5pct']:>12,.0f}  vs  €{bm_res.risk_metrics['CVaR_5pct']:>12,.0f} │")
print(f"│ Fill Rate        :  {adv_res.risk_metrics['Fill_Rate']:>10.1%}  vs   {bm_res.risk_metrics['Fill_Rate']:>10.1%} │")
print(f"│ VSS              : €{vss:>12,.0f}                     │")
print("└─────────────────────────────────────────────┘")
```

---

## 11. Rolling-Replan Workflow

The primary use case for the package is a **rolling-replan backtest**: repeatedly re-fitting the scenario model, re-optimizing, executing a fixed execution window against realized data, and carrying inventory state forward. The following pattern implements this workflow.

```python
import pickle
import pandas as pd
from pathlib import Path

from src.data.loader import DataLoader
from src.scenario.regime_switching import RegimeSwitchingGenerator
from src.scenario.base import StressConfig
from src.optimization.advanced import AdvancedTacticalPlanningModel, ModelParameters
from src.optimization.benchmark import SafetyStockModel, SafetyStockParams
from src.params.risk import RiskProfile

# ── Configuration ──────────────────────────────────────────────────────────────
BACKTEST_DATES = pd.date_range("2007-12", "2022-06", freq="6MS").astype(str).tolist()
HORIZON        = 12
EXECUTE_MONTHS = 6
N_SCENARIOS    = 3_000
N_CLUSTERS     = 300
RISK_PROFILE   = RiskProfile(risk_aversion=0.3, cvar_alpha=0.05)

# ── Initial state ─────────────────────────────────────────────────────────────
real_data = {"P": 520.0, "C": 130.0, "D": 50_000.0}
I_rm0, I_fg0 = 5_000.0, 5_000.0
results = []

for date in BACKTEST_DATES:

    # 1. Load and fit on trailing 180 months
    loader = (
        DataLoader()
        .load_from_fred(plot_data=False)
        .subset(n_observations=180, last_observation=date)
        .convert_to_real_prices(anchor_date=date, real_data=real_data)
        .compute_log_returns()
    )
    actual      = loader.get_future_data(last_observation=date)
    future_data = actual.rename(columns={"C": "c_spot", "P": "p_sell"})

    # 2. Generate scenarios
    gen     = RegimeSwitchingGenerator(n_regimes=2)
    gen.fit(loader)
    raw     = gen.generate(n_scenarios=N_SCENARIOS, horizon=HORIZON, seed=42)
    reduced = gen.reduce(raw, n_clusters=N_CLUSTERS)

    # 3. Build parameters from current market prices
    params = ModelParameters(
        c_fix=real_data["C"] * 0.96,
        f_opt=real_data["C"] * 0.04,
        x_fix_max=40_000.0, x_opt_max=20_000.0,
        Cap_base=42_000.0,  Cap_flex=10_000.0,
        c_base=50.0,        c_flex=80.0,
        alpha=1.05,         h_rm=1.8,   h_fg=5.0,
        I_rm0=I_rm0,        I_fg0=I_fg0,
        pen_short_pct=1.5,
    )

    # 4. Optimize and backtest
    model  = AdvancedTacticalPlanningModel(solver="highs")
    result = model.run(
        reduced.scenarios, reduced.probabilities,
        variable_mapping={"C": "c_spot", "P": "p_sell"},
        params=params, risk_profile=RISK_PROFILE,
    )
    bt = model.backtest(decisions=result, actual_data=future_data, params=params)

    # 5. Record window profit
    window_profit = bt.timeline.iloc[:EXECUTE_MONTHS]["Profit"].sum()
    results.append({"date": date, "profit": window_profit})
    print(f"{date}  profit: {window_profit/1e6:+.2f}M")

    # 6. Carry forward inventory state and price anchor
    I_rm0     = float(bt.timeline["I_rm"].iloc[EXECUTE_MONTHS - 1])
    I_fg0     = float(bt.timeline["I_fg"].iloc[EXECUTE_MONTHS - 1])
    real_data = {
        "P": float(actual["P"].iloc[EXECUTE_MONTHS - 1]),
        "C": float(actual["C"].iloc[EXECUTE_MONTHS - 1]),
        "D": float(actual["D"].iloc[EXECUTE_MONTHS - 1]),
    }

print(f"\nTotal profit: {sum(r['profit'] for r in results)/1e6:.1f}M")
```

**Key design points:**

| Step | What happens | Why it matters |
|------|-------------|----------------|
| `subset(..., last_observation=date)` | Training window ends at replan date | Prevents look-ahead bias |
| `convert_to_real_prices(anchor_date=date, ...)` | Rescales indices to current price levels | Keeps contract ratios calibrated to market |
| `get_future_data(last_observation=date)` | Out-of-sample actuals for evaluation | Simulates real execution against unknown future |
| Carry `I_rm0`, `I_fg0` | Inventory levels propagate between windows | Realistic simulation — over-procurement in one window affects the next |
| Update `real_data` | Price anchor updated from realized prices | Contract pricing stays anchored to actual market at each replan |
