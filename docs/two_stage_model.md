# Two-Stage Capacity and Procurement Planning Model

## Table of Contents
1. [Problem Formulation](#problem-formulation)
   - [Business Context](#business-context)
   - [Decision Timeline](#decision-timeline)
   - [Sources of Uncertainty](#sources-of-uncertainty)
2. [Model Description](#model-description)
   - [Data Sources & Loading](#data-sources--loading)
   - [VAR Model: Selection, Fitting, and Testing](#var-model-selection-fitting-and-testing)
   - [Scenario Generation & Reduction](#scenario-generation--reduction)
   - [Stochastic Program](#stochastic-program)
   - [Expected Profit Analysis](#expected-profit-analysis)
3. [Usage](#usage)
   - [Method Reference](#method-reference)

---

## Problem Formulation

### Business Context

As a steel industry manufacturer, we face the critical challenge of defining an optimal **Capacity and Procurement Plan** for the upcoming 12-24 month planning horizon. The objective is to maximize expected cumulative profit while managing downside risks arising from market uncertainty in steel prices, raw material (scrap) costs, and demand volatility.

The planning problem involves making strategic decisions today that will govern operations throughout the entire planning horizon, while retaining the flexibility to adapt to market conditions as they unfold. This creates a classic two-stage decision problem under uncertainty.

### Decision Timeline

#### **Stage 1: Upfront Strategic Decisions (Today)**

At the beginning of the planning horizon, we must commit to two key strategic decisions:

**1. Base Capacity Planning ($\text{Cap}_{\text{base},t}$)**

The base capacity decision determines the maximum production volume (in tons per month) that we commit to for each period in the planning horizon. This capacity commitment involves:

- **Workforce contracts**: Hiring permanent staff with long-term employment agreements
- **Machinery rental**: Securing production equipment through annual or multi-year contracts
- **Facility preparation**: Setting up production lines and infrastructure

**Economic Implications:**
- **Fixed cost structure**: We pay a fixed cost per unit of base capacity ($c_{\text{cap,base}}$ per ton/month) for each period, paid upfront at the start of the horizon
- **Sunk cost nature**: This cost is incurred regardless of actual capacity utilization—if we produce below capacity, the fixed cost is already committed
- **Lower unit cost**: Base capacity has a lower per-unit cost compared to flexible capacity due to volume commitments and contractual terms

**2. Base Procurement Contract ($Q_{\text{base},t}$)**

The procurement decision involves securing a **call option contract** with the scrap metal supplier for each period. This financial instrument provides:

- **Quantity guarantee**: The right (but not obligation) to purchase up to $Q_{\text{base},t}$ tons of scrap per period
- **Price flexibility**: Materials are purchased at the prevailing spot market price when the option is exercised
- **No minimum commitment**: We can call for any quantity from zero up to the contracted amount without penalty
- **Availability assurance**: The supplier guarantees material availability up to the contracted quantity

**Economic Implications:**
- **Premium cost**: We pay an upfront premium ($\delta_{\text{base}}$ per ton) for the option contract, regardless of whether we exercise it
- **Spot price exposure**: When we exercise the option, we pay the spot market price ($C_t$) for the actual quantity procured
- **Flexibility value**: The option provides protection against supply shortages without committing to fixed volumes

#### **Stage 2: Operational Recourse Decisions (Throughout the Horizon)**

As market conditions are revealed over time, we can make adaptive operational decisions for each period:

**1. Flexible Capacity Extension ($\text{Cap}_{\text{flex},t,s}$)**

If realized demand exceeds our base capacity, we can temporarily expand production through:

- **Short-term employment**: Hiring temporary workers or adding extra shifts
- **On-demand machinery rental**: Leasing equipment on a month-to-month basis
- **Outsourcing**: Contracting production to third parties

**Economic Implications:**
- **Higher unit cost**: Flexible capacity costs more per unit ($c_{\text{cap,flex}} > c_{\text{cap,base}}$) due to short-term premium
- **Bounded flexibility**: Limited to a fraction of base capacity (Cap<sub>flex</sub> ≤ γ<sub>cap</sub> · Cap<sub>base</sub>) due to operational constraints
- **Scenario-dependent**: Decision is made after observing market conditions

**2. Spot Procurement Extension ($q_{\text{spot},t,s}$)**

If scrap requirements exceed the base contract quantity, we can procure additional material on the spot market:

- **Extended procurement**: Purchase beyond the contracted quantity at spot prices
- **Market availability**: Subject to spot market liquidity and supplier capacity

**Economic Implications:**
- **Additional premium**: Extra procurement incurs a higher premium ($\delta_{\text{spot}} > \delta_{\text{base}}$) reflecting urgency and lack of commitment
- **Bounded extension**: Limited to a fraction of base contract ($q_{\text{spot}} \leq \gamma_{\text{scrap}} \cdot Q_{\text{base}}$) due to market constraints
- **Price risk**: Fully exposed to spot market price volatility

**3. Production and Sales Decisions ($x_{t,s}$, $y_{t,s}$)**

Operational decisions on actual production volumes and sales to meet demand:

- **Production quantity** ($x_{t,s}$): How much steel to produce given capacity and material availability
- **Sales volume** ($y_{t,s}$): How much to sell (no inventory holding assumed)
- **Unmet demand** ($u_{t,s}$): Demand not satisfied, incurring penalty costs

### Sources of Uncertainty

The planning problem is complicated by three interrelated sources of market uncertainty:

1. **Steel Price Volatility** ($P_t$): Fluctuations in steel market prices affect revenue per ton sold
2. **Scrap Cost Volatility** ($C_t$): Variations in raw material costs impact production economics
3. **Demand Uncertainty** ($D_t$): Unpredictable customer demand influences production volumes and capacity utilization

These uncertainties are **correlated and exhibit temporal dependencies**, necessitating a sophisticated scenario generation approach using Vector Autoregression (VAR) models.

### Optimization Objective

**Maximize expected cumulative profit** over the planning horizon:

$$\mathbb{E}\left[\sum_{t=1}^{T} \Pi_t \right] = \mathbb{E}\left[\text{Revenue} - \text{Costs} - \text{Penalties}\right]$$

Subject to:
- Capacity constraints (base + flexible)
- Material availability constraints (base contract + spot procurement)
- Demand satisfaction requirements (with penalties for unmet demand)
- Recourse flexibility bounds ($\gamma_{\text{cap}}$, $\gamma_{\text{scrap}}$)

### Key Trade-offs

The optimization problem balances several competing objectives:

1. **Capacity sizing**: Over-investment in base capacity leads to high fixed costs and low utilization; under-investment requires expensive flexible capacity
2. **Procurement strategy**: Large contracts provide security but incur high premiums; small contracts require expensive spot market purchases
3. **Risk vs. cost**: Conservative strategies (high base capacity, large contracts) reduce operational risk but increase upfront costs
4. **Flexibility value**: Recourse options (flexible capacity, spot procurement) provide adaptation capability but at premium costs

The two-stage stochastic programming framework explicitly models these trade-offs, finding the optimal balance between commitment (Stage 1) and flexibility (Stage 2) under uncertainty.

---

## Model Description

The Two-Stage Capacity and Procurement Planning model is a stochastic optimization framework that implements the problem formulation described above. It uses Vector Autoregression (VAR) to capture temporal dependencies between steel prices, scrap costs, and demand, generating realistic scenarios for future periods.

### Overview

The model follows a two-stage stochastic programming approach:

1. **First Stage (Here-and-Now Decisions)**: Determine base capacity and procurement contracts for each period before uncertainty is revealed
2. **Second Stage (Wait-and-See Decisions)**: After observing realized scenarios, decide on flexible capacity usage, spot procurement, production, and sales levels

The computational workflow integrates econometric forecasting with mathematical optimization to support robust decision-making under uncertainty.

---

### Data Sources & Loading

#### Data Description

The model utilizes three key time series from the Federal Reserve Economic Data (FRED) database:

1. **Steel Price (P)**: Producer Price Index for steel products
2. **Scrap Cost (C)**: Producer Price Index for scrap metal
3. **Steel Demand (D)**: Industrial production index for steel products

These series are collected at monthly frequency and represent the primary sources of uncertainty in steel industry planning.

#### Mathematical Representation

Let the observed data be represented as:

$$\mathbf{X}_t = \begin{bmatrix} P_t \\ C_t \\ D_t \end{bmatrix} \in \mathbb{R}^3$$

where:
- $P_t$ = Steel price index at time $t$
- $C_t$ = Scrap cost index at time $t$
- $D_t$ = Demand/production index at time $t$

#### Data Processing Pipeline

1. **Data Retrieval**: Series downloaded via FRED API
2. **Resampling**: All series converted to monthly frequency (month start) using mean aggregation
3. **Alignment**: Data aligned to common time periods where all three variables are available
4. **Missing Value Handling**: Periods with any missing values are removed

---

### VAR Model: Selection, Fitting, and Testing

#### Log Returns Transformation

To achieve stationarity and capture relative changes, the model works with log returns:

$$\Delta \log \mathbf{X}_t = \log \mathbf{X}_t - \log \mathbf{X}_{t-1} = \begin{bmatrix} \Delta \log P_t \\ \Delta \log C_t \\ \Delta \log D_t \end{bmatrix}$$

where:
- $\Delta \log P_t = \log(P_t / P_{t-1})$ represents the continuously compounded return on steel prices
- $\Delta \log C_t = \log(C_t / C_{t-1})$ represents the continuously compounded return on scrap costs
- $\Delta \log D_t = \log(D_t / D_{t-1})$ represents the growth rate in demand

#### Vector Autoregression (VAR) Model

The VAR(p) model captures the interdependencies between the three time series:

$$\Delta \log \mathbf{X}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \Delta \log \mathbf{X}_{t-i} + \boldsymbol{\varepsilon}_t$$

where:
- $\mathbf{c} \in \mathbb{R}^3$ is the vector of intercepts
- $\mathbf{A}_i \in \mathbb{R}^{3 \times 3}$ are coefficient matrices for lag $i$
- $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ is the error term vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ is the covariance matrix of innovations
- $p$ is the lag order

Explicitly, for each variable:

$$\begin{aligned}
\Delta \log P_t &= c_P + \sum_{i=1}^{p} (a_{PP,i} \Delta \log P_{t-i} + a_{PC,i} \Delta \log C_{t-i} + a_{PD,i} \Delta \log D_{t-i}) + \varepsilon_{P,t} \\
\Delta \log C_t &= c_C + \sum_{i=1}^{p} (a_{CP,i} \Delta \log P_{t-i} + a_{CC,i} \Delta \log C_{t-i} + a_{CD,i} \Delta \log D_{t-i}) + \varepsilon_{C,t} \\
\Delta \log D_t &= c_D + \sum_{i=1}^{p} (a_{DP,i} \Delta \log P_{t-i} + a_{DC,i} \Delta \log C_{t-i} + a_{DD,i} \Delta \log D_{t-i}) + \varepsilon_{D,t}
\end{aligned}$$

#### Lag Order Selection

The optimal lag order $p^*$ is selected by minimizing an information criterion:

$$p^* = \arg\min_{p \in \{1, \ldots, p_{\max}\}} \text{IC}(p)$$

Common information criteria:
- **AIC (Akaike)**: $\text{AIC}(p) = \log|\hat{\boldsymbol{\Sigma}}_p| + \frac{2pk^2}{T}$
- **BIC (Bayesian)**: $\text{BIC}(p) = \log|\hat{\boldsymbol{\Sigma}}_p| + \frac{pk^2 \log T}{T}$
- **HQIC (Hannan-Quinn)**: $\text{HQIC}(p) = \log|\hat{\boldsymbol{\Sigma}}_p| + \frac{2pk^2 \log \log T}{T}$

where:
- $k = 3$ is the number of variables
- $T$ is the sample size
- $\hat{\boldsymbol{\Sigma}}_p$ is the estimated covariance matrix for lag order $p$

#### Model Diagnostics

After fitting, the model performs several diagnostic tests:

1. **Residual Correlation Test**: Verifies that residuals are uncorrelated (white noise)
2. **Impulse Response Functions (IRF)**: Analyzes how shocks propagate through the system
3. **Forecast Error Variance Decomposition**: Determines the contribution of each variable to forecast uncertainty

---

### Scenario Generation & Reduction

#### Scenario Generation via Monte Carlo Simulation

Starting from the fitted VAR model at decision time $t_0$, we generate scenarios for future periods $t_0+1, \ldots, t_0+H$ where $H$ is the planning horizon.

**Step 1: Generate Innovation Shocks**

For scenario $\omega \in \{1, \ldots, N\}$ and period $h \in \{1, \ldots, H\}$:

$$\boldsymbol{\varepsilon}_h^{\omega} \sim \mathcal{N}(\mathbf{0}, \hat{\boldsymbol{\Sigma}})$$

or from alternative distributions (Student-t, empirical bootstrap, etc.)

**Step 2: Compute Future Returns**

Using the VAR recursion:

$$\Delta \log \mathbf{X}_{t_0+h}^{\omega} = \hat{\mathbf{c}} + \sum_{i=1}^{p} \hat{\mathbf{A}}_i \Delta \log \mathbf{X}_{t_0+h-i}^{\omega} + \boldsymbol{\varepsilon}_h^{\omega}$$

where initial conditions are taken from historical data:

$$\Delta \log \mathbf{X}_{t_0}, \ldots, \Delta \log \mathbf{X}_{t_0-p+1}$$

**Step 3: Reconstruct Levels**

Transform back to levels:

$$\mathbf{X}_{t_0+h}^{\omega} = \mathbf{X}_{t_0+h-1}^{\omega} \cdot \exp(\Delta \log \mathbf{X}_{t_0+h}^{\omega})$$

where $\mathbf{X}_{t_0}$ is the last observed value (anchor point).

This produces scenario matrix:

$$\boldsymbol{\xi}^{\omega} = \begin{bmatrix} 
P_{t_0+1}^{\omega} & C_{t_0+1}^{\omega} & D_{t_0+1}^{\omega} \\
\vdots & \vdots & \vdots \\
P_{t_0+H}^{\omega} & C_{t_0+H}^{\omega} & D_{t_0+H}^{\omega}
\end{bmatrix} \in \mathbb{R}^{H \times 3}$$

#### Scenario Reduction (K-Medoids Clustering)

To manage computational complexity, the $N$ scenarios are reduced to $K \ll N$ representative scenarios using K-medoids clustering.

**Objective**: Find $K$ medoid scenarios that minimize total dissimilarity:

$$\min_{\mathcal{M} \subset \Omega, |\mathcal{M}| = K} \sum_{\omega=1}^{N} \min_{m \in \mathcal{M}} d(\boldsymbol{\xi}^{\omega}, \boldsymbol{\xi}^m)$$

where $d(\cdot, \cdot)$ is a distance metric (typically Euclidean distance across all periods and variables).

**Updated Probabilities**: Each reduced scenario $m$ inherits the probability mass of all scenarios assigned to it:

$$\tilde{p}_m = \sum_{\omega: \text{closest}(\omega) = m} p_{\omega}$$

For equally probable initial scenarios: $\tilde{p}_m = \frac{|\{\omega: \text{closest}(\omega) = m\}|}{N}$

**Stress Scenarios**: Optionally add extreme scenarios (high/low quantiles) to ensure tail risks are represented:

$$\mathcal{M}_{\text{stress}} = \{\omega: \boldsymbol{\xi}^{\omega} \in \text{extreme quantiles}\}$$

---

### Stochastic Program

#### Decision Variables

**First-Stage Variables** (decided at time $t$ before observing scenarios):
- $\text{Cap}_{\text{base},t}$: Base capacity at period $t$ [tons/month]
- $Q_{\text{base},t}$: Base scrap contract quantity at period $t$ [tons/month]

**Second-Stage Variables** (decided after observing scenario $s$ in period $t$):
- $\text{Cap}_{\text{flex},t,s}$: Flexible capacity in period $t$, scenario $s$ [tons/month]
- $x_{t,s}$: Production quantity in period $t$, scenario $s$ [tons/month]
- $y_{t,s}$: Sales quantity in period $t$, scenario $s$ [tons/month]
- $u_{t,s}$: Unmet demand in period $t$, scenario $s$ [tons/month]
- $q_{\text{base},t,s}$: Scrap called from base contract in period $t$, scenario $s$ [tons/month]
- $q_{\text{spot},t,s}$: Spot scrap procurement in period $t$, scenario $s$ [tons/month]

#### Parameters

**Cost Parameters**:
- $c_{\text{var}}$: Variable production cost per unit [€/ton]
- $c_{\text{cap,base}}$: Fixed cost per unit of base capacity per period [€/ton/month]
- $c_{\text{cap,flex}}$: Fixed cost per unit of flexible capacity per period [€/ton/month]
- $\delta_{\text{base}}$: Premium cost for base scrap procurement option [€/ton]
- $\delta_{\text{spot}}$: Premium cost for spot scrap procurement [€/ton]
- $\text{pen}_{\text{unmet}}$: Penalty cost per unit of unmet demand [€/ton]

**Technical Parameters**:
- $\alpha$: Scrap consumption rate per unit of production (≥ 1)
- $\gamma_{\text{cap}}$: Maximum flexible capacity as fraction of base capacity (0 ≤ γ_cap ≤ 1)
- $\gamma_{\text{scrap}}$: Maximum spot scrap procurement as fraction of base contract (0 ≤ γ_scrap ≤ 2)

**Scenario Parameters**:
- $D_{t,s}$: Demand in period $t$ under scenario $s$ [tons/month]
- $P_{t,s}$: Steel selling price in period $t$ under scenario $s$ [€/ton]
- $C_{t,s}$: Scrap cost in period $t$ under scenario $s$ [€/ton]
- $p_s$: Probability of scenario $s$ (Σ_s p_s = 1)

#### Objective Function

Maximize expected profit over all time periods and scenarios:

$$\begin{aligned}
\max \quad & \mathbb{E}[\text{Profit}] = \sum_{s \in S} p_s \left[ \sum_{t \in T} \left( P_{t,s} \cdot y_{t,s} - C_{t,s} \cdot (q_{\text{base},t,s} + q_{\text{spot},t,s}) \right. \right. \\
& \left. \left. \quad - c_{\text{var}} \cdot x_{t,s} - c_{\text{cap,flex}} \cdot \text{Cap}_{\text{flex},t,s} - \delta_{\text{spot}} \cdot q_{\text{spot},t,s} - \text{pen}_{\text{unmet}} \cdot u_{t,s} \right) \right] \\
& \quad - \sum_{t \in T} \left( c_{\text{cap,base}} \cdot \text{Cap}_{\text{base},t} + \delta_{\text{base}} \cdot Q_{\text{base},t} \right)
\end{aligned}$$

**Decomposition**:

**Stage 1 Costs** (deterministic, paid upfront):
Sum over t ∈ T of  
c_cap,base · Cap_base,t + δ_base · Q_base,t

**Expected Stage 2 Profit** (stochastic, scenario-dependent):
- Revenue from sales: $P_{t,s} \cdot y_{t,s}$
- Scrap costs: $C_{t,s} \cdot (q_{\text{base},t,s} + q_{\text{spot},t,s})$
- Variable production costs: $c_{\text{var}} \cdot x_{t,s}$
- Flexible capacity costs: $c_{\text{cap,flex}} \cdot \text{Cap}_{\text{flex},t,s}$
- Spot procurement premium: $\delta_{\text{spot}} \cdot q_{\text{spot},t,s}$
- Unmet demand penalty: pen_unmet · u_t,s

#### Constraints

**All variables non-negative**:
Cap_base,t, Q_base,t, Cap_flex,t,s, x_t,s, y_t,s, u_t,s,  
q_base,t,s, q_spot,t,s ≥ 0 ∀ t, s

**Second-Stage Constraints** (for each scenario $s$ and period $t$):

1. **Demand balance**:
$$y_{t,s} + u_{t,s} = D_{t,s} \quad \forall t, s$$
Sales plus unmet demand must equal total demand.

2. **No finished-goods inventory**:
$$y_{t,s} \leq x_{t,s} \quad \forall t, s$$
Sales cannot exceed production (no inventory).

3. **Capacity constraint**:
$$x_{t,s} \leq Cap_{base,t} + Cap_{flex,t,s} \quad \forall t, s$$
Production limited by total available capacity.

4. **Scrap balance**:
$$\alpha \cdot x_{t,s} = q_{\text{base},t,s} + q_{\text{spot},t,s} \quad \forall t, s$$
Scrap consumption equals total scrap procurement.

5. **Base contract limit**:
$$q_{\text{base},t,s} \leq Q_{\text{base},t} \quad \forall t, s$$
Base scrap usage limited by contract quantity.

6. **Flexible capacity bound**:
$$Cap_{flex,t,s} \leq \gamma_{cap} \cdot Cap_{base,t} \quad \forall t, s$$
Flexible capacity limited as fraction of base capacity.

7. **Spot procurement bound**:
$$q_{\text{spot},t,s} \leq \gamma_{\text{scrap}} \cdot Q_{\text{base},t} \quad \forall t, s$$
Spot scrap limited as fraction of base contract.

#### Complete Formulation

$$\begin{aligned}
\max \quad & \sum_{s \in S} p_s \sum_{t \in T} \Big[ P_{t,s} \cdot y_{t,s} - C_{t,s} \cdot (q_{\text{base},t,s} + q_{\text{spot},t,s}) - c_{\text{var}} \cdot x_{t,s} \\
& \qquad - c_{\text{cap,flex}} \cdot \text{Cap}_{\text{flex},t,s} - \delta_{\text{spot}} \cdot q_{\text{spot},t,s} - \text{pen}_{\text{unmet}} \cdot u_{t,s} \Big] \\
& \quad - \sum_{t \in T} (c_{\text{cap,base}} \cdot \text{Cap}_{\text{base},t} + \delta_{\text{base}} \cdot Q_{\text{base},t}) \\[1em]
\text{s.t.} \quad & y_{t,s} + u_{t,s} = D_{t,s}, \quad \forall t, s \\
& y_{t,s} \leq x_{t,s}, \quad \forall t, s \\
& x_{t,s} \leq \text{Cap}_{\text{base},t} + \text{Cap}_{\text{flex},t,s}, \quad \forall t, s \\
& \alpha \cdot x_{t,s} = q_{\text{base},t,s} + q_{\text{spot},t,s}, \quad \forall t, s \\
& q_{\text{base},t,s} \leq Q_{\text{base},t}, \quad \forall t, s \\
& \text{Cap}_{\text{flex},t,s} \leq \gamma_{\text{cap}} \cdot \text{Cap}_{\text{base},t}, \quad \forall t, s \\
& q_{\text{spot},t,s} \leq \gamma_{\text{scrap}} \cdot Q_{\text{base},t}, \quad \forall t, s \\
& \text{All variables} \geq 0
\end{aligned}$$

This is a **linear programming (LP)** problem that can be solved efficiently using standard solvers.

**Key Model Features**:
- **Two-stage structure**: First-stage decisions (Cap_base, Q_base) are made for each period before uncertainty resolves; second-stage decisions adapt to realized scenarios
- **Recourse flexibility**: Bounded flexibility through γ_cap and γ_scrap parameters prevents unlimited recourse
- **Economic trade-offs**: Balances fixed capacity/contract costs against variable production costs and recourse flexibility
- **No inventory**: Simplified by assuming no finished-goods storage (y ≤ x constraint)

---

### Expected Profit Analysis

#### Expected Profit Decomposition

The expected profit can be decomposed into components:

$$\mathbb{E}[\Pi] = \underbrace{-c_{\text{cap,base}} \cdot x_{\text{base}}^* - \delta_{\text{base}} \cdot y_{\text{contract}}^*}_{\text{First-stage costs}} + \underbrace{\sum_{\omega=1}^{K} p_{\omega} \Pi^{\omega}}_{\text{Expected second-stage profit}}$$

where:

$$\Pi^{\omega} = -c_{\text{cap,flex}} \cdot x_{\text{flex}}^{\omega*} + \sum_{h=1}^{H} \left[ \text{Revenue}^{\omega,h} - \text{Cost}^{\omega,h} \right]$$

#### Risk Metrics

**Conditional Value-at-Risk (CVaR)**:

$$\text{CVaR}_{\alpha}(\Pi) = \mathbb{E}[\Pi \mid \Pi \leq \text{VaR}_{\alpha}(\Pi)]$$

where $\text{VaR}_{\alpha}$ is the $\alpha$-quantile of the profit distribution.

**Profit Distribution Moments**:
- **Mean**: $\mu_{\Pi} = \mathbb{E}[\Pi]$
- **Variance**: $\sigma_{\Pi}^2 = \mathbb{E}[(\Pi - \mu_{\Pi})^2]$
- **Sharpe Ratio**: $\text{SR} = \frac{\mu_{\Pi}}{\sigma_{\Pi}}$ (if baseline profit is zero)

#### Backtesting

To validate model performance, backtesting simulates the execution of first-stage decisions under realized scenarios:

1. **Historical Simulation**: Roll forward through historical data
2. **Decision Fixing**: Use optimized first-stage decisions
3. **Realization**: Observe actual prices and demand
4. **Second-Stage Adjustment**: Solve for optimal recourse actions given realizations
5. **Performance Metrics**: Calculate realized profit, tracking error, and decision quality

---

## Usage

### Method Reference

#### 1. `load_data_from_fredapi`

**Description**: Loads steel industry time series data from the FRED API.

**Inputs**:
- `api_key` (str): Your FRED API key ([Get one here](https://fred.stlouisfed.org/docs/api/api_key.html))
- `steel_price_identifier` (str, optional): FRED series ID for steel prices (default: 'WPU101704')
- `scrap_price_identifier` (str, optional): FRED series ID for scrap costs (default: 'WPU1012')
- `steel_demand_identifier` (str, optional): FRED series ID for steel demand (default: 'IPG3311A2S')
- `plot_data` (bool, optional): Whether to plot the loaded data (default: True)

**Returns**: `pd.DataFrame` with columns ['P', 'C', 'D'] and DatetimeIndex

**Example**:
```python
from src.models.basic import TwoStageCapacityAndProcurementPlanning

# Load data with default series
data = TwoStageCapacityAndProcurementPlanning.load_data_from_fredapi(
    api_key='your_api_key_here',
    plot_data=True
)

# Use custom series identifiers
data = TwoStageCapacityAndProcurementPlanning.load_data_from_fredapi(
    api_key='your_api_key_here',
    steel_price_identifier='WPU101702',
    scrap_price_identifier='WPU101201',
    steel_demand_identifier='IPG331111CN',
    plot_data=False
)
```

---

#### 2. `get_n_observations`

**Description**: Extracts a subset of historical data for VAR model estimation, useful for backtesting.

**Inputs**:
- `data` (pd.DataFrame): Full historical dataset with DatetimeIndex
- `n` (int, optional): Number of observations to keep (default: 90 * p)
- `p` (int, optional): VAR lag order for empirical rule (default: 2)
- `last_observation` (int/str/datetime, optional): Last observation to include (default: None = use all)
- `plot_data` (bool, optional): Whether to plot the subset (default: False)

**Returns**: `pd.DataFrame` with n observations

**Example**:
```python
# Automatic sample size selection
data_subset = TwoStageCapacityAndProcurementPlanning.get_n_observations(
    data, 
    p=2, 
    plot_data=True
)

# Manual sample size with date cutoff for backtesting
data_subset = TwoStageCapacityAndProcurementPlanning.get_n_observations(
    data,
    n=180,
    last_observation='2019-12-01',
    plot_data=True
)

# Using index position
data_subset = TwoStageCapacityAndProcurementPlanning.get_n_observations(
    data,
    n=150,
    last_observation=200  # First 201 observations
)
```

---

#### 3. `log_returns`

**Description**: Computes log returns (continuously compounded returns) from price/demand levels.

**Inputs**:
- `data` (pd.DataFrame): Price and demand level data
- `plot_data` (bool, optional): Whether to plot the log returns (default: False)
- `print_stats` (bool, optional): Whether to print summary statistics (default: False)

**Returns**: `pd.DataFrame` with log returns

**Example**:
```python
# Compute log returns with diagnostics
log_ret = TwoStageCapacityAndProcurementPlanning.log_returns(
    data_subset,
    plot_data=True,
    print_stats=True
)

# The transformation: Δlog(X_t) = log(X_t) - log(X_{t-1})
# For price: Δlog(P_t) = log(P_t / P_{t-1})
```

---

#### 4. `VAR_order_selection`

**Description**: Selects the optimal VAR lag order using information criteria.

**Inputs**:
- `Δlog` (pd.DataFrame): Log returns data
- `maxlags` (int, optional): Maximum number of lags to consider (default: 12)
- `method` (str, optional): Information criterion to use: 'aic', 'bic', 'hqic', 'fpe' (default: 'bic')

**Returns**: `int` - Optimal lag order p*

**Example**:
```python
# Select optimal lag order using BIC
optimal_p = TwoStageCapacityAndProcurementPlanning.VAR_order_selection(
    log_ret,
    maxlags=12,
    method='bic'
)
print(f"Optimal lag order: {optimal_p}")

# Try different criteria
optimal_p_aic = TwoStageCapacityAndProcurementPlanning.VAR_order_selection(
    log_ret, maxlags=12, method='aic'
)
optimal_p_hqic = TwoStageCapacityAndProcurementPlanning.VAR_order_selection(
    log_ret, maxlags=12, method='hqic'
)
```

---

#### 5. `fit_VAR_model`

**Description**: Fits a VAR model to the data with automatic lag selection and diagnostic testing.

**Inputs**:
- `data` (pd.DataFrame, optional): Raw level data (will compute log returns internally)
- `Δlog` (pd.DataFrame, optional): Pre-computed log returns (use either data or Δlog)
- `p` (int, optional): Lag order (if None, automatically selected)
- `maxlags` (int, optional): Maximum lags for automatic selection (default: 12)
- `method` (str, optional): Information criterion (default: 'bic')
- `testing` (list, optional): Diagnostic tests to run: ['corr', 'irf', 'sim_stats'] (default: all)
- `print_warnings` (bool, optional): Whether to print warnings (default: True)

**Returns**: Fitted VAR model object

**Example**:
```python
# Automatic lag selection and full diagnostics
var_model = TwoStageCapacityAndProcurementPlanning.fit_VAR_model(
    Δlog=log_ret,
    maxlags=12,
    method='bic',
    testing=['corr', 'irf', 'sim_stats']
)

# Manual lag specification
var_model = TwoStageCapacityAndProcurementPlanning.fit_VAR_model(
    data=data_subset,  # Can pass raw data instead of log returns
    p=2,
    testing=['corr']
)

# Skip diagnostics for faster fitting
var_model = TwoStageCapacityAndProcurementPlanning.fit_VAR_model(
    Δlog=log_ret,
    p=2,
    testing=[],
    print_warnings=False
)
```

---

#### 6. `analyze_shock_distributions`

**Description**: Analyzes the distribution of VAR model residuals (shocks) to determine appropriate scenario generation methods.

**Inputs**:
- `var_model`: Fitted VAR model object
- `plot_diagnostics` (bool, optional): Whether to generate diagnostic plots (default: True)

**Returns**: Dictionary with distribution parameters and test results

**Example**:
```python
# Analyze residual distributions
shock_analysis = TwoStageCapacityAndProcurementPlanning.analyze_shock_distributions(
    var_model,
    plot_diagnostics=True
)

# Results include:
# - Normality tests (Jarque-Bera, Shapiro-Wilk)
# - Kurtosis and skewness
# - Recommended distribution for scenario generation
print(shock_analysis)
```

---

#### 7. `generate_future_returns_scenarios`

**Description**: Generates Monte Carlo scenarios for future log returns using the fitted VAR model.

**Inputs**:
- `var_model`: Fitted VAR model object
- `simulation_start_date` (str/datetime): Date from which to start simulation
- `horizon` (int, optional): Number of periods ahead to simulate (default: 12)
- `n_scenarios` (int, optional): Number of scenarios to generate (default: 1000)
- `seed` (int, optional): Random seed for reproducibility (default: 42)
- `shock_distribution` (str, optional): Distribution for innovations: 'normal', 't', 'bootstrap' (default: 'normal')
- `distribution_params` (dict, optional): Additional parameters for non-normal distributions

**Returns**: Dictionary with 'scenarios' (array) and 'prob' (array of probabilities)

**Example**:
```python
# Generate scenarios with normal shocks
scenarios_returns = TwoStageCapacityAndProcurementPlanning.generate_future_returns_scenarios(
    var_model=var_model,
    simulation_start_date='2020-01-01',
    horizon=12,
    n_scenarios=1000,
    seed=42,
    shock_distribution='normal'
)

# Use Student-t distribution for fat tails
scenarios_returns = TwoStageCapacityAndProcurementPlanning.generate_future_returns_scenarios(
    var_model=var_model,
    simulation_start_date='2020-01-01',
    horizon=12,
    n_scenarios=1000,
    shock_distribution='t',
    distribution_params={'df': 5}  # degrees of freedom
)

# Bootstrap from historical residuals
scenarios_returns = TwoStageCapacityAndProcurementPlanning.generate_future_returns_scenarios(
    var_model=var_model,
    simulation_start_date='2020-01-01',
    horizon=12,
    n_scenarios=1000,
    shock_distribution='bootstrap'
)
```

---

#### 8. `reconstruct_levels_from_returns`

**Description**: Converts log return scenarios back to price/demand level scenarios.

**Inputs**:
- `scenario_returns` (dict): Output from `generate_future_returns_scenarios`
- `historical_data` (pd.DataFrame): Historical data for anchoring
- `anchor_date` (str/datetime, optional): Date to use as starting point (default: last date in historical data)
- `real_prices` (pd.DataFrame, optional): Actual realized prices for comparison
- `use_demand_scaling` (bool, optional): Whether to scale demand scenarios (default: True)

**Returns**: Dictionary with 'scenarios' (levels) and 'prob' arrays

**Example**:
```python
# Reconstruct level scenarios
scenarios = TwoStageCapacityAndProcurementPlanning.reconstruct_levels_from_returns(
    scenario_returns=scenarios_returns,
    historical_data=data_subset,
    anchor_date='2019-12-01',
    use_demand_scaling=True
)

# With actual prices for validation
scenarios = TwoStageCapacityAndProcurementPlanning.reconstruct_levels_from_returns(
    scenario_returns=scenarios_returns,
    historical_data=data_subset,
    real_prices=actual_data,  # For comparison
    use_demand_scaling=True
)

# Access scenarios: scenarios['scenarios'] has shape (n_scenarios, horizon, 3)
# scenarios['scenarios'][omega, h, :] = [P, C, D] for scenario omega at period h
```

---

#### 9. `plot_scenarios_evolution`

**Description**: Visualizes the evolution of scenarios over the planning horizon with confidence intervals.

**Inputs**:
- `scenarios` (dict): Scenario data from `reconstruct_levels_from_returns`
- `historical_data` (pd.DataFrame): Historical data for context
- `prob` (np.array): Scenario probabilities
- `max_number_of_scenarios` (int, optional): Maximum number of trajectories to plot (default: 50)
- `max_history` (int, optional): Number of historical periods to show (default: 36)
- `future_trajectory` (pd.DataFrame, optional): Specific future path to highlight
- `real_prices` (pd.DataFrame, optional): Actual realized values to overlay
- `figsize` (tuple, optional): Figure size (default: (18, 6))
- `title_prefix` (str, optional): Prefix for plot titles
- `save_path` (str, optional): Path to save the figure
- `show_statistics` (bool, optional): Whether to show statistical summary (default: True)

**Returns**: matplotlib figure object

**Example**:
```python
# Basic visualization
fig = TwoStageCapacityAndProcurementPlanning.plot_scenarios_evolution(
    scenarios=scenarios,
    historical_data=data_subset,
    prob=scenarios['prob'],
    max_number_of_scenarios=50,
    show_statistics=True
)

# With actual future data for comparison
fig = TwoStageCapacityAndProcurementPlanning.plot_scenarios_evolution(
    scenarios=scenarios,
    historical_data=data_subset,
    prob=scenarios['prob'],
    real_prices=actual_future_data,
    title_prefix="Backtest ",
    save_path="results/scenario_plot.png"
)
```

---

#### 10. `reduce_scenarios_kmedoids`

**Description**: Reduces the number of scenarios using K-medoids clustering to manage computational complexity.

**Inputs**:
- `scenarios` (dict): Full scenario set from `reconstruct_levels_from_returns`
- `prob` (np.array): Original scenario probabilities
- `n_scenario_clusters` (int, optional): Target number of scenarios (default: 50)
- `stress_pct` (float, optional): Percentage of extreme scenarios to include (default: 0.01)
- `seed` (int, optional): Random seed for reproducibility (default: 42)

**Returns**: Dictionary with reduced 'scenarios' and updated 'prob' arrays

**Example**:
```python
# Reduce from 1000 to 50 scenarios
scenarios_reduced = TwoStageCapacityAndProcurementPlanning.reduce_scenarios_kmedoids(
    scenarios=scenarios,
    prob=scenarios['prob'],
    n_scenario_clusters=50,
    stress_pct=0.01,
    seed=42
)

# More aggressive reduction
scenarios_reduced = TwoStageCapacityAndProcurementPlanning.reduce_scenarios_kmedoids(
    scenarios=scenarios,
    prob=scenarios['prob'],
    n_scenario_clusters=20,
    stress_pct=0.02  # Include more stress scenarios
)

# Access reduced scenarios
print(f"Original: {scenarios['scenarios'].shape[0]} scenarios")
print(f"Reduced: {scenarios_reduced['scenarios'].shape[0]} scenarios")
```

---

#### 11. `optimize_capacity_and_procurement`

**Description**: Solves the two-stage stochastic programming problem to find optimal capacity and procurement decisions.

**Inputs**:
- `scenarios` (dict): Scenario data (can be full or reduced)
- `prob` (np.array): Scenario probabilities
- `alpha` (float): Existing capacity [units]
- `c_var` (float): Variable production cost [$/unit]
- `c_cap_base` (float): Base capacity expansion cost [$/unit]
- `c_cap_flex` (float): Flexible capacity cost [$/unit]
- `delta_base` (float): Long-term contract procurement cost [$/unit]
- `delta_spot` (float): Spot procurement cost multiplier
- `pen_unmet` (float): Penalty for unmet demand [$/unit]
- `gamma_cap` (float): Steel production per unit capacity
- `gamma_scrap` (float): Scrap requirement per unit steel produced
- `solver` (str, optional): LP solver to use: 'highs', 'glpk', 'gurobi' (default: 'highs')

**Returns**: Dictionary with optimal decisions and objective value

**Example**:
```python
# Define problem parameters
params = {
    'alpha': 100,              # Existing capacity
    'c_var': 500,             # Variable cost per unit
    'c_cap_base': 10000,      # Base capacity expansion cost
    'c_cap_flex': 8000,       # Flexible capacity cost
    'delta_base': 300,        # Contract procurement cost
    'delta_spot': 1.2,        # Spot price multiplier
    'pen_unmet': 2000,        # Unmet demand penalty
    'gamma_cap': 1.0,         # Production per capacity unit
    'gamma_scrap': 0.8        # Scrap per steel unit
}

# Solve optimization problem
decisions = TwoStageCapacityAndProcurementPlanning.optimize_capacity_and_procurement(
    scenarios=scenarios_reduced,
    prob=scenarios_reduced['prob'],
    solver='highs',
    **params
)

# Access results
print(f"Base capacity expansion: {decisions['x_base']}")
print(f"Contract procurement: {decisions['y_contract']}")
print(f"Expected profit: {decisions['expected_profit']}")
print(f"First-stage cost: {decisions['first_stage_cost']}")
print(f"Expected second-stage profit: {decisions['expected_second_stage']}")

# Second-stage decisions per scenario
flex_capacity = decisions['x_flex']  # Shape: (n_scenarios,)
spot_procurement = decisions['y_spot']  # Shape: (n_scenarios, horizon)
production = decisions['z']  # Shape: (n_scenarios, horizon)
unmet_demand = decisions['u']  # Shape: (n_scenarios, horizon)
```

---

#### 12. `backtesting_simulation`

**Description**: Simulates the execution of first-stage decisions under historical realizations to validate model performance.

**Inputs**:
- `decisions` (dict): Optimal decisions from `optimize_capacity_and_procurement`
- `actual_future_data` (pd.DataFrame): Realized historical prices and demand
- `alpha` (float): Existing capacity
- `c_var`, `c_cap_base`, `c_cap_flex`, `delta_base`, `delta_spot`, `pen_unmet`, `gamma_cap`, `gamma_scrap` (float): Same parameters as optimization
- `real_prices` (pd.DataFrame, optional): Additional price data for visualization
- `plot_results` (bool, optional): Whether to generate diagnostic plots (default: True)
- `figsize` (tuple, optional): Figure size (default: (15, 10))

**Returns**: Dictionary with realized profit and performance metrics

**Example**:
```python
# Prepare actual future data (next 12 months after decision point)
actual_data = data.loc['2020-01-01':'2020-12-01']

# Run backtest
backtest_results = TwoStageCapacityAndProcurementPlanning.backtesting_simulation(
    decisions=decisions,
    actual_future_data=actual_data,
    plot_results=True,
    figsize=(15, 10),
    **params  # Same parameters as optimization
)

# Analyze results
print(f"Realized profit: {backtest_results['realized_profit']}")
print(f"Expected profit: {backtest_results['expected_profit']}")
print(f"Tracking error: {backtest_results['tracking_error']}")
print(f"Average capacity utilization: {backtest_results['avg_utilization']:.2%}")
print(f"Total unmet demand: {backtest_results['total_unmet']}")

# Period-by-period details
period_profits = backtest_results['period_profits']
period_production = backtest_results['period_production']
```

---

#### 13. `plot_profit_distribution_over_time`

**Description**: Analyzes and visualizes the distribution of profits across scenarios over time, including confidence intervals and risk metrics.

**Inputs**:
- `decisions` (dict): Optimal decisions from optimization
- `scenarios` (dict): Scenario data used in optimization
- `prob` (np.array): Scenario probabilities
- `alpha`, `c_var`, `c_cap_base`, `c_cap_flex`, `delta_base`, `delta_spot`, `pen_unmet`, `gamma_cap`, `gamma_scrap` (float): Model parameters
- `confidence_levels` (list, optional): Confidence interval levels (default: [0.05, 0.95])
- `figsize` (tuple, optional): Figure size (default: (16, 8))
- `save_path` (str, optional): Path to save figure
- `show_stats` (bool, optional): Whether to display statistics (default: True)

**Returns**: Dictionary with profit statistics and matplotlib figure

**Example**:
```python
# Analyze profit distribution
profit_analysis = TwoStageCapacityAndProcurementPlanning.plot_profit_distribution_over_time(
    decisions=decisions,
    scenarios=scenarios_reduced,
    prob=scenarios_reduced['prob'],
    confidence_levels=[0.05, 0.95],  # 5th and 95th percentiles
    figsize=(16, 8),
    save_path='results/profit_distribution.png',
    show_stats=True,
    **params
)

# Access statistics
print(f"Mean profit per period: {profit_analysis['mean_profit']}")
print(f"Std dev profit per period: {profit_analysis['std_profit']}")
print(f"5th percentile (VaR): {profit_analysis['percentile_05']}")
print(f"95th percentile: {profit_analysis['percentile_95']}")
print(f"Worst-case scenario profit: {profit_analysis['worst_case']}")
print(f"Best-case scenario profit: {profit_analysis['best_case']}")

# CVaR (Conditional Value at Risk)
cvar_5 = profit_analysis['cvar_05']
```

---

### Complete Workflow Example

```python
from src.models.basic import TwoStageCapacityAndProcurementPlanning
import pandas as pd

# ========================================
# Step 1: Load and Prepare Data
# ========================================

# Load historical data
data = TwoStageCapacityAndProcurementPlanning.load_data_from_fredapi(
    api_key='your_fred_api_key',
    plot_data=True
)

# Select training data (e.g., last 180 months)
data_subset = TwoStageCapacityAndProcurementPlanning.get_n_observations(
    data, 
    n=180, 
    last_observation='2019-12-01',  # Decision point
    plot_data=True
)

# ========================================
# Step 2: Fit VAR Model
# ========================================

# Compute log returns
log_ret = TwoStageCapacityAndProcurementPlanning.log_returns(
    data_subset, 
    plot_data=True, 
    print_stats=True
)

# Fit VAR model with automatic lag selection
var_model = TwoStageCapacityAndProcurementPlanning.fit_VAR_model(
    Δlog=log_ret,
    maxlags=12,
    method='bic',
    testing=['corr', 'irf', 'sim_stats']
)

# Analyze residuals
shock_analysis = TwoStageCapacityAndProcurementPlanning.analyze_shock_distributions(
    var_model, 
    plot_diagnostics=True
)

# ========================================
# Step 3: Generate Scenarios
# ========================================

# Generate 1000 scenarios for 12 months ahead
scenarios_returns = TwoStageCapacityAndProcurementPlanning.generate_future_returns_scenarios(
    var_model=var_model,
    simulation_start_date='2020-01-01',
    horizon=12,
    n_scenarios=1000,
    seed=42,
    shock_distribution='normal'
)

# Reconstruct price/demand levels
scenarios = TwoStageCapacityAndProcurementPlanning.reconstruct_levels_from_returns(
    scenario_returns=scenarios_returns,
    historical_data=data_subset,
    anchor_date='2019-12-01',
    use_demand_scaling=True
)

# Visualize scenarios
TwoStageCapacityAndProcurementPlanning.plot_scenarios_evolution(
    scenarios=scenarios,
    historical_data=data_subset,
    prob=scenarios['prob'],
    max_number_of_scenarios=50,
    show_statistics=True
)

# ========================================
# Step 4: Reduce Scenarios (Optional)
# ========================================

scenarios_reduced = TwoStageCapacityAndProcurementPlanning.reduce_scenarios_kmedoids(
    scenarios=scenarios,
    prob=scenarios['prob'],
    n_scenario_clusters=50,
    stress_pct=0.01,
    seed=42
)

# ========================================
# Step 5: Optimize Capacity & Procurement
# ========================================

# Define parameters
params = {
    'alpha': 100,
    'c_var': 500,
    'c_cap_base': 10000,
    'c_cap_flex': 8000,
    'delta_base': 300,
    'delta_spot': 1.2,
    'pen_unmet': 2000,
    'gamma_cap': 1.0,
    'gamma_scrap': 0.8
}

# Solve optimization
decisions = TwoStageCapacityAndProcurementPlanning.optimize_capacity_and_procurement(
    scenarios=scenarios_reduced,
    prob=scenarios_reduced['prob'],
    solver='highs',
    **params
)

print(f"Optimal base capacity expansion: {decisions['x_base']:.2f}")
print(f"Optimal contract procurement: {decisions['y_contract']:.2f}")
print(f"Expected profit: ${decisions['expected_profit']:,.2f}")

# ========================================
# Step 6: Analyze Results
# ========================================

# Profit distribution analysis
profit_analysis = TwoStageCapacityAndProcurementPlanning.plot_profit_distribution_over_time(
    decisions=decisions,
    scenarios=scenarios_reduced,
    prob=scenarios_reduced['prob'],
    confidence_levels=[0.05, 0.95],
    save_path='results/profit_dist.png',
    **params
)

# ========================================
# Step 7: Backtest (if historical data available)
# ========================================

# Get actual future data
actual_future = data.loc['2020-01-01':'2020-12-01']

# Run backtest
backtest_results = TwoStageCapacityAndProcurementPlanning.backtesting_simulation(
    decisions=decisions,
    actual_future_data=actual_future,
    plot_results=True,
    **params
)

print(f"Realized profit: ${backtest_results['realized_profit']:,.2f}")
print(f"Expected vs. Realized difference: ${backtest_results['tracking_error']:,.2f}")
```

---

## References

1. **Vector Autoregression**: Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
2. **Stochastic Programming**: Birge, J. R., & Louveaux, F. (2011). *Introduction to Stochastic Programming*. Springer.
3. **Scenario Generation**: Dupačová, J., Consigli, G., & Wallace, S. W. (2000). Scenarios for multistage stochastic programs. *Annals of Operations Research*, 100(1), 25-53.
4. **Scenario Reduction**: Heitsch, H., & Römisch, W. (2003). Scenario reduction algorithms in stochastic programming. *Computational optimization and applications*, 24(2), 187-206.

---

*Last updated: December 26, 2025*
