# Mathematical Formulation

This document covers the complete mathematics behind the two components of the framework: (1) the **VAR-based scenario generation pipeline** and (2) the **two-stage stochastic optimization model**.

---

## Part 1: Scenario Generation

### 1.1 Data Representation

Let $\mathbf{p}_t \in \mathbb{R}^k$ denote the vector of $k$ observed economic variables at time $t$:

$$\mathbf{p}_t = \begin{pmatrix} P_t \\ C_t \\ D_t \end{pmatrix}$$

where $P_t$ is the steel selling price index, $C_t$ is the scrap cost index, and $D_t$ is the demand index, all at monthly frequency $t = 1, \ldots, N$.

Because commodity price series are non-stationary (they trend and have time-varying variance), all model fitting is performed on **log returns**:

$$r_{i,t} = \log\!\left(\frac{p_{i,t}}{p_{i,t-1}}\right), \quad i \in \{P, C, D\}$$

Let $\mathbf{r}_t = (r_{P,t},\ r_{C,t},\ r_{D,t})^\top$ denote the vector of concurrent log returns.

---

### 1.2 Vector Autoregression Model

A VAR model of order $p$ captures the joint serial dependence in the returns:

$$\mathbf{r}_t = \boldsymbol{\mu} + \sum_{\ell=1}^{p} \mathbf{A}_\ell\, \mathbf{r}_{t-\ell} + \boldsymbol{\varepsilon}_t, \qquad t = p+1, \ldots, N$$

where:

- $\boldsymbol{\mu} \in \mathbb{R}^k$ is the intercept vector (drift)
- $\mathbf{A}_\ell \in \mathbb{R}^{k \times k}$ is the coefficient matrix at lag $\ell$
- $\boldsymbol{\varepsilon}_t \sim \mathcal{D}(\mathbf{0}, \boldsymbol{\Sigma})$ are i.i.d. innovations with zero mean and covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{k \times k}$

The model is estimated by ordinary least squares (OLS) equation by equation (equivalent to GLS for a VAR with the same regressors). The coefficient matrices and covariance are recovered from standard multivariate regression:

$$\hat{\mathbf{B}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$$

$$\hat{\boldsymbol{\Sigma}} = \frac{1}{T - kp - 1}(\mathbf{Y} - \mathbf{X}\hat{\mathbf{B}})^\top(\mathbf{Y} - \mathbf{X}\hat{\mathbf{B}})$$

where $\mathbf{Y}$ stacks the return vectors and $\mathbf{X}$ stacks the lagged regressor blocks.

#### Stability Condition

A VAR($p$) is covariance-stationary if and only if all eigenvalues of the companion matrix $\mathbf{F}$ lie strictly inside the unit circle:

$$\mathbf{F} = \begin{pmatrix} \mathbf{A}_1 & \mathbf{A}_2 & \cdots & \mathbf{A}_p \\ \mathbf{I}_k & \mathbf{0} & \cdots & \mathbf{0} \\ \vdots & & \ddots & \vdots \\ \mathbf{0} & \cdots & \mathbf{I}_k & \mathbf{0} \end{pmatrix} \in \mathbb{R}^{kp \times kp}$$

$$\text{Stable} \iff \rho(\mathbf{F}) = \max_i |\lambda_i(\mathbf{F})| < 1$$

The implementation checks this condition after fitting and warns if violated.

---

### 1.3 Lag Order Selection

The lag order $p$ is selected by minimizing the Bayesian Information Criterion (BIC) over candidate orders $p \in \{1, \ldots, p_{\max}\}$:

$$\text{BIC}(p) = \log|\hat{\boldsymbol{\Sigma}}_p| + \frac{\log(T)}{T} \cdot k^2 p$$

BIC is preferred over AIC for longer samples because its stronger penalty on model complexity discourages overfitting and typically selects $p \leq 3$ on monthly commodity data.

---

### 1.4 Shock Distribution

The innovation covariance $\boldsymbol{\Sigma}$ is decomposed via Cholesky factorization:

$$\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$$

Correlated innovations for scenario simulation are constructed as:

$$\boldsymbol{\varepsilon}_t = \mathbf{L}\, \boldsymbol{\eta}_t$$

where $\boldsymbol{\eta}_t \in \mathbb{R}^k$ is a vector of i.i.d. draws from a chosen univariate distribution. Available distributions:

| Distribution | Parameterization | Typical Use |
|-------------|-----------------|-------------|
| Normal | $\mathcal{N}(0,1)$ | Baseline; assumes symmetric, thin-tailed shocks |
| Student-*t* | $t(\nu)$, `df` parameter | Recommended — captures fat tails in commodity returns |
| Skewed-*t* | $ST(\nu, \gamma)$, `df` + `skew` | Asymmetric tails (downside-heavy) |
| Laplace | $\text{Lap}(0,1)$ | Very fat tails |

Empirical residual analysis typically finds commodity log returns have excess kurtosis ($\hat{\kappa} > 3$), supporting the use of Student-*t* shocks with $\nu \approx 5$–7.

---

### 1.5 Forward Simulation

Given the fitted VAR and a simulation start date $\tau$, $S$ forward trajectories of length $H$ (horizon) are generated as follows.

**Initialization**: Set the initial condition to the last $p$ observed return vectors:

$$\hat{\mathbf{r}}_{\tau-p+1}, \ldots, \hat{\mathbf{r}}_\tau \quad \text{(from training data)}$$

**Recursive simulation**: For scenario $\omega = 1, \ldots, S$ and horizon step $h = 1, \ldots, H$:

$$\tilde{\mathbf{r}}_h^{(\omega)} = \hat{\boldsymbol{\mu}} + \sum_{\ell=1}^{p} \hat{\mathbf{A}}_\ell\, \tilde{\mathbf{r}}_{h-\ell}^{(\omega)} + \mathbf{L}\, \boldsymbol{\eta}_h^{(\omega)}$$

where $\boldsymbol{\eta}_h^{(\omega)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{D}(\mathbf{0}, \mathbf{I})$ are freshly drawn innovations.

**Level reconstruction**: The simulated return paths are converted back to price/quantity levels using the last observed price as the base:

$$\tilde{p}_{i,h}^{(\omega)} = p_{i,\tau} \cdot \exp\!\left(\sum_{j=1}^{h} \tilde{r}_{i,j}^{(\omega)}\right)$$

This ensures all scenario values are in the same units as the business inputs (€/ton, tons/month).

All $S$ scenarios are assigned equal probability: $p_\omega = 1/S$.

---

### 1.6 Scenario Reduction via K-Medoids

Running the optimization with $S = 3{,}000$ scenarios is computationally expensive. The scenario set is reduced to $N_c \ll S$ representative scenarios while preserving the probability distribution.

The reduction uses **K-medoids clustering** (Partitioning Around Medoids, PAM), which identifies $N_c$ actual scenarios from the full set as cluster centers (medoids), unlike K-means which computes artificial centroids.

**Distance metric**: Each scenario $\omega$ is represented by its $H \times k$ matrix of values. The pairwise distance is the Frobenius norm of the difference between scenario matrices:

$$d(\omega, \omega') = \left\|\mathbf{S}^{(\omega)} - \mathbf{S}^{(\omega')}\right\|_F = \sqrt{\sum_{h=1}^{H}\sum_{i=1}^{k}\left(s_{i,h}^{(\omega)} - s_{i,h}^{(\omega')}\right)^2}$$

**Probability reassignment**: After clustering, the probability of each retained medoid $m$ equals the sum of probabilities of all scenarios assigned to its cluster $\mathcal{C}_m$:

$$\hat{p}_m = \sum_{\omega \in \mathcal{C}_m} p_\omega$$

This preserves the first moment of the distribution: $\sum_m \hat{p}_m = 1$.

---

### 1.7 Stress Scenario Preservation

Before clustering, a subset of $n_s = \lceil N_c \cdot \text{pct} \rceil$ stress scenarios are identified and guaranteed inclusion in the reduced set. This prevents K-medoids from discarding rare but consequential tail events.

**Variable-tail selection**: For each variable $i$ with direction $d_i \in \{\text{lower}, \text{upper}, \text{both}\}$, the mean scenario value across the entire horizon is computed:

$$\bar{s}_i^{(\omega)} = \frac{1}{H} \sum_{h=1}^{H} s_{i,h}^{(\omega)}$$

Then the $n_s / n_{\text{tails}}$ most extreme scenarios per tail are selected and reserved. The remainder of the budget $N_c - n_s$ scenarios are allocated to K-medoids on the non-stress scenarios.

**Composite selection**: A scalar score is computed as a weighted sum of scenario mean values:

$$g^{(\omega)} = \sum_i w_i \cdot \bar{s}_i^{(\omega)}$$

The most extreme scenarios by this score (lower tail, upper tail, or both) are reserved.

---

### 1.8 Markov-Switching VAR (Regime-Switching Generator)

The `RegimeSwitchingGenerator` extends the plain VAR by allowing the joint return dynamics to shift between $R$ distinct market regimes — capturing the empirically observed alternation between calm and volatile periods in commodity markets.

#### Hidden Markov Chain

Let $s_t \in \{0, 1, \ldots, R-1\}$ denote the latent regime at time $t$, governed by a first-order Markov chain with transition matrix $\mathbf{P} \in \mathbb{R}^{R \times R}$:

$$P_{ij} = \mathbb{P}(s_t = j \mid s_{t-1} = i) \geq 0, \qquad \sum_{j=0}^{R-1} P_{ij} = 1 \quad \forall\, i$$

The default specification uses $R = 2$: regime 0 (normal / low-volatility) and regime 1 (stress / high-volatility).

#### Steady-State Distribution

The unconditional probability of each regime is the stationary distribution $\boldsymbol{\pi}$, satisfying $\boldsymbol{\pi}^\top \mathbf{P} = \boldsymbol{\pi}^\top$ with $\sum_j \pi_j = 1$:

$$(\mathbf{P}^\top - \mathbf{I})\,\boldsymbol{\pi} = \mathbf{0}, \qquad \mathbf{1}^\top \boldsymbol{\pi} = 1$$

The expected duration of regime $i$ before switching is $(1 - P_{ii})^{-1}$ months.

#### Implementation Strategy

A direct multivariate Markov-Switching VAR is not available in standard statistical libraries. The implementation uses a computationally stable three-step approach:

**Step 1 — Full-sample VAR for shared dynamics.** A single VAR($p$) is fitted on the full training sample (all regimes pooled):

$$\mathbf{r}_t = \hat{\boldsymbol{\mu}} + \sum_{\ell=1}^{p} \hat{\mathbf{A}}_\ell\, \mathbf{r}_{t-\ell} + \mathbf{e}_t$$

This yields stable, well-estimated AR coefficients and a full-sample residual series $\{\mathbf{e}_t\}$.

**Step 2 — Regime identification via MS-AR on PC1.** The first principal component of the VAR residuals captures the dominant co-movement:

$$\text{PC1}_t = \mathbf{v}_1^\top \mathbf{e}_t$$

where $\mathbf{v}_1$ is the leading eigenvector of $\text{Cov}(\mathbf{e})$. A univariate Markov-Switching AR is fitted on $\{\text{PC1}_t\}$ via the EM algorithm (Hamilton filter), yielding smoothed regime probabilities and the most-likely regime path $\{\hat{s}_t\}$. Regimes are labelled by ascending PC1 volatility: regime 0 = lowest volatility (normal), regime 1 = highest volatility (stress).

**Step 3 — Regime-conditional mean shifts and covariances.** Using the labels $\{\hat{s}_t\}$ from Step 2, regime-specific statistics are estimated from the full-sample VAR residuals:

$$\hat{\boldsymbol{\mu}}_r = \frac{1}{n_r}\sum_{t:\,\hat{s}_t = r} \mathbf{e}_t, \qquad \hat{\boldsymbol{\Sigma}}_r = \frac{1}{n_r - 1}\sum_{t:\,\hat{s}_t = r} (\mathbf{e}_t - \hat{\boldsymbol{\mu}}_r)(\mathbf{e}_t - \hat{\boldsymbol{\mu}}_r)^\top$$

To guard against instability when a regime has few observations, shrinkage is applied toward the full-sample covariance $\hat{\boldsymbol{\Sigma}}$:

$$\tilde{\boldsymbol{\Sigma}}_r = w_r\,\hat{\boldsymbol{\Sigma}}_r + (1 - w_r)\,\hat{\boldsymbol{\Sigma}}, \qquad w_r = \frac{n_r}{n_r + \kappa}$$

where $\kappa = 10$ is the regularization strength (hardcoded in the implementation).

#### Regime-Switching Forward Simulation

For each scenario $\omega$, a regime path is drawn from the Markov chain and regime-specific dynamics are applied at each step:

1. **Initialize**: draw $s_1^{(\omega)} \sim \boldsymbol{\pi}$ (steady-state distribution).
2. **Transition**: for $h \geq 2$, draw $s_h^{(\omega)} \sim \mathbf{P}[s_{h-1}^{(\omega)},\, :]$.
3. **Return generation**: apply shared AR dynamics plus a regime-specific mean shift and correlated shock:

$$\tilde{\mathbf{r}}_h^{(\omega)} = \hat{\boldsymbol{\mu}} + \sum_{\ell=1}^{p} \hat{\mathbf{A}}_\ell\,\tilde{\mathbf{r}}_{h-\ell}^{(\omega)} + \hat{\boldsymbol{\mu}}_{s_h^{(\omega)}} + \mathbf{L}_{s_h^{(\omega)}}\,\boldsymbol{\eta}_h^{(\omega)}$$

where $\mathbf{L}_{s_h} = \text{chol}(\tilde{\boldsymbol{\Sigma}}_{s_h})$ is the Cholesky factor of the regime covariance and $\boldsymbol{\eta}_h^{(\omega)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{D}(\mathbf{0}, \mathbf{I})$.

Level reconstruction follows Section 1.5. The approach produces scenarios with fat-tailed joint distributions, realistic crash episodes, and empirically calibrated regime persistence — without requiring a fully parameterised multivariate MS-VAR.

---

## Part 2: Two-Stage Stochastic Program

### 2.1 Sets, Indices, and Parameters

**Sets:**
- $T = \{1, \ldots, \mathcal{T}\}$ — monthly planning periods
- $\Omega$ — set of retained scenarios, $|\Omega| = N_c$

**Scenario probabilities:**
- $p_\omega > 0$ for all $\omega \in \Omega$, with $\sum_\omega p_\omega = 1$

**Scenario-dependent inputs** (known *per scenario* in stage 2):
- $D(t, \omega)$ — demand in tons of finished goods (Tn FG)
- $c_{\text{spot}}(t, \omega)$ — spot scrap cost (€/Tn RM)
- $p_{\text{sell}}(t, \omega)$ — steel selling price (€/Tn FG)

**Deterministic inputs** (known at stage 1):
- $c_{\text{fix}}(t)$ — fixed contract scrap cost (€/Tn RM)
- $\bar{x}_{\text{fix}}(t)$ — maximum fixed contract volume (Tn RM)
- $\bar{x}_{\text{opt}}(t)$ — maximum framework reservation (Tn RM)
- $f_{\text{opt}}(t)$ — framework reservation fee (€/Tn reserved)
- $\text{basis}_{\text{opt}}(t)$, $\text{floor}_{\text{opt}}(t)$, $\text{cap}_{\text{opt}}(t)$ — framework pricing bounds
- $\text{Cap}_{\text{base}}(t)$ — base production capacity (Tn FG)
- $\text{Cap}_{\text{flex}}(t)$ — flexible production capacity (Tn FG)
- $c_{\text{base}}(t)$ — base production cost (€/Tn FG)
- $c_{\text{flex}}(t)$ — flex production cost (€/Tn FG), $c_{\text{flex}} \geq c_{\text{base}}$
- $\alpha$ — RM yield: tons of scrap per ton of steel (Tn RM / Tn FG), $\alpha > 1$
- $h_{\text{rm}}(t)$ — raw material holding cost (€/Tn RM per period)
- $h_{\text{fg}}(t)$ — finished goods holding cost (€/Tn FG per period)
- $I_{\text{rm}}^0$ — initial RM inventory (Tn RM)
- $I_{\text{fg}}^0$ — initial FG inventory (Tn FG)
- $\pi(t)$ — unmet demand penalty (€/Tn FG)

---

### 2.2 Framework Exercise Price

The per-scenario exercise price for the framework contract is determined by a bounded pricing rule that prevents the call-off from becoming worse than spot in extreme markets:

$$c_{\text{opt}}(t, \omega) = \min\!\Big(\max\!\big(c_{\text{spot}}(t,\omega) + \text{basis}_{\text{opt}}(t),\ \text{floor}_{\text{opt}}(t)\big),\ \text{cap}_{\text{opt}}(t)\Big)$$

Setting $\text{floor}_{\text{opt}} = 0$ and $\text{cap}_{\text{opt}} = +\infty$ reduces to pure spot-plus-basis pricing.

---

### 2.3 Decision Variables

**Stage 1 (here-and-now) — decided before uncertainty resolves:**

| Variable | Domain | Description |
|----------|--------|-------------|
| $x_{\text{fix}}(t)$ | $\mathbb{R}_{\geq 0}$ | Fixed RM procurement (Tn RM) |
| $x_{\text{opt}}(t)$ | $\mathbb{R}_{\geq 0}$ | Framework reservation volume (Tn RM) |
| $P_{\text{base}}(t)$ | $\mathbb{R}_{\geq 0}$ | Base production plan (Tn FG) |

**Stage 2 (recourse) — decided after scenario $\omega$ is revealed:**

| Variable | Domain | Description |
|----------|--------|-------------|
| $y_{\text{opt}}(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | Framework call-off exercised (Tn RM) |
| $x_{\text{spot}}(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | Spot RM purchases (Tn RM) |
| $P_{\text{flex}}(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | Flex production activated (Tn FG) |
| $I_{\text{rm}}(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | RM inventory at end of period (Tn RM) |
| $I_{\text{fg}}(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | FG inventory at end of period (Tn FG) |
| $S(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | Sales served (Tn FG) |
| $U(t, \omega)$ | $\mathbb{R}_{\geq 0}$ | Unmet demand (Tn FG) |

Define total production as the derived expression:

$$P(t, \omega) = P_{\text{base}}(t) + P_{\text{flex}}(t, \omega)$$

---

### 2.4 Objective Function

The primary objective is to **maximize expected profit** across all scenarios:

$$\begin{aligned}
\max \quad & \sum_{\omega \in \Omega} p_\omega \sum_{t \in T} \Bigg[ \\
  & \quad p_{\text{sell}}(t,\omega)\, S(t,\omega) \\
  & - c_{\text{fix}}(t)\, x_{\text{fix}}(t) - f_{\text{opt}}(t)\, x_{\text{opt}}(t) \\
  & - c_{\text{opt}}(t,\omega)\, y_{\text{opt}}(t,\omega) - c_{\text{spot}}(t,\omega)\, x_{\text{spot}}(t,\omega) \\
  & - c_{\text{base}}(t)\, P_{\text{base}}(t) - c_{\text{flex}}(t)\, P_{\text{flex}}(t,\omega) \\
  & - h_{\text{rm}}(t)\, I_{\text{rm}}(t,\omega) - h_{\text{fg}}(t)\, I_{\text{fg}}(t,\omega) - \pi(t)\, U(t,\omega) \Bigg]
\end{aligned}$$

Let $Z(\omega) = \sum_{t} [\cdots]_\omega$ denote the realized profit in scenario $\omega$. Then the objective is $\max \mathbb{E}[Z] = \sum_\omega p_\omega Z(\omega)$.

---

### 2.5 Constraints

#### (1) Contract bounds — Stage 1

$$0 \leq x_{\text{fix}}(t) \leq \bar{x}_{\text{fix}}(t) \qquad \forall\, t \in T$$

$$0 \leq x_{\text{opt}}(t) \leq \bar{x}_{\text{opt}}(t) \qquad \forall\, t \in T$$

#### (2) Framework exercise rule — you cannot exercise what you have not reserved

$$0 \leq y_{\text{opt}}(t, \omega) \leq x_{\text{opt}}(t) \qquad \forall\, t \in T,\ \omega \in \Omega$$

#### (3) Capacity constraints

$$0 \leq P_{\text{base}}(t) \leq \text{Cap}_{\text{base}}(t) \qquad \forall\, t \in T$$

$$0 \leq P_{\text{flex}}(t, \omega) \leq \text{Cap}_{\text{flex}}(t) \qquad \forall\, t \in T,\ \omega \in \Omega$$

#### (4) Raw material inventory balance

For $t = 1$ (using initial inventory $I_{\text{rm}}^0$):

$$I_{\text{rm}}(1, \omega) = I_{\text{rm}}^0 + x_{\text{fix}}(1) + y_{\text{opt}}(1,\omega) + x_{\text{spot}}(1,\omega) - \alpha\, P(1,\omega)$$

For $t \geq 2$:

$$I_{\text{rm}}(t, \omega) = I_{\text{rm}}(t-1,\omega) + x_{\text{fix}}(t) + y_{\text{opt}}(t,\omega) + x_{\text{spot}}(t,\omega) - \alpha\, P(t,\omega)$$


$$I_{\text{rm}}(t, \omega) \geq 0 \qquad \forall\, t,\, \omega$$

#### (5) Finished goods inventory balance

For $t = 1$:

$$I_{\text{fg}}(1,\omega) = I_{\text{fg}}^0 + P(1,\omega) - S(1,\omega)$$

For $t \geq 2$:

$$I_{\text{fg}}(t,\omega) = I_{\text{fg}}(t-1,\omega) + P(t,\omega) - S(t,\omega)$$

$$I_{\text{fg}}(t,\omega) \geq 0 \qquad \forall\, t,\, \omega$$

#### (6) Demand accounting

$$S(t,\omega) + U(t,\omega) = D(t,\omega) \qquad \forall\, t \in T,\ \omega \in \Omega$$

$$S(t,\omega),\, U(t,\omega) \geq 0$$

---

### 2.6 Risk-Adjusted Objective: CVaR

For risk-averse decision makers, the objective is modified to include **Conditional Value-at-Risk (CVaR)** of profits.

**Value-at-Risk** at level $\alpha$ is the profit threshold below which the worst $(1-\alpha) \times 100\%$ of scenarios fall:

$$\text{VaR}_\alpha = \inf\!\left\{ z \in \mathbb{R} : \mathbb{P}[Z \leq z] \geq \alpha \right\}$$

Note: we work with *profit* ($Z$), not loss, so a *low* profit is the bad outcome. The leftmost $\alpha$-tail of the profit distribution represents the worst cases.

**CVaR** (also called Expected Shortfall) is the expected profit conditional on being in the worst $(1 - \alpha)$ fraction:

$$\text{CVaR}_\alpha = \mathbb{E}\!\left[\, Z \mid Z \leq \text{VaR}_\alpha \,\right]$$

Equivalently, using the Rockafellar-Uryasev representation (which is linear and suitable for optimization):

$$\text{CVaR}_\alpha = \text{VaR}_\alpha - \frac{1}{1-\alpha}\,\mathbb{E}\!\left[\max\!\left(0,\, \text{VaR}_\alpha - Z\right)\right]$$

In the discrete scenario setting this becomes:

$$\text{CVaR}_\alpha = \nu - \frac{1}{1-\alpha} \sum_{\omega \in \Omega} p_\omega\, z_\omega$$

where the auxiliary variables $\nu \in \mathbb{R}$ and $z_\omega \geq 0$ satisfy:

$$z_\omega \geq \nu - Z(\omega), \quad z_\omega \geq 0 \qquad \forall\, \omega \in \Omega$$

and $\nu$ is optimized jointly with the decision variables (it converges to $\text{VaR}_\alpha$ at optimality).

**Risk-adjusted objective**: The model mixes expected profit (upside focus) with CVaR (downside protection) via the risk aversion parameter $\lambda \in [0, 1]$:

$$\max \quad (1 - \lambda)\,\mathbb{E}[Z] + \lambda\,\text{CVaR}_\alpha(Z)$$

- $\lambda = 0$: risk-neutral (pure expected profit maximization)
- $\lambda = 1$: fully risk-averse (maximize worst-case expected profit)
- $\lambda = 0.3$, $\alpha = 0.05$: typical moderate parameterization — gives 70% weight to expected profit and 30% weight to the expected profit in the worst 5% of scenarios

---

### 2.7 Model Size and Solver

For a planning horizon of $\mathcal{T} = 12$ months and $N_c = 300$ scenarios the LP has approximately:

| Component | Count |
|-----------|-------|
| Stage-1 continuous variables | $3\mathcal{T} \approx 36$ |
| Stage-2 continuous variables | $7\mathcal{T} N_c \approx 25{,}200$ |
| Constraints | $\approx 8\mathcal{T} N_c + 3\mathcal{T} \approx 28{,}836$ |

This is a medium-scale linear program, solvable in seconds with HiGHS (default), CPLEX, or Gurobi.

---

### 2.8 Backtesting Protocol

The backtest operates as a **rolling-replan simulation**. Let $W$ denote the set of replan dates, spaced $E$ months apart (the execution window). For each replan date $t_0 \in W$:

1. **Fit window**: Fit the scenario model (VAR or Markov-Switching VAR) on the trailing $N$ months of data up to $t_0$ — no look-ahead.
2. **Generate and reduce**: Simulate $S$ forward paths of length $H$ months; reduce to $N_c$ representative scenarios via K-medoids (Section 1.6).
3. **Optimize**: Solve the two-stage stochastic program (Section 2.3–2.6) with initial inventories $(I_\text{rm}^0, I_\text{fg}^0)$ carried forward from the previous window.
4. **Execute**: Apply the first-stage decisions against the realized out-of-sample data for months $t_0 + 1$ to $t_0 + E$.
5. **Carry forward state**: Update the initial conditions for the next window using the realized end-of-window inventory levels:

$$I_\text{rm}^{0,\,\text{next}} = I_\text{rm}(E,\, \omega_\text{realized}), \qquad I_\text{fg}^{0,\,\text{next}} = I_\text{fg}(E,\, \omega_\text{realized})$$

The price anchor $\mathbf{p}_{t_0}$ (used in `convert_to_real_prices`) is also updated from the realized prices at the execution boundary, so contract pricing ratios remain calibrated to current market levels.

6. **Benchmark**: The safety stock model runs the identical process in parallel — same training window, same scenarios, same initial conditions — ensuring a like-for-like comparison.

The **Value of Stochastic Solution** accumulated over all windows is:

$$\text{VSS}_\text{total} = \sum_{t_0 \in W} \left[\sum_{t=t_0+1}^{t_0+E} \Pi_\text{stoch}(t) - \sum_{t=t_0+1}^{t_0+E} \Pi_\text{benchmark}(t)\right]$$

where $\Pi(t)$ is the realized profit in month $t$ under the respective plan. The rolling carry-forward ensures that inventory imbalances from one window propagate realistically into the next, rather than resetting to fixed initial conditions at each replan date.

---

## Part 3: Safety Stock Benchmark

The `SafetyStockModel` is not an optimizer — it is a **deterministic rule-based planner** that mimics traditional ERP/MRP logic. Its purpose is to provide a rigorous baseline so that the value of stochastic optimization (VSS) can be measured objectively.

The benchmark operates in two phases. First it computes first-stage decisions from point forecasts and statistical rules (no scenario awareness). Then it evaluates those fixed decisions across the full scenario set using the same recourse simulation as the stochastic model.

---

### 3.1 Point Forecasts

The benchmark uses **probability-weighted means** across all scenarios as its demand and price forecast for each period $t$:

$$\hat{D}(t) = \sum_{\omega \in \Omega} p_\omega\, D(t,\omega)$$

$$\hat{c}_{\text{spot}}(t) = \sum_{\omega \in \Omega} p_\omega\, c_{\text{spot}}(t,\omega)$$

$$\hat{p}_{\text{sell}}(t) = \sum_{\omega \in \Omega} p_\omega\, p_{\text{sell}}(t,\omega)$$

The overall demand standard deviation across all periods and scenarios:

$$\hat{\sigma}_D = \sqrt{\frac{1}{|\Omega||\mathcal{T}|-1} \sum_\omega \sum_t \left(D(t,\omega) - \bar{D}\right)^2}$$

is used to set safety stock levels.

---

### 3.2 Safety Stock Calculation

**Service-level method** (default): Safety stock is sized so that the probability of a stockout within one review period is at most $1 - \text{SL}$:

$$SS_{FG} = z_{\text{SL}}\, \hat{\sigma}_D\, \sqrt{L}$$

where $z_{\text{SL}} = \Phi^{-1}(\text{SL})$ is the standard normal quantile (e.g. $z_{0.95} \approx 1.645$) and $L$ is the review period in months.

**Fixed-periods method** (when `safety_stock_periods` $= k$ is set): Simply hold $k$ periods of average demand as buffer:

$$SS_{FG} = k \cdot \bar{D}$$

Raw material safety stock mirrors the FG buffer through the yield factor:

$$SS_{RM} = \alpha \cdot SS_{FG}$$

---

### 3.3 First-Stage Decision Rules

The benchmark constructs stage-1 commitments using **static allocation percentages** rather than optimizing over the scenario distribution.

**Fixed contract volume** — cover a fixed fraction of the expected RM requirement:

$$x_{\text{fix}}(t) = \min\!\Big(\phi_{\text{fix}} \cdot \alpha\, \hat{D}(t),\ \bar{x}_{\text{fix}}(t)\Big)$$

where $\phi_{\text{fix}}$ (`fixed_pct`, default 0.60) is the fraction of expected RM need locked in.

**Framework reservation** — hedge proportion of demand variability:

$$x_{\text{opt}}(t) = \min\!\Big(\phi_{\text{opt}} \cdot \alpha\, \hat{\sigma}_D(t),\ \bar{x}_{\text{opt}}(t)\Big)$$

where $\phi_{\text{opt}}$ (`framework_pct`, default 0.25) controls how much demand variance is covered by framework options.

**Base production plan** (level-load mode):

$$P_{\text{base}}(t) = \min\!\left(\bar{D} + \frac{SS_{FG}}{\mathcal{T}},\ \text{Cap}_{\text{base}}(t)\right)$$

In chase mode $\bar{D}$ is replaced by the period-specific forecast $\hat{D}(t)$.

---

### 3.4 Recourse Simulation

With stage-1 decisions fixed, the benchmark evaluates what happens in each scenario $\omega$ using a **greedy sequential simulation** (not an optimization):

**Framework exercise**: Exercise the full reservation if and only if the call-off price is at or below spot; otherwise let it lapse:

$$y_{\text{opt}}(t,\omega) = \begin{cases} x_{\text{opt}}(t) & \text{if } c_{\text{opt}}(t,\omega) \leq c_{\text{spot}}(t,\omega) \\ 0 & \text{otherwise} \end{cases}$$

**Spot purchases**: Cover any RM shortfall after fixed + framework inflows, plus a gradual safety-stock replenishment:

$$x_{\text{spot}}(t,\omega) = \max\!\Big(0,\, \alpha\,P(t,\omega) - I_{\text{rm}}(t-1,\omega) - x_{\text{fix}}(t) - y_{\text{opt}}(t,\omega)\Big) + 0.5\cdot\max\!\Big(0,\, SS_{RM} - I_{\text{rm}}^{\text{residual}}\Big)$$

**Flexible production**: Activated to close demand gaps and partially rebuild FG buffer:

$$P_{\text{flex}}(t,\omega) = \min\!\Big(\max\!\big(0,\, D(t,\omega) - I_{\text{fg}}(t-1,\omega) - P_{\text{base}}(t)\big) + 0.3\cdot\delta_{FG}(t,\omega),\ \text{Cap}_{\text{flex}}(t)\Big)$$

where $\delta_{FG}(t,\omega) = \max(0,\, SS_{FG} - I_{\text{fg}}^{\text{residual}})$ is the FG buffer shortfall.

**Sales and unmet demand** follow the same accounting as the stochastic model:

$$S(t,\omega) = \min\!\big(D(t,\omega),\, I_{\text{fg}}(t-1,\omega) + P_{\text{base}}(t) + P_{\text{flex}}(t,\omega)\big)$$

$$U(t,\omega) = D(t,\omega) - S(t,\omega)$$

The profit in each scenario $\omega$ is then computed using the same cost function as Section 2.4, enabling a like-for-like comparison.

---

### 3.5 Value of Stochastic Solution

Let $Z^*(\omega)$ denote the profit of the stochastic optimizer in scenario $\omega$ and $Z^{SS}(\omega)$ the profit of the safety stock benchmark in the same scenario. The VSS quantifies the benefit of replacing rule-based planning with explicit scenario optimization:

$$\text{VSS} = \sum_{\omega \in \Omega} p_\omega Z^*(\omega) - \sum_{\omega \in \Omega} p_\omega Z^{SS}(\omega) = \mathbb{E}[Z^*] - \mathbb{E}[Z^{SS}]$$

Because both models are evaluated on the **same scenario set**, the VSS isolates the pure informational value of scenario awareness; differences in computational effort or data requirements do not affect the comparison.

A positive VSS indicates that distributional knowledge — knowing the joint probability of price, cost, and demand outcomes, rather than only their means — allows the planner to choose first-stage commitments that perform better in expectation. The VSS tends to be largest when:

- Demand and price uncertainty is high (wide distributions)
- Correlations between variables are strong (joint tails matter)
- The cost of over- or under-commitment is asymmetric (convex recourse cost)
