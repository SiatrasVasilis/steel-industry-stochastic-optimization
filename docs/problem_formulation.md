# Business Problem Formulation

## Context

A steel manufacturer (or a steel-product trading company) must plan procurement and production across a rolling horizon of roughly 12 months. During this window the business is exposed to three interacting sources of uncertainty that it cannot eliminate but must actively manage:

- **Steel selling price** — driven by global demand cycles, construction activity, automotive output, and import flows
- **Scrap metal cost** — the primary raw material input, highly correlated with steel prices but with its own supply-side dynamics (scrap availability, export bans, recycling rates)
- **Customer demand** — driven by downstream industry cycles, seasonal patterns, and macroeconomic conditions

These three variables move together in non-trivial ways. A construction boom raises both end-product prices and scrap demand; a recession compresses both revenue and input costs. A naive model that treats them independently will systematically mis-price the value of optionality.

---

## The Planning Cycle

Decisions are made on a **monthly cadence** with a **12-month forward horizon**. At the start of each planning cycle the company must decide:

1. **How much base production capacity to commit to** for each month ahead — this means paying for machinery, scheduled labor contracts, and shift planning. These costs are incurred regardless of how much is actually produced.

2. **How much raw material to lock in under fixed contracts** — these give certainty of supply at a known price but are obligatory; you pay whether you use them or not.

3. **How large a framework (call-off) reservation to secure** — these give the *right but not the obligation* to purchase material at a bounded price. They cost a small reservation fee upfront; the actual purchase decision is made later once prices and demand are clearer.

After committing to these first-stage choices, the horizon plays out month by month. As each month arrives, the company observes actual prices and demand and takes **recourse actions**:

- Exercise some or all of the framework reservation to buy raw material
- Buy additional material on the **spot market** (most expensive, unlimited but price-exposed)
- Activate **flexible capacity** (overtime, extra shifts) if demand exceeds base plan
- Draw on **raw material inventory** built up from earlier over-procurement
- Draw on **finished goods inventory** built up from earlier over-production
- Accept some **unmet demand** (with a penalty) if all options are exhausted

---

## Contract Types

### Fixed Procurement Contracts

A fixed contract commits the company to purchasing a specified volume of scrap at a pre-agreed price. They offer:

- **Price certainty**: the cost-per-ton is known when the contract is signed
- **Supply security**: volume is guaranteed regardless of spot market tightness
- **Constraint**: the volume is purchased regardless of how demand actually develops

Fixed contracts are suited to the firm "base load" of expected demand — the portion the company is confident it will need regardless of market conditions.

### Framework (Call-Off) Contracts

A framework contract gives the right to purchase up to a reserved volume at a **price bounded by a floor and a cap**. Structurally:

- At planning time, the company reserves a maximum volume and pays a small reservation fee per reserved ton
- During the horizon, the company exercises (calls off) as much of that reservation as it needs, at a price that tracks spot but is bounded: `exercise price = clip(spot + basis, floor, cap)`
- Unexercised volume simply lapses at no additional cost

Framework contracts are suited to the portion of procurement that may or may not be needed depending on how demand develops. They convert uncertain procurement cost from open-ended spot exposure into a bounded range.

### Spot Purchases

Spot purchases are made at the prevailing market price at the time of purchase. They provide unlimited flexibility but carry the highest price risk and volatility. In the model they serve as the *instrument of last resort* — used only when fixed contracts and framework call-offs are insufficient.

---

## Production Structure

### Base Production (Committed Capacity)

Base production represents standard scheduled operations: regular shifts, contracted labor, committed machinery use. The company sets a **base capacity level before uncertainty resolves**. This level:

- Incurs a fixed cost per ton of capacity regardless of utilization
- Can produce up to the declared capacity level in any given period
- Cannot be increased after commitment without activating flexible capacity

### Flexible Production (Recourse Capacity)

Flexible production represents temporary capacity extensions: overtime shifts, temporary labor, short-term subcontracting. It is activated **after demand is observed**, making it a recourse decision. Flexible production:

- Is more expensive per ton than base production (premium for last-minute availability)
- Is bounded by a maximum flex capacity (a fraction of base)
- Allows the company to serve demand spikes without pre-committing to permanent capacity

---

## Inventory Buffers

### Raw Material (RM) Inventory

Scrap metal can be purchased in advance and stored. RM inventory:

- Absorbs timing mismatches between procurement and production
- Allows the company to take advantage of low spot prices by buying ahead
- Incurs a holding cost per ton per month (storage, financing, handling)
- Decouples procurement decisions from production timing

A **yield conversion factor** (α > 1) converts RM inventory consumed to finished goods produced: producing 1 ton of steel requires α tons of scrap.

### Finished Goods (FG) Inventory

Produced steel can be held in buffer stock before delivery. FG inventory:

- Decouples production timing from demand timing
- Allows level-load production when demand is seasonal or lumpy
- Incurs a holding cost per ton per month
- Enables the model to serve short-term demand spikes from stock rather than activating expensive flex capacity

---

## Demand, Sales, and Unmet Demand

Customer demand each month is uncertain at planning time. Once demand is realized:

- **Sales served**: the portion of demand actually fulfilled from production and inventory
- **Unmet demand**: the shortfall when total supply is insufficient

Unmet demand triggers a penalty cost representing lost margin, customer goodwill damage, and emergency expediting costs. The penalty is high enough that the optimizer treats it as a last resort, but finite — reflecting the business reality that occasionally missing a delivery is preferable to carrying permanently excessive capacity.

---

## Scenario Generation

The uncertainty in steel price, scrap cost, and demand is not modelled with simple point forecasts or independent distributions. Instead, the three variables are jointly simulated using a **Markov-Switching VAR** model fitted on historical monthly data from FRED.

The model identifies two latent market regimes — a normal regime characterised by moderate volatility and mean-reverting dynamics, and a stress regime with elevated volatility and larger cross-variable shocks. A Markov chain governs transitions between regimes, calibrated from historical data. Forward simulation draws regime paths from this chain and applies regime-specific dynamics at each step, naturally producing scenarios that reflect both calm periods and crisis episodes, with realistic cross-variable correlations throughout.

Three thousand correlated forward paths are generated for each planning horizon, then reduced to approximately 300 representative scenarios via K-medoids clustering (preserving tail/stress scenarios explicitly). This compact set is what enters the optimizer.

---

## Why Two-Stage?

The two-stage structure matches the actual information structure of the planning problem:

**Before uncertainty resolves** (Stage 1), the company chooses:
- Base capacity commitment → determines maximum "normal" production across the entire horizon
- Fixed procurement volumes → locked at known prices, binding
- Framework reservation → option size secured, paying only a small fee

These choices must be made under uncertainty because they require lead times (capacity scheduling, procurement negotiations) that exceed the forecast horizon.

**After uncertainty resolves** (Stage 2), the company observes the actual scenario and chooses:
- Framework exercise → how much of the reserved volume to actually call off
- Spot purchases → how much additional material to buy at observed prices
- Flex production → whether to activate overtime to meet demand
- Inventory decisions → how much RM and FG to carry to the next period

This stage-2 flexibility is what makes the approach more valuable than deterministic planning: the optimizer learns to reserve *option value* — choosing stage-1 commitments that keep stage-2 adaption possible — rather than betting on a single forecast.

---

## The Benchmark: Safety Stock Planning

The traditional alternative to optimization is a rule-based approach typical of ERP/MRP systems:

1. **Forecast**: Take the mean (expected) demand and price for each period
2. **Safety stock**: Add a buffer based on demand variability and a target service level (e.g., 95th percentile)
3. **Allocation split**: Allocate procurement across channels using fixed percentages (e.g., 60% fixed, 25% framework, remainder spot)
4. **Production**: Follow a level-load or demand-chase heuristic

This approach ignores correlations between price and demand, treats uncertainty as additive noise rather than a joint distribution, and cannot adapt the commitment/flexibility split to market conditions. The **Value of Stochastic Solution (VSS)** quantifies how much expected profit the optimization framework gains relative to this benchmark.

---

## What Matters to the Business

The business cares about:

- **Expected profit**: revenue from sales minus all procurement, production, inventory, and penalty costs across the planning horizon
- **Downside risk**: the worst-case profit in adverse scenarios (demand collapse + cost spike)
- **Fill rate**: the fraction of customer demand that is actually served — a service metric
- **Procurement mix**: how much is locked (fixed contracts) vs. flexible (framework + spot)
- **Capacity utilization**: whether base capacity is being used efficiently

The optimization model provides all of these outputs, enabling management to understand not just **what to do** but **why** — which scenarios drive the recommendation and how sensitive the decision is to assumptions.

---

## Rolling-Replan Backtesting

In practice the model is validated through a **rolling-replan simulation** rather than a single in-sample evaluation:

1. At each replan date, the scenario model is re-fitted on the most recent 15 years of data (no look-ahead).
2. The optimizer plans 12 months ahead and commits first-stage decisions.
3. The first 6 months of the plan are executed against realized out-of-sample market data.
4. Inventory levels (raw material and finished goods) are carried forward to the next planning window.
5. Price anchors are updated from realized market prices before the next replan.

This process is run identically for the safety stock benchmark — same training data, same scenarios, same execution window — ensuring that the **Value of Stochastic Solution (VSS)** reflects only the value of distributional information, not any data advantage.

The backtest covers 30 rolling windows from December 2007 to June 2022, spanning the 2008 financial crisis, the COVID-19 demand shock, and the post-COVID supply spike — regimes that stress-test every dimension of the planning problem.
