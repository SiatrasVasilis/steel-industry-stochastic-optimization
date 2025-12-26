# Steel Industry Stochastic Optimization

A collection of scenario-based stochastic programming models tailored for steel industry capacity planning and procurement optimization under market uncertainty. All models are backtested using real historical market data.

## 📋 Overview

This repository provides stochastic optimization frameworks for steel industry decision-making under uncertainty. The models leverage Vector Autoregression (VAR) for scenario generation and two-stage stochastic programming to balance upfront commitments with operational flexibility.

## 🎯 Business Problem

Steel manufacturers must make strategic capacity and procurement decisions months or years in advance, while facing:
- **Price volatility**: Uncertain steel selling prices and scrap material costs
- **Demand uncertainty**: Unpredictable customer demand patterns
- **Operational constraints**: Limited flexibility in capacity adjustment and material procurement

These models help optimize profit while managing downside risks through a balance of commitment and flexibility.

## 📚 Available Models

### 1. Two-Stage Capacity and Procurement Planning

**Status**: ✅ Implemented  
**Documentation**: [docs/two_stage_model.md](docs/two_stage_model.md)  
**Implementation**: [src/models/basic.py](src/models/basic.py)

A two-stage stochastic programming model for optimizing capacity expansion and raw material procurement decisions under uncertainty.

**Key Features**:
- **Stage 1 (Strategic)**: Base capacity planning and procurement contracts set upfront
- **Stage 2 (Operational)**: Flexible capacity and spot procurement as recourse actions
- **Scenario Generation**: VAR-based scenarios capturing correlations between prices, costs, and demand
- **Scenario Reduction**: K-medoids clustering with stress scenarios for computational efficiency
- **Backtesting**: Historical validation of optimization decisions against realized market outcomes

**Decision Variables**:
- Base capacity commitment (tons/month)
- Base scrap procurement contracts (call options)
- Flexible capacity usage (recourse)
- Spot market procurement (recourse)
- Production and sales volumes (recourse)

**Typical Planning Horizon**: 12-24 months

**Applications**:
- Annual capacity planning and budget allocation
- Raw material contract negotiations
- Risk assessment and stress testing
- Scenario analysis for strategic planning

---

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/steel-industry-stochastic-optimization.git
cd steel-industry-stochastic-optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Example

```python
from src.models.basic import TwoStageCapacityAndProcurementPlanning

# Load historical data from FRED
data = TwoStageCapacityAndProcurementPlanning.load_data_from_fredapi(
    api_key='your_api_key_here'
)

# Fit VAR model
log_ret = TwoStageCapacityAndProcurementPlanning.log_returns(data)
var_model = TwoStageCapacityAndProcurementPlanning.fit_VAR_model(Δlog=log_ret)

# Generate and reduce scenarios
scenarios = TwoStageCapacityAndProcurementPlanning.generate_future_returns_scenarios(
    var_model, simulation_start_date='2020-01-01', horizon=12, n_scenarios=1000
)
scenarios_reduced, prob_reduced = TwoStageCapacityAndProcurementPlanning.reduce_scenarios_kmedoids(
    scenarios, prob=scenarios['prob'], n_scenario_clusters=50
)

# Optimize capacity and procurement
decisions = TwoStageCapacityAndProcurementPlanning.optimize_capacity_and_procurement(
    scenarios=scenarios_reduced,
    prob=prob_reduced,
    alpha=1.5,
    c_var=200.0,
    c_cap_base=10.0,
    c_cap_flex=25.0,
    delta_base=5.0,
    delta_spot=15.0,
    pen_unmet=500.0,
    gamma_cap=0.3,
    gamma_scrap=0.8
)
```

For detailed documentation and advanced usage, see [docs/two_stage_model.md](docs/two_stage_model.md).

---

## 📖 Documentation

### Model Documentation
- **[Two-Stage Model](docs/two_stage_model.md)**: Complete documentation including problem formulation, mathematical model, VAR forecasting, scenario generation, and usage examples

### Key Sections
1. **Problem Formulation**: Business context and decision timeline
2. **Model Description**: VAR modeling, scenario generation, and stochastic program formulation
3. **Usage Guide**: Comprehensive method reference with examples

---

## 🗂️ Repository Structure

```
steel-industry-stochastic-optimization/
├── src/
│   ├── models/
│   │   └── basic.py              # Two-stage model implementation
│   └── utils/                     # Shared utilities (future)
├── data/                          # Input data (not tracked in git)
├── results/                       # Optimization results and plots
├── notebooks/                     # Jupyter notebooks for experiments
├── tests/                         # Unit tests (future)
├── docs/
│   └── two_stage_model.md        # Two-stage model documentation
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```

---

## 🔧 Requirements

- Python 3.8+
- Key dependencies:
  - `pyomo` - Optimization modeling
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `statsmodels` - VAR modeling
  - `scikit-learn` - Scenario reduction (K-medoids)
  - `fredapi` - Data retrieval from FRED
  - `matplotlib` - Visualization

See [requirements.txt](requirements.txt) for complete list.

---

## 📊 Data Sources

The models use publicly available economic data from the Federal Reserve Economic Data (FRED) database:

- **Steel Prices**: Producer Price Index for steel products (WPU101704)
- **Scrap Costs**: Producer Price Index for scrap metal (WPU1012)
- **Steel Demand**: Industrial production index for steel (IPG3311A2S)

**Note**: You need a free FRED API key. [Get one here](https://fred.stlouisfed.org/docs/api/api_key.html).

---

## 🔮 Future Models (Roadmap)

- **Multi-Stage Model**: Extended planning with more decision stages
- **Risk-Averse Formulations**: CVaR and robust optimization variants
- **Multi-Product Planning**: Models for diversified product portfolios
- **Inventory Management**: Models with finished goods storage
- **Network Optimization**: Multi-facility capacity planning

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

---

## 🎓 References

- Birge, J.R. & Louveaux, F. (2011). *Introduction to Stochastic Programming*. Springer.
- Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2009). *Lectures on Stochastic Programming*. SIAM.
- Kall, P. & Wallace, S.W. (1994). *Stochastic Programming*. Wiley.
