import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import io
import contextlib

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models.basic import TwoStageCapacityAndProcurementPlanning

warnings.filterwarnings('ignore')

# Helper function to capture printed output
@contextlib.contextmanager
def capture_output():
    """Capture stdout for displaying in UI"""
    old_out = sys.stdout
    try:
        out = io.StringIO()
        sys.stdout = out
        yield out
    finally:
        sys.stdout = old_out

# Page configuration
st.set_page_config(
    page_title="Steel Industry Stochastic Optimization",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Steel Industry Stochastic Optimization")
st.markdown("### Two-Stage Capacity and Procurement Planning")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Step", [
    "1. Data Loading",
    "2. VAR Model",
    "3. Scenario Generation",
    "4. Scenario Reduction",
    "5. Optimization",
    "6. Analysis"
])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_subset' not in st.session_state:
    st.session_state.data_subset = None
if 'var_model' not in st.session_state:
    st.session_state.var_model = None
if 'scenario_returns' not in st.session_state:
    st.session_state.scenario_returns = None
if 'prob' not in st.session_state:
    st.session_state.prob = None
if 'scenario_levels' not in st.session_state:
    st.session_state.scenario_levels = None
if 'scenarios_reduced' not in st.session_state:
    st.session_state.scenarios_reduced = None
if 'prob_reduced' not in st.session_state:
    st.session_state.prob_reduced = None
if 'decisions' not in st.session_state:
    st.session_state.decisions = None

# ================================
# PAGE 1: DATA LOADING
# ================================
if page == "1. Data Loading":
    st.header("Step 1: Load Historical Data")
    
    with st.expander("ℹ️ About This Step", expanded=False):
        st.markdown("""
        Load historical steel industry data from FRED (Federal Reserve Economic Data):
        - **Steel Prices**: Producer price index for steel products
        - **Scrap Costs**: Price index for scrap metal
        - **Demand**: Industrial production index for steel
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Source Configuration")
        
        api_key = st.text_input(
            "FRED API Key", 
            value="2f32e3af8652ea8fbfebca8f7ec39be6",
            type="password",
            help="Get your free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        
        steel_price_id = st.text_input("Steel Price Identifier", value="WPU101704")
        scrap_price_id = st.text_input("Scrap Price Identifier", value="WPU1012")
        steel_demand_id = st.text_input("Steel Demand Identifier", value="IPG3311A2S")
        
        if st.button("📥 Load Data", type="primary"):
            with st.spinner("Loading data from FRED..."):
                try:
                    with capture_output() as output:
                        data = TwoStageCapacityAndProcurementPlanning.load_data_from_fredapi(
                            api_key=api_key,
                            steel_price_identifier=steel_price_id,
                            scrap_price_identifier=scrap_price_id,
                            steel_demand_identifier=steel_demand_id,
                            plot_data=False
                        )
                    st.session_state.data = data
                    st.success(f"✅ Data loaded successfully! {len(data)} observations from {data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')}")
                    
                    # Show detailed output
                    output_text = output.getvalue()
                    if output_text.strip():
                        with st.expander("📋 View Detailed Output"):
                            st.text(output_text)
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.subheader("Data Subsetting")
        
        if st.session_state.data is not None:
            n_obs = st.number_input("Number of observations", min_value=50, max_value=len(st.session_state.data), value=180, step=10)
            lag_order = st.number_input("VAR lag order (p)", min_value=1, max_value=12, value=2, step=1)
            last_obs = st.date_input("Last observation date", value=pd.to_datetime("2019-12-01"))
            
            if st.button("✂️ Create Subset"):
                with st.spinner("Creating data subset..."):
                    data_subset = TwoStageCapacityAndProcurementPlanning.get_n_observations(
                        st.session_state.data,
                        n=n_obs,
                        p=lag_order,
                        last_observation=last_obs.strftime('%Y-%m-%d'),
                        plot_data=False
                    )
                    st.session_state.data_subset = data_subset
                    st.success(f"✅ Subset created: {len(data_subset)} observations")
        else:
            st.info("👆 Load data first to create a subset")
    
    # Display loaded data
    if st.session_state.data is not None:
        st.subheader("Loaded Data Preview")
        
        tab1, tab2 = st.tabs(["📊 Chart", "📋 Table"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 6))
            st.session_state.data.plot(ax=ax)
            ax.set_title("Historical Steel Industry Data")
            ax.set_ylabel("Index (1982=100)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab2:
            st.dataframe(st.session_state.data.tail(20))
            
        if st.session_state.data_subset is not None:
            st.subheader("Data Subset Preview")
            st.dataframe(st.session_state.data_subset.describe())

# ================================
# PAGE 2: VAR MODEL
# ================================
elif page == "2. VAR Model":
    st.header("Step 2: VAR Model Estimation")
    
    if st.session_state.data_subset is None:
        st.warning("⚠️ Please load and subset data in Step 1 first")
    else:
        with st.expander("ℹ️ About VAR Models", expanded=False):
            st.markdown("""
            Vector Autoregression (VAR) models capture the dynamic relationships between:
            - Demand, Prices, and Costs
            - Used to generate realistic future scenarios
            - Lag order selection determines how many past periods influence the future
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Configuration")
            
            max_lags = st.number_input("Maximum lags to test", min_value=1, max_value=12, value=12, step=1)
            ic_method = st.selectbox("Information Criterion", ["bic", "aic", "hqic"], index=0)
            
            testing_options = st.multiselect(
                "Validation Tests",
                ["stability", "irf", "corr", "sim_stats", "residual_tests"],
                default=["corr", "stability"]
            )
            
            if st.button("🔧 Fit VAR Model", type="primary"):
                with st.spinner("Fitting VAR model..."):
                    try:
                        with capture_output() as output:
                            var_model = TwoStageCapacityAndProcurementPlanning.fit_VAR_model(
                                data=st.session_state.data_subset,
                                p=None,
                                maxlags=max_lags,
                                method=ic_method,
                                testing=testing_options,
                                print_warnings=False
                            )
                        st.session_state.var_model = var_model
                        st.success(f"✅ VAR model fitted successfully! Optimal lag order: {var_model.k_ar}")
                        
                        # Show detailed output
                        output_text = output.getvalue()
                        if output_text.strip():
                            with st.expander("📋 View Model Diagnostics"):
                                st.text(output_text)
                    except Exception as e:
                        st.error(f"❌ Error fitting VAR model: {str(e)}")
        
        with col2:
            if st.session_state.var_model is not None:
                st.subheader("Model Summary")
                
                var_model = st.session_state.var_model
                
                st.metric("Lag Order", var_model.k_ar)
                st.metric("AIC", f"{var_model.aic:.2f}")
                st.metric("BIC", f"{var_model.bic:.2f}")
                
                # Shock distribution analysis
                if st.button("📊 Analyze Shock Distributions"):
                    with st.spinner("Analyzing residuals..."):
                        shock_analysis = TwoStageCapacityAndProcurementPlanning.analyze_shock_distributions(
                            var_model,
                            plot_diagnostics=True
                        )
                        st.session_state.shock_analysis = shock_analysis
                        
                        st.write("**Recommended Distribution:**", shock_analysis['overall_distribution'])
                        st.write("**Parameters:**", shock_analysis['overall_params'])

# ================================
# PAGE 3: SCENARIO GENERATION
# ================================
elif page == "3. Scenario Generation":
    st.header("Step 3: Generate Future Scenarios")
    
    if st.session_state.var_model is None:
        st.warning("⚠️ Please fit VAR model in Step 2 first")
    else:
        with st.expander("ℹ️ About Scenario Generation", expanded=False):
            st.markdown("""
            Generate multiple possible future trajectories for:
            - Steel demand
            - Steel prices
            - Scrap costs
            
            These scenarios capture market uncertainty and are used in optimization.
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Scenario Parameters")
            
            horizon = st.number_input("Forecast Horizon (months)", min_value=6, max_value=48, value=24, step=6)
            n_scenarios = st.number_input("Number of Scenarios", min_value=100, max_value=10000, value=1000, step=100)
            
            shock_dist = st.selectbox(
                "Shock Distribution",
                ["normal", "t", "skewed_t", "laplace"],
                index=1
            )
            
            # Distribution parameters
            if shock_dist == "t":
                df = st.slider("Degrees of Freedom", min_value=3, max_value=10, value=7)
                dist_params = {'df': df}
            elif shock_dist == "skewed_t":
                df = st.slider("Degrees of Freedom", min_value=3, max_value=10, value=5)
                dist_params = {'df': df, 'skewness': [0, 0, 0]}
            else:
                dist_params = {}
            
            seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
            
            if st.button("🎲 Generate Scenarios", type="primary"):
                with st.spinner(f"Generating {n_scenarios} scenarios..."):
                    try:
                        with capture_output() as output:
                            scenario_returns, prob = TwoStageCapacityAndProcurementPlanning.generate_future_returns_scenarios(
                                var_model=st.session_state.var_model,
                                simulation_start_date=st.session_state.data_subset.index.max().strftime('%Y-%m'),
                                horizon=horizon,
                                n_scenarios=n_scenarios,
                                seed=seed,
                                shock_distribution=shock_dist,
                                distribution_params=dist_params
                            )
                        st.session_state.scenario_returns = scenario_returns
                        st.session_state.prob = prob
                        st.success(f"✅ Generated {n_scenarios} scenarios over {horizon} months")
                        
                        # Show detailed output
                        output_text = output.getvalue()
                        if output_text.strip():
                            with st.expander("📋 View Generation Details"):
                                st.text(output_text)
                    except Exception as e:
                        st.error(f"❌ Error generating scenarios: {str(e)}")
        
        with col2:
            if st.session_state.scenario_returns is not None:
                st.subheader("Convert to Levels")
                
                st.write("Convert log returns to actual price/quantity levels")
                
                use_real_prices = st.checkbox("Use Real Prices", value=True)
                
                if use_real_prices:
                    steel_price = st.number_input("Steel Price (€/ton)", value=800.0, step=10.0)
                    scrap_cost = st.number_input("Scrap Cost (€/ton)", value=400.0, step=10.0)
                    demand = st.number_input("Demand (tons/month)", value=50000.0, step=1000.0)
                    real_prices = {'P': steel_price, 'C': scrap_cost, 'D': demand}
                else:
                    real_prices = None
                
                if st.button("📈 Convert to Levels"):
                    with st.spinner("Converting returns to levels..."):
                        try:
                            scenario_levels, info = TwoStageCapacityAndProcurementPlanning.reconstruct_levels_from_returns(
                                scenario_returns=st.session_state.scenario_returns,
                                historical_data=st.session_state.data_subset,
                                anchor_date=None,
                                real_prices=real_prices
                            )
                            st.session_state.scenario_levels = scenario_levels
                            st.success("✅ Scenarios converted to levels")
                            
                            st.write("**Units:**", info['units'])
                        except Exception as e:
                            st.error(f"❌ Error converting: {str(e)}")

# ================================
# PAGE 4: SCENARIO REDUCTION
# ================================
elif page == "4. Scenario Reduction":
    st.header("Step 4: Reduce Scenario Count")
    
    if st.session_state.scenario_levels is None:
        st.warning("⚠️ Please generate scenarios in Step 3 first")
    else:
        with st.expander("ℹ️ About Scenario Reduction", expanded=False):
            st.markdown("""
            Reduce the number of scenarios using K-Medoids clustering:
            - Maintains statistical properties
            - Faster optimization
            - Can include stress scenarios for tail risk
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Reduction Parameters")
            
            n_clusters = st.number_input("Target Scenarios", min_value=20, max_value=200, value=50, step=10)
            stress_pct = st.slider("Stress Scenario %", min_value=0.0, max_value=0.2, value=0.01, step=0.01, format="%.2f")
            
            stress_dir = st.selectbox(
                "Stress Direction",
                ["both", "downside", "upside"],
                index=1
            )
            
            max_iter = st.number_input("Max Iterations", min_value=50, max_value=300, value=100, step=50)
            
            seed = st.number_input("Random Seed ", min_value=0, value=42, step=1, key="reduction_seed")
            
            if st.button("🔍 Reduce Scenarios", type="primary"):
                with st.status(f"Reducing to {n_clusters} scenarios...", expanded=True) as status:
                    try:
                        with capture_output() as output:
                            scenarios_red, prob_red = TwoStageCapacityAndProcurementPlanning.reduce_scenarios_kmedoids(
                                scenarios=st.session_state.scenario_levels,
                                prob=st.session_state.prob,
                                n_scenario_clusters=n_clusters,
                                stress_pct=stress_pct,
                                stress_direction=stress_dir,
                                seed=seed,
                                max_iter=max_iter
                            )
                        
                        st.session_state.scenarios_reduced = scenarios_red
                        st.session_state.prob_reduced = prob_red
                        
                        # Display detailed output in status
                        output_text = output.getvalue()
                        if output_text.strip():
                            st.text(output_text)
                        
                        status.update(label=f"✅ Reduced to {n_clusters} scenarios", state="complete")
                    except Exception as e:
                        status.update(label="❌ Reduction failed", state="error")
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.session_state.scenarios_reduced is not None:
                st.subheader("Reduction Summary")
                
                original_count = st.session_state.scenario_levels['Scenario'].nunique()
                reduced_count = st.session_state.scenarios_reduced['Scenario'].nunique()
                
                st.metric("Original Scenarios", original_count)
                st.metric("Reduced Scenarios", reduced_count)
                st.metric("Reduction Ratio", f"{reduced_count/original_count:.1%}")
                
                st.write("**Probability Distribution:**")
                st.write(f"- Min: {st.session_state.prob_reduced.min():.4f}")
                st.write(f"- Max: {st.session_state.prob_reduced.max():.4f}")
                st.write(f"- Mean: {st.session_state.prob_reduced.mean():.4f}")

# ================================
# PAGE 5: OPTIMIZATION
# ================================
elif page == "5. Optimization":
    st.header("Step 5: Solve Optimization Problem")
    
    if st.session_state.scenarios_reduced is None:
        st.warning("⚠️ Please reduce scenarios in Step 4 first")
    else:
        with st.expander("ℹ️ About Optimization", expanded=False):
            st.markdown("""
            Two-stage stochastic programming optimization:
            - **Stage 1**: Set base capacity and procurement contracts
            - **Stage 2**: Adapt production to realized scenarios
            - **Objective**: Maximize expected profit under uncertainty
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Parameters")
            
            st.write("**Technical Parameters:**")
            alpha = st.number_input("Scrap Consumption Rate (α)", min_value=1.0, max_value=2.0, value=1.01, step=0.01)
            
            st.write("**Cost Parameters (€/ton):**")
            c_var = st.number_input("Variable Production Cost", value=250.0, step=10.0)
            c_cap_base = st.number_input("Base Capacity Cost", value=10.0, step=1.0)
            c_cap_flex = st.number_input("Flexible Capacity Cost", value=30.0, step=1.0)
            delta_base = st.number_input("Base Procurement Premium", value=5.0, step=1.0)
            delta_spot = st.number_input("Spot Procurement Premium", value=15.0, step=1.0)
            pen_unmet = st.number_input("Unmet Demand Penalty", value=0.0, step=10.0)
            
        with col2:
            st.write("**Flexibility Parameters:**")
            gamma_cap = st.slider("Max Flexible Capacity (% of base)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            gamma_scrap = st.slider("Max Spot Procurement (% of base)", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
            
            solver = st.selectbox("Solver", ["highs", "glpk", "gurobi", "cplex"], index=0)
            
            if st.button("⚙️ Solve Optimization", type="primary"):
                with st.status("Solving stochastic programming problem...", expanded=True) as status:
                    try:
                        with capture_output() as output:
                            decisions = TwoStageCapacityAndProcurementPlanning.optimize_capacity_and_procurement(
                                scenarios=st.session_state.scenarios_reduced,
                                prob=st.session_state.prob_reduced,
                                alpha=alpha,
                                c_var=c_var,
                                c_cap_base=c_cap_base,
                                c_cap_flex=c_cap_flex,
                                delta_base=delta_base,
                                delta_spot=delta_spot,
                                pen_unmet=pen_unmet,
                                gamma_cap=gamma_cap,
                                gamma_scrap=gamma_scrap,
                                solver=solver
                            )
                        
                        st.session_state.decisions = decisions
                        st.session_state.opt_params = {
                            'alpha': alpha, 'c_var': c_var, 'c_cap_base': c_cap_base,
                            'c_cap_flex': c_cap_flex, 'delta_base': delta_base,
                            'delta_spot': delta_spot, 'pen_unmet': pen_unmet,
                            'gamma_cap': gamma_cap, 'gamma_scrap': gamma_scrap
                        }
                        
                        # Display solver output
                        output_text = output.getvalue()
                        if output_text.strip():
                            st.text(output_text)
                        
                        status.update(label="✅ Optimization completed successfully!", state="complete")
                    except Exception as e:
                        status.update(label="❌ Optimization failed", state="error")
                        st.error(f"Error: {str(e)}")
        
        # Display results
        if st.session_state.decisions is not None:
            st.subheader("Optimal Decisions")
            
            tab1, tab2 = st.tabs(["📊 Charts", "📋 Table"])
            
            with tab1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                st.session_state.decisions['Cap_base'].plot(ax=ax1, marker='o')
                ax1.set_title("Optimal Base Capacity")
                ax1.set_ylabel("Capacity (tons/month)")
                ax1.grid(True, alpha=0.3)
                
                st.session_state.decisions['Q_base'].plot(ax=ax2, marker='o', color='orange')
                ax2.set_title("Optimal Base Procurement Contract")
                ax2.set_ylabel("Contract Quantity (tons/month)")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                st.dataframe(st.session_state.decisions)

# ================================
# PAGE 6: ANALYSIS
# ================================
elif page == "6. Analysis":
    st.header("Step 6: Results Analysis")
    
    if st.session_state.decisions is None:
        st.warning("⚠️ Please solve optimization in Step 5 first")
    else:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Profit Distribution", "Backtesting"]
        )
        
        if analysis_type == "Profit Distribution":
            st.subheader("Profit Distribution Over Time")
            
            conf_lower = st.slider("Confidence Interval Lower Bound", 0.0, 0.5, 0.01, 0.01)
            conf_upper = st.slider("Confidence Interval Upper Bound", 0.5, 1.0, 0.99, 0.01)
            
            if st.button("📊 Analyze Profit Distribution", type="primary"):
                with st.spinner("Computing profit distributions..."):
                    try:
                        profit_analysis = TwoStageCapacityAndProcurementPlanning.plot_profit_distribution_over_time(
                            decisions=st.session_state.decisions,
                            scenarios=st.session_state.scenarios_reduced,
                            prob=st.session_state.prob_reduced,
                            confidence_levels=[conf_lower, conf_upper],
                            show_stats=True,
                            **st.session_state.opt_params
                        )
                        st.session_state.profit_analysis = profit_analysis
                        
                        st.success("✅ Analysis completed!")
                        
                        # Display summary stats
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Expected Cumulative Profit", 
                                     f"€{profit_analysis['cumulative_stats']['expected_cumulative']:,.0f}")
                        with col2:
                            st.metric("Standard Deviation",
                                     f"€{profit_analysis['cumulative_stats']['std_cumulative']:,.0f}")
                        with col3:
                            sharpe = profit_analysis['cumulative_stats']['expected_cumulative'] / profit_analysis['cumulative_stats']['std_cumulative']
                            st.metric("Sharpe-like Ratio", f"{sharpe:.2f}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
        
        elif analysis_type == "Backtesting":
            st.subheader("Backtesting with Actual Data")
            
            st.info("Simulate optimization performance using actual realized market data")
            
            # Date range selection
            if st.session_state.data is not None:
                max_date = st.session_state.data.index.max()
                min_backtest_date = st.session_state.decisions.index.min()
                
                backtest_start = st.date_input("Backtest Start", value=pd.to_datetime("2020-01-01"))
                backtest_end = st.date_input("Backtest End", value=min(pd.to_datetime("2021-12-01"), max_date))
                
                if st.button("🔬 Run Backtest", type="primary"):
                    with st.spinner("Running backtest simulation..."):
                        try:
                            actual_data = st.session_state.data.loc[
                                backtest_start.strftime('%Y-%m-%d'):backtest_end.strftime('%Y-%m-%d')
                            ]
                            
                            backtest_results = TwoStageCapacityAndProcurementPlanning.backtesting_simulation(
                                decisions=st.session_state.decisions,
                                actual_future_data=actual_data,
                                plot_results=True,
                                **st.session_state.opt_params,
                                real_prices={'P': 800, 'C': 400, 'D': 50000}
                            )
                            
                            st.session_state.backtest_results = backtest_results
                            
                            st.success("✅ Backtest completed!")
                            
                            # Display summary
                            st.subheader("Backtest Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Profit",
                                         f"€{backtest_results['summary_stats']['total_profit']:,.0f}")
                            with col2:
                                st.metric("Avg Monthly Profit",
                                         f"€{backtest_results['summary_stats']['average_monthly_profit']:,.0f}")
                            with col3:
                                st.metric("Capacity Utilization",
                                         f"{backtest_results['summary_stats']['average_capacity_utilization']:.1%}")
                            with col4:
                                st.metric("Demand Satisfaction",
                                         f"{backtest_results['summary_stats']['average_demand_satisfaction']:.1%}")
                            
                        except Exception as e:
                            st.error(f"❌ Backtest failed: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Progress")
progress = 0
if st.session_state.data is not None:
    progress += 1
if st.session_state.var_model is not None:
    progress += 1
if st.session_state.scenario_levels is not None:
    progress += 1
if st.session_state.scenarios_reduced is not None:
    progress += 1
if st.session_state.decisions is not None:
    progress += 1

st.sidebar.progress(progress / 5)
st.sidebar.write(f"Completed: {progress}/5 steps")
