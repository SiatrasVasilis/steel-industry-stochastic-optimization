"""
Optimization Result Objects

This module provides two flat result containers for optimization output:

- ``OptimizationResult``: Holds decisions, scenario-level outcomes, and risk metrics.
- ``BacktestResult``: Holds realized performance, operations, and comparison to scenarios.

Both classes provide ``summary()``, ``plot()``, and ``__repr__()`` for convenient
inspection.  ``BacktestResult`` adds ``detailed_report()`` for full text output.

Design Philosophy
-----------------
Result objects are *dumb containers* — they hold data and know how to display it.
The model classes are responsible for populating them.  This keeps the interface
stable even as new model types are added.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# Optimization Result
# =============================================================================

@dataclass
class OptimizationResult:
    """
    Container for stochastic optimization output.

    Attributes
    ----------
    decisions : pd.DataFrame
        First-stage (here-and-now) decisions indexed by Date.
    scenario_profits : pd.DataFrame
        Per-scenario, per-period profits.
        Columns = scenario IDs, Index = Date.
    probabilities : pd.Series
        Scenario probabilities (sums to 1).
    objective_value : float
        Optimal objective function value.
    risk_metrics : dict[str, float]
        Pre-computed risk metrics (expected_profit, var_95, cvar_95, …).
    stage2_results : dict[str, pd.DataFrame]
        Scenario-level recourse variables (production, sales, etc.).
    metadata : dict[str, Any]
        Model-specific extras (solver, params, timings, …).
    """

    decisions: pd.DataFrame
    scenario_profits: pd.DataFrame
    probabilities: pd.Series
    objective_value: float
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    stage2_results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Computed helpers
    # -----------------------------------------------------------------

    @property
    def total_profits(self) -> pd.Series:
        """Total (cumulative across periods) profit per scenario."""
        return self.scenario_profits.sum(axis=0)

    @property
    def n_scenarios(self) -> int:
        return len(self.probabilities)

    @property
    def horizon(self) -> int:
        return len(self.scenario_profits)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising key decision and risk metrics.

        Returns
        -------
        pd.DataFrame
            One-column frame labelled ``'Value'`` with all metrics.
        """
        totals = self.total_profits
        weighted = (totals * self.probabilities).sum()

        rows = {
            'Expected Profit': weighted,
            'Profit Std': totals.std(),
            'Min Profit': totals.min(),
            'Max Profit': totals.max(),
            'Median Profit': totals.median(),
        }
        rows.update(self.risk_metrics)

        return pd.DataFrame.from_dict(rows, orient='index', columns=['Value'])

    # -----------------------------------------------------------------
    # __repr__
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        totals = self.total_profits
        weighted = (totals * self.probabilities).sum()
        rm = self.risk_metrics

        parts = [
            "=" * 60,
            "  OPTIMIZATION RESULT",
            "=" * 60,
            f"  Model           : {self.metadata.get('model', '?')}",
            f"  Solver          : {self.metadata.get('solver', '?')}",
            f"  Scenarios       : {self.n_scenarios}",
            f"  Horizon         : {self.horizon} periods",
            "-" * 60,
            "  DECISIONS (avg over horizon)",
            "-" * 60,
        ]

        for col in self.decisions.columns:
            parts.append(f"    {col:20s}: {self.decisions[col].mean():>14,.2f}")

        parts += [
            "-" * 60,
            "  RISK METRICS",
            "-" * 60,
            f"    {'Expected Profit':20s}: {weighted:>14,.2f}",
            f"    {'Profit Std':20s}: {totals.std():>14,.2f}",
        ]

        metric_order = [
            'VaR_95', 'VaR_99', 'CVaR_95', 'CVaR_99',
            'Sharpe', 'Min_Profit', 'Max_Profit',
        ]
        for key in metric_order:
            if key in rm:
                parts.append(f"    {key:20s}: {rm[key]:>14,.2f}")

        # Any remaining metrics not in the ordered list
        for key, val in rm.items():
            if key not in metric_order:
                parts.append(f"    {key:20s}: {val:>14,.2f}")

        parts.append("=" * 60)
        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    def plot(self, height: int = 1200) -> None:
        """
        Interactive Plotly optimization dashboard.

        Layout (3 rows x 2 columns):
        - Row 1: Risk metrics KPI table (spans both columns)
        - Row 2: Profit distribution histogram | Cumulative profit fan chart
        - Row 3: Revenue, costs & profit breakdown (spans both columns)
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        totals = self.total_profits
        weighted = float((totals * self.probabilities).sum())
        rm = self.risk_metrics

        # --- Curated KPI list (no duplicates) ---
        def _fmt(v: float) -> str:
            return f"\u20ac{v:,.0f}" if abs(v) > 10 else f"{v:.4f}"

        kpis = [
            ("E[Profit]", _fmt(weighted)),
            ("Profit Std", _fmt(float(totals.std()))),
            ("Median", _fmt(float(totals.median()))),
            ("Min Profit", _fmt(float(totals.min()))),
            ("Max Profit", _fmt(float(totals.max()))),
            ("VaR 95%", _fmt(rm["VaR_95"]) if "VaR_95" in rm else "\u2014"),
            ("CVaR 95%", _fmt(rm["CVaR_95"]) if "CVaR_95" in rm else "\u2014"),
            ("VaR 99%", _fmt(rm["VaR_99"]) if "VaR_99" in rm else "\u2014"),
            ("CVaR 99%", _fmt(rm["CVaR_99"]) if "CVaR_99" in rm else "\u2014"),
            ("Sharpe", f"{rm['Sharpe']:.2f}" if "Sharpe" in rm else "\u2014"),
        ]

        metric_names = [k for k, _ in kpis]
        metric_values = [v for _, v in kpis]

        # --- Build subplot grid ---
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.22, 0.39, 0.39],
            vertical_spacing=0.07,
            horizontal_spacing=0.10,
            specs=[
                [{"type": "table", "colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None],
            ],
            subplot_titles=[
                "",
                "Total Profit Distribution", "Cumulative Profit Fan Chart",
                "Revenue, Costs & Profit Breakdown",
            ],
        )

        # ================================================================
        # Row 1: Risk Metrics KPI Table (vertical Metric | Value)
        # ================================================================
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="#4C72B0",
                    font=dict(color="white", size=12),
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=[metric_names, metric_values],
                    fill_color=[["#f0f4f8", "#e8ecf1"] * (len(kpis) // 2 + 1)],
                    font=dict(size=12),
                    align=["left", "right"],
                    height=26,
                ),
                columnwidth=[1, 1.5],
            ),
            row=1, col=1,
        )

        # ================================================================
        # Row 2, Col 1: Profit Distribution Histogram
        # ================================================================
        # Compute histogram max for vertical line heights
        hist_counts, _ = np.histogram(totals, bins=30)
        y_top = float(hist_counts.max()) * 1.05

        fig.add_trace(
            go.Histogram(
                x=totals, nbinsx=30,
                marker_color="#4C72B0", opacity=0.8,
                name="Profit",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # E[Profit] vertical line
        fig.add_trace(
            go.Scatter(
                x=[weighted, weighted],
                y=[0, y_top],
                mode="lines",
                line=dict(color="green", width=2, dash="dash"),
                name=f"E[Profit] = \u20ac{weighted:,.0f}",
            ),
            row=2, col=1,
        )

        # VaR 95% vertical line
        if "VaR_95" in self.risk_metrics:
            var95 = self.risk_metrics["VaR_95"]
            fig.add_trace(
                go.Scatter(
                    x=[var95, var95],
                    y=[0, y_top],
                    mode="lines",
                    line=dict(color="orange", width=1.5, dash="dash"),
                    name=f"VaR 95% = \u20ac{var95:,.0f}",
                ),
                row=2, col=1,
            )

        # CVaR 95% vertical line
        if "CVaR_95" in self.risk_metrics:
            cvar95 = self.risk_metrics["CVaR_95"]
            fig.add_trace(
                go.Scatter(
                    x=[cvar95, cvar95],
                    y=[0, y_top],
                    mode="lines",
                    line=dict(color="red", width=1.5, dash="dash"),
                    name=f"CVaR 95% = \u20ac{cvar95:,.0f}",
                ),
                row=2, col=1,
            )

        fig.update_xaxes(title_text="Total Profit (\u20ac)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        # ================================================================
        # Row 2, Col 2: Cumulative Profit Fan Chart
        # ================================================================
        cum = self.scenario_profits.cumsum(axis=0)
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        q_df = cum.T.quantile(quantiles).T
        dates = cum.index

        # P5-P95 band
        fig.add_trace(
            go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(q_df[0.95]) + list(q_df[0.05].iloc[::-1]),
                fill="toself", fillcolor="rgba(76,114,176,0.15)",
                line=dict(width=0), name="P5\u2013P95",
                showlegend=True,
            ),
            row=2, col=2,
        )

        # P25-P75 band
        fig.add_trace(
            go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(q_df[0.75]) + list(q_df[0.25].iloc[::-1]),
                fill="toself", fillcolor="rgba(76,114,176,0.30)",
                line=dict(width=0), name="P25\u2013P75",
                showlegend=True,
            ),
            row=2, col=2,
        )

        # Median line
        fig.add_trace(
            go.Scatter(
                x=dates, y=q_df[0.50],
                mode="lines",
                line=dict(color="#4C72B0", width=2),
                name="Median",
            ),
            row=2, col=2,
        )

        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Profit (\u20ac)", row=2, col=2)

        # ================================================================
        # Row 3: Revenue, Costs & Profit Breakdown (horizontal bars)
        # ================================================================
        self._plot_cost_breakdown_plotly(fig, row=3, col=1)

        # ================================================================
        # Layout
        # ================================================================
        fig.update_layout(
            height=height,
            template="plotly_white",
            title=dict(
                text="Optimization Result Dashboard",
                font=dict(size=16),
                x=0.5,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.08,
                xanchor="center", x=0.5,
            ),
            margin=dict(t=60, b=80),
        )

        fig.show()

    # helper
    def _plot_cost_breakdown_plotly(
        self, fig: Any, row: int, col: int
    ) -> None:
        """Add waterfall-style horizontal bars for revenue/costs/profit to a Plotly figure."""
        import plotly.graph_objects as go

        meta = self.metadata
        cost_data = meta.get("cost_components", {})

        if not cost_data:
            fig.add_annotation(
                text="No cost data available",
                xref=f"x{2 * (row - 1) + col}" if row > 1 or col > 1 else "x",
                yref=f"y{2 * (row - 1) + col}" if row > 1 or col > 1 else "y",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="gray"),
                row=row, col=col,
            )
            return

        total_profits = self.total_profits
        expected_profit = float((total_profits * self.probabilities).sum())

        first_val = next(iter(cost_data.values()), None)
        is_scalar = np.isscalar(first_val) or isinstance(first_val, (int, float))

        if is_scalar:
            total_costs = sum(float(v) for v in cost_data.values())
        else:
            total_costs = sum(np.sum(v) for v in cost_data.values())

        total_revenue = expected_profit + total_costs

        # Build categories & values
        categories = ["Revenue"]
        values = [total_revenue]
        colors_list = ["#55A868"]

        cost_colors = [
            "#C44E52", "#DD8452", "#CCB974", "#4C72B0", "#8172B2",
            "#937860", "#64B5CD", "#8C564B", "#E377C2", "#D65F5F",
        ]

        for i, (label, val) in enumerate(cost_data.items()):
            cost_val = float(val) if is_scalar else float(np.sum(val))
            if cost_val > 0:
                categories.append(label.replace("_", " "))
                values.append(-cost_val)
                colors_list.append(cost_colors[i % len(cost_colors)])

        categories.append("Profit")
        values.append(expected_profit)
        colors_list.append("#2C7BB6")

        scale = 1e6
        values_scaled = [v / scale for v in values]

        fig.add_trace(
            go.Bar(
                y=categories,
                x=values_scaled,
                orientation="h",
                marker_color=colors_list,
                text=[f"\u20ac{abs(v):.1f}M" for v in values_scaled],
                textposition="outside",
                showlegend=False,
            ),
            row=row, col=col,
        )

        fig.update_xaxes(title_text="\u20ac Millions", row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)

        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[categories[0], categories[-1]],
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False,
            ),
            row=row, col=col,
        )

    # -----------------------------------------------------------------
    # Plotly Decision Dashboard
    # -----------------------------------------------------------------

    def _compute_stage2_percentiles(
        self,
        var_name: str,
        percentiles: List[float] = [10, 50, 90],
    ) -> pd.DataFrame:
        """
        Compute percentiles across scenarios for a stage-2 variable.

        Parameters
        ----------
        var_name : str
            Name of variable in stage2_results (e.g., 'y_opt', 'I_rm').
        percentiles : list of float
            Percentiles to compute (0-100 scale).

        Returns
        -------
        pd.DataFrame
            Columns = percentile labels ('P10', 'P50', 'P90'), Index = dates.
        """
        if var_name not in self.stage2_results:
            return pd.DataFrame()
        
        df = self.stage2_results[var_name].astype(float)
        result = {}
        for p in percentiles:
            result[f"P{int(p)}"] = df.apply(lambda row: np.percentile(row.dropna(), p), axis=1)
        return pd.DataFrame(result, index=df.index)

    def plot_decisions(
        self,
        percentile_bands: Tuple[float, float] = (10, 90),
        height: int = 1200,
        width: int = 1400,
        save_path: Optional[Union[str, Path]] = "auto",
        show: bool = True,
    ) -> "go.Figure":
        """
        Interactive Plotly dashboard showing monthly decisions.

        Panels:
        - A: First-stage decisions (x_fix, x_opt, P_base)
        - B: Recourse procurement (y_opt, x_spot) with percentile bands
        - C: Production (P_base, P_flex) with percentile bands
        - D: Inventory levels (I_rm, I_fg) with percentile bands
        - E: Sales & unmet demand with percentile bands
        - F: Monthly cost breakdown over time

        Parameters
        ----------
        percentile_bands : tuple
            Lower and upper percentile bounds (default: 10th, 90th).
        height : int
            Figure height in pixels.
        width : int
            Figure width in pixels.
        save_path : str, Path, or "auto"
            If "auto", saves to results/decisions_<model>.html.
            If None, does not save.
        show : bool
            Whether to display the figure.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plot_decisions(). Install with: pip install plotly")
        
        model_name = self.metadata.get('model', 'Model')
        p_low, p_high = percentile_bands
        dates = self.decisions.index
        
        # Color scheme
        colors = {
            'primary': '#4C72B0',
            'secondary': '#DD8452',
            'tertiary': '#55A868',
            'quaternary': '#C44E52',
            'band': 'rgba(76, 114, 176, 0.2)',
            'band2': 'rgba(221, 132, 82, 0.2)',
            'band3': 'rgba(85, 168, 104, 0.2)',
        }
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "A: First-Stage Decisions",
                "B: Recourse Procurement (Fan Chart)",
                "C: Production (Fan Chart)",
                "D: Inventory Levels (Fan Chart)",
                "E: Sales & Shortages (Fan Chart)",
                "F: Monthly Profit by Scenario",
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )
        
        # ---- Panel A: First-stage decisions ----
        for col, color, name in [
            ('x_fix', colors['primary'], 'Fixed Contracts'),
            ('x_opt', colors['secondary'], 'Framework Reserv.'),
            ('P_base', colors['tertiary'], 'Base Production'),
        ]:
            if col in self.decisions.columns:
                fig.add_trace(go.Scatter(
                    x=dates, y=self.decisions[col],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup='stage1',
                    showlegend=True,
                ), row=1, col=1)
        
        # ---- Panel B: Recourse procurement (y_opt, x_spot) ----
        for var, color, band_color, name in [
            ('y_opt', colors['primary'], colors['band'], 'Framework Exercise'),
            ('x_spot', colors['quaternary'], 'rgba(196, 78, 82, 0.2)', 'Spot Purchases'),
        ]:
            pct = self._compute_stage2_percentiles(var, [p_low, 50, p_high])
            if not pct.empty:
                # Band
                fig.add_trace(go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=list(pct[f'P{int(p_low)}']) + list(pct[f'P{int(p_high)}'])[::-1],
                    fill='toself',
                    fillcolor=band_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{name} P{int(p_low)}-P{int(p_high)}',
                    legendgroup='stage2_proc',
                    showlegend=False,
                    hoverinfo='skip',
                ), row=1, col=2)
                # Median
                fig.add_trace(go.Scatter(
                    x=dates, y=pct['P50'],
                    mode='lines',
                    name=f'{name} (median)',
                    line=dict(color=color, width=2),
                    legendgroup='stage2_proc',
                ), row=1, col=2)
        
        # ---- Panel C: Production ----
        # P_base is first-stage (constant)
        if 'P_base' in self.decisions.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=self.decisions['P_base'],
                mode='lines',
                name='Base Production',
                line=dict(color=colors['primary'], width=2, dash='dash'),
                legendgroup='prod',
            ), row=2, col=1)
        
        # P_flex is stage-2
        pct_flex = self._compute_stage2_percentiles('P_flex', [p_low, 50, p_high])
        if not pct_flex.empty:
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates)[::-1],
                y=list(pct_flex[f'P{int(p_low)}']) + list(pct_flex[f'P{int(p_high)}'])[::-1],
                fill='toself',
                fillcolor=colors['band2'],
                line=dict(color='rgba(0,0,0,0)'),
                name=f'Flex P{int(p_low)}-P{int(p_high)}',
                legendgroup='prod',
                showlegend=False,
                hoverinfo='skip',
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=dates, y=pct_flex['P50'],
                mode='lines',
                name='Flex Production (median)',
                line=dict(color=colors['secondary'], width=2),
                legendgroup='prod',
            ), row=2, col=1)
        
        # ---- Panel D: Inventory ----
        for var, color, band_color, name in [
            ('I_rm', colors['primary'], colors['band'], 'RM Inventory'),
            ('I_fg', colors['tertiary'], colors['band3'], 'FG Inventory'),
        ]:
            pct = self._compute_stage2_percentiles(var, [p_low, 50, p_high])
            if not pct.empty:
                fig.add_trace(go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=list(pct[f'P{int(p_low)}']) + list(pct[f'P{int(p_high)}'])[::-1],
                    fill='toself',
                    fillcolor=band_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{name} P{int(p_low)}-P{int(p_high)}',
                    legendgroup='inv',
                    showlegend=False,
                    hoverinfo='skip',
                ), row=2, col=2)
                fig.add_trace(go.Scatter(
                    x=dates, y=pct['P50'],
                    mode='lines',
                    name=f'{name} (median)',
                    line=dict(color=color, width=2),
                    legendgroup='inv',
                ), row=2, col=2)
        
        # Safety stock target lines (if SafetyStockModel)
        if 'policy' in self.metadata and 'SS_FG' in self.metadata.get('policy', {}):
            ss_fg = self.metadata['policy']['SS_FG']
            ss_rm = self.metadata['policy'].get('SS_RM', 0)
            fig.add_trace(go.Scatter(
                x=[dates[0], dates[-1]], y=[ss_fg, ss_fg],
                mode='lines',
                name='SS_FG Target',
                line=dict(color=colors['tertiary'], width=1, dash='dot'),
                legendgroup='inv',
            ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=[dates[0], dates[-1]], y=[ss_rm, ss_rm],
                mode='lines',
                name='SS_RM Target',
                line=dict(color=colors['primary'], width=1, dash='dot'),
                legendgroup='inv',
            ), row=2, col=2)
        
        # ---- Panel E: Sales & Shortages ----
        for var, color, band_color, name in [
            ('S', colors['tertiary'], colors['band3'], 'Sales'),
            ('U', colors['quaternary'], 'rgba(196, 78, 82, 0.2)', 'Unmet Demand'),
        ]:
            pct = self._compute_stage2_percentiles(var, [p_low, 50, p_high])
            if not pct.empty:
                fig.add_trace(go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=list(pct[f'P{int(p_low)}']) + list(pct[f'P{int(p_high)}'])[::-1],
                    fill='toself',
                    fillcolor=band_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{name} P{int(p_low)}-P{int(p_high)}',
                    legendgroup='service',
                    showlegend=False,
                    hoverinfo='skip',
                ), row=3, col=1)
                fig.add_trace(go.Scatter(
                    x=dates, y=pct['P50'],
                    mode='lines',
                    name=f'{name} (median)',
                    line=dict(color=color, width=2),
                    legendgroup='service',
                ), row=3, col=1)
        
        # ---- Panel F: Monthly profit fan chart ----
        profit_pct = self.scenario_profits.T.quantile([p_low/100, 0.5, p_high/100]).T
        profit_pct.columns = [f'P{int(p_low)}', 'P50', f'P{int(p_high)}']
        
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=list(profit_pct[f'P{int(p_low)}']) + list(profit_pct[f'P{int(p_high)}'])[::-1],
            fill='toself',
            fillcolor='rgba(85, 168, 104, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'Profit P{int(p_low)}-P{int(p_high)}',
            legendgroup='profit',
            showlegend=False,
            hoverinfo='skip',
        ), row=3, col=2)
        fig.add_trace(go.Scatter(
            x=dates, y=profit_pct['P50'],
            mode='lines',
            name='Profit (median)',
            line=dict(color=colors['tertiary'], width=2),
            legendgroup='profit',
        ), row=3, col=2)
        
        # Expected profit line
        expected_profit = (self.scenario_profits * self.probabilities).sum(axis=1)
        fig.add_trace(go.Scatter(
            x=dates, y=expected_profit,
            mode='lines',
            name='E[Profit]',
            line=dict(color=colors['primary'], width=2, dash='dash'),
            legendgroup='profit',
        ), row=3, col=2)
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f"📊 Monthly Decision Dashboard — {model_name}",
                font=dict(size=18),
            ),
            height=height,
            width=width,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.08,
                xanchor='center',
                x=0.5,
            ),
            hovermode='x unified',
        )
        
        # Axis labels
        fig.update_yaxes(title_text="Tons RM", row=1, col=1)
        fig.update_yaxes(title_text="Tons RM", row=1, col=2)
        fig.update_yaxes(title_text="Tons FG", row=2, col=1)
        fig.update_yaxes(title_text="Tons", row=2, col=2)
        fig.update_yaxes(title_text="Tons FG", row=3, col=1)
        fig.update_yaxes(title_text="€", row=3, col=2)
        
        # Save
        if save_path:
            if save_path == "auto":
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                safe_name = model_name.replace(" ", "_").lower()
                save_path = results_dir / f"decisions_{safe_name}.html"
            fig.write_html(str(save_path))
            print(f"Dashboard saved to: {save_path}")
        
        if show:
            fig.show()
        
        return fig


# =============================================================================
# Compare Decisions (Standalone Function)
# =============================================================================

def compare_decisions(
    result_a: OptimizationResult,
    result_b: OptimizationResult,
    name_a: str = "Model A",
    name_b: str = "Model B",
    percentile_bands: Tuple[float, float] = (10, 90),
    height: int = 1000,
    width: int = 1400,
    save_path: Optional[Union[str, Path]] = "auto",
    show: bool = True,
) -> "go.Figure":
    """
    Side-by-side comparison of two optimization results.

    Creates an interactive Plotly dashboard comparing:
    - First-stage decisions
    - Recourse procurement medians
    - Inventory levels
    - Profit distributions

    Parameters
    ----------
    result_a, result_b : OptimizationResult
        Two results to compare.
    name_a, name_b : str
        Display names for each result.
    percentile_bands : tuple
        Lower/upper percentile bounds.
    height, width : int
        Figure dimensions.
    save_path : str, Path, or "auto"
        Save location. "auto" saves to results/comparison.html.
    show : bool
        Display the figure.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required. Install with: pip install plotly")
    
    p_low, p_high = percentile_bands
    dates = result_a.decisions.index
    
    # Colors for each model
    colors_a = {'line': '#4C72B0', 'band': 'rgba(76, 114, 176, 0.15)'}
    colors_b = {'line': '#DD8452', 'band': 'rgba(221, 132, 82, 0.15)'}
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "First-Stage: Fixed Contracts (x_fix)",
            "First-Stage: Framework Reserv. (x_opt)",
            "Inventory: FG (median ± band)",
            "Profit Distribution",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    # ---- Panel 1: x_fix comparison ----
    for result, name, color in [(result_a, name_a, colors_a), (result_b, name_b, colors_b)]:
        if 'x_fix' in result.decisions.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=result.decisions['x_fix'],
                mode='lines+markers',
                name=f'{name}',
                line=dict(color=color['line'], width=2),
                marker=dict(size=5),
                legendgroup=name,
            ), row=1, col=1)
    
    # ---- Panel 2: x_opt comparison ----
    for result, name, color in [(result_a, name_a, colors_a), (result_b, name_b, colors_b)]:
        if 'x_opt' in result.decisions.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=result.decisions['x_opt'],
                mode='lines+markers',
                name=f'{name}',
                line=dict(color=color['line'], width=2),
                marker=dict(size=5),
                legendgroup=name,
                showlegend=False,
            ), row=1, col=2)
    
    # ---- Panel 3: I_fg comparison with bands ----
    for result, name, color in [(result_a, name_a, colors_a), (result_b, name_b, colors_b)]:
        pct = result._compute_stage2_percentiles('I_fg', [p_low, 50, p_high])
        if not pct.empty:
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates)[::-1],
                y=list(pct[f'P{int(p_low)}']) + list(pct[f'P{int(p_high)}'])[::-1],
                fill='toself',
                fillcolor=color['band'],
                line=dict(color='rgba(0,0,0,0)'),
                name=f'{name} P{int(p_low)}-P{int(p_high)}',
                legendgroup=name,
                showlegend=False,
                hoverinfo='skip',
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=dates, y=pct['P50'],
                mode='lines',
                name=f'{name}',
                line=dict(color=color['line'], width=2),
                legendgroup=name,
                showlegend=False,
            ), row=2, col=1)
    
    # ---- Panel 4: Profit distribution histograms ----
    for result, name, color in [(result_a, name_a, colors_a), (result_b, name_b, colors_b)]:
        profits = result.total_profits
        fig.add_trace(go.Histogram(
            x=profits,
            name=f'{name}',
            marker_color=color['line'],
            opacity=0.6,
            legendgroup=name,
            showlegend=False,
        ), row=2, col=2)
        # Vertical line for expected profit
        exp_profit = (profits * result.probabilities).sum()
        fig.add_vline(
            x=exp_profit,
            line=dict(color=color['line'], width=2, dash='dash'),
            row=2, col=2,
        )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"📊 Decision Comparison: {name_a} vs {name_b}",
            font=dict(size=18),
        ),
        height=height,
        width=width,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.12,
            xanchor='center',
            x=0.5,
        ),
        hovermode='x unified',
        barmode='overlay',
    )
    
    fig.update_yaxes(title_text="Tons RM", row=1, col=1)
    fig.update_yaxes(title_text="Tons RM", row=1, col=2)
    fig.update_yaxes(title_text="Tons FG", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    fig.update_xaxes(title_text="Total Profit (€)", row=2, col=2)
    
    # Save
    if save_path:
        if save_path == "auto":
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            save_path = results_dir / "comparison.html"
        fig.write_html(str(save_path))
        print(f"Comparison saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig


# =============================================================================
# Backtest Result
# =============================================================================

@dataclass
class BacktestResult:
    """
    Container for out-of-sample backtesting output.

    Attributes
    ----------
    timeline : pd.DataFrame
        All monthly time-series data (profits, costs, revenue, operations)
        indexed by Date.  Columns include at minimum:
        ``Profit``, ``Revenue``, ``Production``, ``Sales``,
        ``Demand_Actual``, ``Capacity_Utilization``, ``Demand_Satisfaction``.
    decisions : pd.DataFrame
        The first-stage decisions that were evaluated.
    metrics : dict[str, float]
        Scalar summary metrics (total_profit, avg utilization, …).
    scenario_comparison : dict[str, float], optional
        How realized outcome compares to the scenario distribution
        (realized_percentile, realized_vs_expected, …).
    metadata : dict[str, Any]
        Model-specific extras.
    """

    timeline: pd.DataFrame
    decisions: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    scenario_comparison: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------

    @property
    def profit(self) -> pd.Series:
        return self.timeline["Profit"]

    @property
    def cumulative_profit(self) -> pd.Series:
        return self.profit.cumsum()

    @property
    def revenue(self) -> pd.Series:
        return self.timeline["Revenue"]

    @property
    def demand_satisfaction(self) -> pd.Series:
        return self.timeline["Demand_Satisfaction"]

    @property
    def capacity_utilization(self) -> pd.Series:
        return self.timeline["Capacity_Utilization"]

    @property
    def cost_columns(self) -> list:
        """Return names of all cost columns present in timeline."""
        return [c for c in self.timeline.columns if c.endswith("_Costs") and c != "Total_Costs"]

    @property
    def costs(self) -> pd.DataFrame:
        """Return a sub-DataFrame with only cost columns."""
        return self.timeline[self.cost_columns]

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a one-column DataFrame of key metrics."""
        rows = dict(self.metrics)
        rows.setdefault("Total Profit", self.profit.sum())
        rows.setdefault("Avg Monthly Profit", self.profit.mean())
        rows.setdefault("Profit Volatility", self.profit.std())
        rows.setdefault("Avg Capacity Util %", self.capacity_utilization.mean() * 100)
        rows.setdefault("Avg Demand Satis %", self.demand_satisfaction.mean() * 100)
        rows.update(self.scenario_comparison)
        return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])

    # -----------------------------------------------------------------
    # __repr__
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.timeline)
        total = self.profit.sum()
        avg = self.profit.mean()
        std = self.profit.std()
        cap = self.capacity_utilization.mean() * 100
        dem = self.demand_satisfaction.mean() * 100

        parts = [
            "=" * 60,
            "  BACKTEST RESULT",
            "=" * 60,
            f"  Periods         : {n}",
            f"  Date range      : {self.timeline.index.min().strftime('%Y-%m')} → "
            f"{self.timeline.index.max().strftime('%Y-%m')}",
            "-" * 60,
            "  FINANCIAL PERFORMANCE",
            "-" * 60,
            f"    {'Total Profit':24s}: {total:>14,.2f} €",
            f"    {'Avg Monthly Profit':24s}: {avg:>14,.2f} €",
            f"    {'Profit Volatility':24s}: {std:>14,.2f} €",
            f"    {'Best Month':24s}: {self.profit.max():>14,.2f} €  ({self.profit.idxmax().strftime('%Y-%m')})",
            f"    {'Worst Month':24s}: {self.profit.min():>14,.2f} €  ({self.profit.idxmin().strftime('%Y-%m')})",
            "-" * 60,
            "  COST BREAKDOWN (total over horizon)",
            "-" * 60,
        ]

        for col in self.cost_columns:
            label = col.replace("_Costs", "").replace("_", " ")
            parts.append(f"    {label:24s}: {self.timeline[col].sum():>14,.2f} €")

        parts += [
            f"    {'Total Revenue':24s}: {self.revenue.sum():>14,.2f} €",
            "-" * 60,
            "  OPERATIONAL PERFORMANCE",
            "-" * 60,
            f"    {'Avg Capacity Util':24s}: {cap:>13.1f} %",
            f"    {'Avg Demand Satisfaction':24s}: {dem:>13.1f} %",
        ]

        if "Unmet_Demand" in self.timeline.columns:
            total_unmet = self.timeline["Unmet_Demand"].sum()
            pct_unmet_months = (self.timeline["Unmet_Demand"] > 0.01).mean() * 100
            parts.append(f"    {'Total Unmet Demand':24s}: {total_unmet:>14,.2f}")
            parts.append(f"    {'Months w/ Shortage':24s}: {pct_unmet_months:>13.1f} %")

        # Contracts / flexibility
        if "Cap_flex" in self.timeline.columns:
            avg_flex = self.timeline["Cap_flex"].mean()
            avg_base = self.timeline["Cap_base"].mean() if "Cap_base" in self.timeline.columns else 0
            flex_pct = (avg_flex / avg_base * 100) if avg_base > 0 else 0
            parts += [
                "-" * 60,
                "  CAPACITY & CONTRACTS",
                "-" * 60,
                f"    {'Avg Base Capacity':24s}: {avg_base:>14,.2f}",
                f"    {'Avg Flex Capacity':24s}: {avg_flex:>14,.2f}  ({flex_pct:.1f}% of base)",
            ]
        if "Q_base" in self.timeline.columns:
            parts.append(f"    {'Avg Base Contract':24s}: {self.timeline['Q_base'].mean():>14,.2f}")

        # Scenario comparison
        if self.scenario_comparison:
            parts += [
                "-" * 60,
                "  SCENARIO COMPARISON",
                "-" * 60,
            ]
            for key, val in self.scenario_comparison.items():
                label = key.replace("_", " ").title()
                if "percentile" in key.lower():
                    parts.append(f"    {label:24s}: {val:>13.1f} %")
                else:
                    parts.append(f"    {label:24s}: {val:>14,.2f}")

        parts.append("=" * 60)
        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Detailed report
    # -----------------------------------------------------------------

    def detailed_report(self) -> str:
        """
        Multi-section text report covering finances, operations,
        costs, contracts, and comparison to the scenario distribution.
        """
        parts = [repr(self)]

        # Period-by-period table
        display_cols = ["Profit", "Revenue"] + self.cost_columns
        operational = ["Production", "Sales", "Demand_Actual",
                       "Capacity_Utilization", "Demand_Satisfaction"]
        display_cols += [c for c in operational if c in self.timeline.columns]

        parts += [
            "",
            "=" * 80,
            "  PERIOD-BY-PERIOD DETAIL",
            "=" * 80,
            self.timeline[display_cols].to_string(float_format=lambda x: f"{x:,.2f}"),
        ]
        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    def plot(
        self,
        height: int = 1200,
        scenario_result: Optional["OptimizationResult"] = None,
    ) -> None:
        """
        Interactive Plotly backtest dashboard.

        Layout (4 rows x 2 columns):
        - Row 1: KPI summary table (spans both columns)
        - Row 2: Cumulative profit | Monthly profit
        - Row 3: Cost breakdown (horizontal bars) | Revenue vs total costs
        - Row 4: Operational KPIs | Production vs demand
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dates = self.timeline.index
        cost_cols = self.cost_columns

        # --- Compute KPIs ---
        total_profit = self.profit.sum()
        total_revenue = self.revenue.sum()
        total_costs = self.costs.sum().sum() if cost_cols else 0
        margin = total_profit / total_revenue * 100 if total_revenue > 0 else 0
        avg_fill = self.demand_satisfaction.mean() * 100
        avg_cap = self.capacity_utilization.mean() * 100
        n_periods = len(self.timeline)

        # --- Build subplot grid ---
        fig = make_subplots(
            rows=4, cols=2,
            row_heights=[0.08, 0.31, 0.31, 0.31],
            vertical_spacing=0.06,
            horizontal_spacing=0.10,
            specs=[
                [{"type": "table", "colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            subplot_titles=[
                "",
                "Cumulative Profit", "Monthly Profit",
                "Cost Breakdown (Total)", "Revenue vs Total Costs",
                "Operational KPIs (%)", "Production vs Demand",
            ],
        )

        # ================================================================
        # Row 1: KPI Summary Table
        # ================================================================
        kpi_headers = [
            "Total Profit", "Total Revenue", "Total Costs",
            "Profit Margin", "Avg Fill Rate", "Avg Cap. Util.", "Periods",
        ]
        kpi_values = [
            f"\u20ac{total_profit:,.0f}",
            f"\u20ac{total_revenue:,.0f}",
            f"\u20ac{total_costs:,.0f}",
            f"{margin:.1f}%",
            f"{avg_fill:.1f}%",
            f"{avg_cap:.1f}%",
            f"{n_periods}",
        ]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in kpi_headers],
                    fill_color="#F8F9FA",
                    align="center",
                    font=dict(size=11, color="gray"),
                    line_color="#E9ECEF",
                    height=28,
                ),
                cells=dict(
                    values=[[v] for v in kpi_values],
                    fill_color="white",
                    align="center",
                    font=dict(size=14, color="#2C3E50"),
                    line_color="#E9ECEF",
                    height=32,
                ),
            ),
            row=1, col=1,
        )

        # ================================================================
        # Row 2, Left: Cumulative Profit
        # ================================================================
        cum = self.cumulative_profit

        if scenario_result is not None:
            scen_cum = scenario_result.scenario_profits.cumsum(axis=0)
            n = min(len(dates), len(scen_cum))
            scen_slice = scen_cum.iloc[:n]
            q = scen_slice.T.quantile([0.10, 0.50, 0.90]).T
            d_slice = dates[:n]

            fig.add_trace(
                go.Scatter(
                    x=d_slice, y=q[0.90], mode="lines",
                    line=dict(width=0), showlegend=False,
                    hoverinfo="skip",
                ),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=d_slice, y=q[0.10], mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(76,114,176,0.15)",
                    name="P10\u2013P90",
                ),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=d_slice, y=q[0.50], mode="lines",
                    line=dict(color="#4C72B0", dash="dash", width=1.5),
                    name="Median scenario",
                ),
                row=2, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=dates, y=cum, mode="lines",
                line=dict(color="#2C7BB6", width=2.5),
                name="Realized",
            ),
            row=2, col=1,
        )

        # ================================================================
        # Row 2, Right: Monthly Profit
        # ================================================================
        bar_colors = ["#55A868" if v >= 0 else "#C44E52" for v in self.profit]
        fig.add_trace(
            go.Bar(
                x=dates, y=self.profit,
                marker_color=bar_colors,
                name="Monthly Profit",
                showlegend=False,
            ),
            row=2, col=2,
        )
        avg_profit = self.profit.mean()
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]], y=[avg_profit, avg_profit],
                mode="lines",
                line=dict(color="#2C7BB6", dash="dot", width=1.5),
                name=f"Avg = \u20ac{avg_profit:,.0f}",
                showlegend=False,
            ),
            row=2, col=2,
        )

        # ================================================================
        # Row 3, Left: Cost Breakdown (horizontal bars)
        # ================================================================
        if cost_cols:
            cost_totals = self.timeline[cost_cols].sum().sort_values(ascending=True)
            labels = [c.replace("_Costs", "").replace("_", " ") for c in cost_totals.index]
            palette = ["#C44E52", "#DD8452", "#CCB974", "#55A868",
                       "#4C72B0", "#8172B2", "#64B5CD", "#E377C2"]
            bar_colors_cost = palette[:len(labels)]

            fig.add_trace(
                go.Bar(
                    x=cost_totals.values,
                    y=labels,
                    orientation="h",
                    marker_color=bar_colors_cost,
                    text=[f"\u20ac{v:,.0f}" for v in cost_totals.values],
                    textposition="outside",
                    showlegend=False,
                ),
                row=3, col=1,
            )

        # ================================================================
        # Row 3, Right: Revenue vs Total Costs
        # ================================================================
        total_costs_ts = self.costs.sum(axis=1) if cost_cols else pd.Series(0, index=dates)
        fig.add_trace(
            go.Scatter(
                x=dates, y=self.revenue, mode="lines",
                line=dict(color="#55A868", width=2),
                name="Revenue",
            ),
            row=3, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=total_costs_ts, mode="lines",
                line=dict(color="#C44E52", width=2),
                name="Total Costs",
                fill="tonexty",
                fillcolor="rgba(85,168,104,0.08)",
            ),
            row=3, col=2,
        )

        # ================================================================
        # Row 4, Left: Operational KPIs
        # ================================================================
        fig.add_trace(
            go.Scatter(
                x=dates, y=self.demand_satisfaction * 100,
                mode="lines+markers",
                line=dict(color="#2C7BB6", width=2),
                marker=dict(size=5),
                name="Demand Satisfaction",
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=self.capacity_utilization * 100,
                mode="lines+markers",
                line=dict(color="#DD8452", width=2),
                marker=dict(size=5, symbol="square"),
                name="Capacity Utilization",
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]], y=[100, 100],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=4, col=1,
        )

        # ================================================================
        # Row 4, Right: Production vs Demand
        # ================================================================
        if "Production" in self.timeline.columns:
            production = self.timeline["Production"]
        elif "P_base" in self.timeline.columns:
            production = self.timeline["P_base"] + self.timeline.get("P_flex", 0)
        elif "Total_Production" in self.timeline.columns:
            production = self.timeline["Total_Production"]
        else:
            production = None

        demand = (self.timeline["Demand_Actual"]
                  if "Demand_Actual" in self.timeline.columns else None)
        cap_base = (self.timeline["Cap_base"]
                    if "Cap_base" in self.timeline.columns else None)

        if production is not None:
            fig.add_trace(
                go.Bar(
                    x=dates, y=production,
                    marker_color="rgba(76,114,176,0.7)",
                    name="Production",
                ),
                row=4, col=2,
            )
        if demand is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates, y=demand, mode="lines+markers",
                    line=dict(color="#55A868", width=2.5, dash="dash"),
                    marker=dict(size=6, symbol="diamond"),
                    name="Demand",
                ),
                row=4, col=2,
            )
        if cap_base is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates, y=cap_base, mode="lines",
                    line=dict(color="gray", dash="dot", width=1.5),
                    name="Base Capacity",
                ),
                row=4, col=2,
            )

        # ================================================================
        # Layout
        # ================================================================
        fig.update_layout(
            height=height,
            title_text="Backtest Dashboard",
            title_font_size=16,
            title_x=0.5,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.06,
                xanchor="center",
                x=0.5,
                font_size=10,
            ),
            margin=dict(t=60, b=60, l=60, r=30),
        )

        # Axis labels
        fig.update_yaxes(title_text="\u20ac", row=2, col=1)
        fig.update_yaxes(title_text="\u20ac", row=2, col=2)
        fig.update_yaxes(title_text="\u20ac", row=3, col=1)
        fig.update_yaxes(title_text="\u20ac", row=3, col=2)
        fig.update_yaxes(title_text="%", range=[0, 110], row=4, col=1)
        fig.update_yaxes(title_text="Tons", row=4, col=2)

        fig.show()
