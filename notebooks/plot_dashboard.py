# %% notebooks/plot_dashboard.py
# Regenerate the LinkedIn-ready dashboard from saved backtest results.
# No need to rerun the full backtest — loads from results/rolling_replan/full_results_multi.pkl
#
# Usage:
#   python notebooks/plot_dashboard.py
#   or run interactively cell-by-cell in VSCode

import sys, pickle
from pathlib import Path

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path('.').resolve()
sys.path.insert(0, str(_here.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Config ─────────────────────────────────────────────────────────────
# Match these to the backtest run you want to visualise
EXECUTE_MONTHS = 12
BACKTEST_DATES = pd.date_range(start='2007-12', end='2022-12', freq=f'{EXECUTE_MONTHS}MS').astype('str').tolist()

_start = pd.to_datetime(BACKTEST_DATES[0]).strftime('%Y%m')
_end   = pd.to_datetime(BACKTEST_DATES[-1]).strftime('%Y%m')
_freq  = f'{EXECUTE_MONTHS}M'
RESULTS_DIR    = _here.parent / 'results' / f'rolling_replan_{_freq}_{_start}_{_end}'

RISK_AVERSIONS = [0.0, 0.3, 0.7, 1.0]
LAMBDA_LABELS  = {0.0: 'Risk Neutral', 0.3: 'Moderate', 0.7: 'Conservative', 1.0: 'Full CVaR'}
LAMBDA_COLORS  = {0.0: '#1565C0', 0.3: '#00897B', 0.7: '#F57F17', 1.0: '#C62828'}
ANNOTATION_EXT = 10  # months of x-axis padding for right-side value labels

# ── Load ───────────────────────────────────────────────────────────────
with open(RESULTS_DIR / 'full_results_multi.pkl', 'rb') as f:
    saved = pickle.load(f)

ss_monthly_df = saved['ss_monthly']
market_df     = saved['market_df']
wp_ss         = saved['wp_ss']
wp_stoch      = saved['wp_stoch']

ss_profits_s = ss_monthly_df['Profit']
ss_cum       = ss_profits_s.cumsum()
months       = ss_profits_s.index

stoch_data = {}
for lam in RISK_AVERSIONS:
    df = saved['stoch_data'][lam]['monthly']
    stoch_data[lam] = {
        'monthly': df,
        'profits': df['Profit'],
        'cum':     df['Profit'].cumsum(),
    }

# ── KPIs ───────────────────────────────────────────────────────────────
best_lam     = max(RISK_AVERSIONS, key=lambda l: wp_stoch[l]['profit'].sum())
best_total   = wp_stoch[best_lam]['profit'].sum()
total_ss_val = wp_ss['profit'].sum()
best_vss     = best_total - total_ss_val
best_vss_pct = best_vss / abs(total_ss_val) * 100
best_wins    = int((wp_stoch[best_lam]['profit'] > wp_ss['profit']).sum())

replan_dates_dt = pd.to_datetime(BACKTEST_DATES)
date_start      = BACKTEST_DATES[0][:7]
date_end        = BACKTEST_DATES[-1][:7]

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

_n_years = round((pd.to_datetime(BACKTEST_DATES[-1]) - pd.to_datetime(BACKTEST_DATES[0])).days / 365.25)

fig = plt.figure(figsize=(16, 13), facecolor='white')
gs  = gridspec.GridSpec(
    4, 2, figure=fig,
    height_ratios=[0.65, 0.92, 1.20, 0.78],
    hspace=0.52, wspace=0.32,
)

x_right = months[-1] + pd.DateOffset(months=ANNOTATION_EXT)

# ═══════════════════════════════════════════════════════════════════════
# PANEL 0 — KPI Banner (full width)
# ═══════════════════════════════════════════════════════════════════════
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis('off')

ax_kpi.text(0.50, 1.00,
            'Stochastic Procurement Optimization vs. Safety Stock Benchmark',
            transform=ax_kpi.transAxes, fontsize=18, fontweight='bold',
            color='#212121', ha='center', va='top')
ax_kpi.text(0.50, 0.78,
            f'{_n_years}-Year Steel Industry Rolling Backtest  ({date_start} – {date_end})  ·  '
            f'Regime-Switching VAR  ·  12-Month Horizon  ·  {EXECUTE_MONTHS}-Month Replan  ·  '
            f'{len(BACKTEST_DATES)} Windows',
            transform=ax_kpi.transAxes, fontsize=12, color='#757575',
            ha='center', va='top')

kpi_blocks = [
    (f'+€{best_vss/1e6:,.0f}M',              'Total Value of Stochastic Solution',          '#1B5E20'),
    (f'+{best_vss_pct:.1f}%',                f'Profit Uplift  ·  {LAMBDA_LABELS[best_lam]}', '#0D47A1'),
    (f'{best_wins} / {len(BACKTEST_DATES)}',  'Planning Windows Outperformed',               '#4A148C'),
]
for i, (val, lbl, col) in enumerate(kpi_blocks):
    x = 0.16 + i * 0.34
    ax_kpi.text(x, 0.48, val, transform=ax_kpi.transAxes,
                fontsize=30, fontweight='bold', color=col, ha='center', va='center')
    ax_kpi.text(x, 0.12, lbl, transform=ax_kpi.transAxes,
                fontsize=11, color='#616161', ha='center', va='center')

for x_sep in [0.335, 0.665]:
    ax_kpi.add_line(Line2D([x_sep, x_sep], [0.04, 0.65],
                           transform=ax_kpi.transAxes,
                           color='#E0E0E0', lw=1.5))

# ═══════════════════════════════════════════════════════════════════════
# PANEL 1 — Market Environment (full width)
# ═══════════════════════════════════════════════════════════════════════
ax_mkt = fig.add_subplot(gs[1, :])
ax_mkt.plot(market_df.index, market_df['P'], color='#1565C0', lw=1.4,
            label='Steel Price (EUR/Tn)')
ax_mkt.plot(market_df.index, market_df['C'], color='#E65100', lw=1.4,
            label='Scrap Cost (EUR/Tn)')

ax_mkt2 = ax_mkt.twinx()
ax_mkt2.fill_between(market_df.index, market_df['D'] / 1e3, alpha=0.10, color='#4CAF50')
ax_mkt2.plot(market_df.index, market_df['D'] / 1e3, color='#4CAF50', lw=1.0,
             alpha=0.55, label='Demand (k Tn)')
ax_mkt2.set_ylabel('Demand (k Tn / month)', fontsize=9, color='#4CAF50')
ax_mkt2.spines['top'].set_visible(False)
ax_mkt2.tick_params(axis='y', labelsize=8.5, colors='#4CAF50')

for rd in replan_dates_dt:
    ax_mkt.axvline(rd, color='#CFD8DC', lw=0.5, alpha=0.5, zorder=0)

crisis_events = [
    (pd.Timestamp('2008-09-01'), '2008 Financial\nCrisis',  '#B71C1C'),
    (pd.Timestamp('2020-04-01'), 'COVID-19',                '#B71C1C'),
    (pd.Timestamp('2021-07-01'), 'Post-COVID\nSupply Shock','#E65100'),
]
for ts, label, col in crisis_events:
    ax_mkt.annotate(label,
                    xy=(ts, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -4), textcoords='offset points',
                    fontsize=8.5, color=col, ha='center', va='top',
                    fontstyle='italic', alpha=0.9)

ax_mkt.set_ylabel('EUR / Tn', fontsize=10)
ax_mkt.tick_params(axis='both', labelsize=8.5)
ax_mkt.set_title('Market Environment', fontsize=12, fontweight='bold', loc='left')
ax_mkt.grid(True, alpha=0.12)

lines1, labels1 = ax_mkt.get_legend_handles_labels()
lines2, labels2 = ax_mkt2.get_legend_handles_labels()
ax_mkt.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5,
              loc='upper left', framealpha=0.85, ncol=3)

# ═══════════════════════════════════════════════════════════════════════
# PANEL 2 — Cumulative EUR Advantage (middle-left)
# ═══════════════════════════════════════════════════════════════════════
ax_adv = fig.add_subplot(gs[2, 0])
ax_adv.set_xlim(left=months[0], right=x_right)

adv_best = (stoch_data[best_lam]['cum'] - ss_cum) / 1e6
ax_adv.fill_between(months, adv_best, 0,
                    where=(adv_best >= 0), alpha=0.10, color='#43A047',
                    interpolate=True, zorder=0)
ax_adv.fill_between(months, adv_best, 0,
                    where=(adv_best < 0), alpha=0.12, color='#E53935',
                    interpolate=True, zorder=0)

for lam in RISK_AVERSIONS:
    adv = (stoch_data[lam]['cum'] - ss_cum) / 1e6
    ax_adv.plot(months, adv, color=LAMBDA_COLORS[lam], lw=1.8,
                label=LAMBDA_LABELS[lam], zorder=2)

# Spread end-of-line labels to avoid overlap
_adv_finals = sorted(
    [((stoch_data[l]['cum'] - ss_cum).iloc[-1] / 1e6, l) for l in RISK_AVERSIONS]
)
_n = len(_adv_finals)
_pt_offsets = [(i - (_n - 1) / 2) * 14 for i in range(_n)]
for (_val, lam), y_off in zip(_adv_finals, _pt_offsets):
    ax_adv.annotate(f'{_val:+,.0f}M',
                    xy=(months[-1], _val),
                    xytext=(8, y_off), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color=LAMBDA_COLORS[lam],
                    ha='left', va='center', clip_on=False,
                    arrowprops=dict(arrowstyle='-', color=LAMBDA_COLORS[lam],
                                    lw=0.7, alpha=0.5))

ax_adv.axhline(0, color='#90A4AE', lw=0.8, alpha=0.5, zorder=1)
ax_adv.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+,.0f}M'))
ax_adv.set_title('Cumulative Advantage over Benchmark', fontsize=12,
                 fontweight='bold', loc='left')
ax_adv.set_ylabel('EUR', fontsize=10)
ax_adv.tick_params(axis='both', labelsize=8.5)
ax_adv.grid(True, alpha=0.12)
ax_adv.legend(fontsize=8.5, framealpha=0.85, loc='upper left',
              title='Risk Profile', title_fontsize=9)

# ═══════════════════════════════════════════════════════════════════════
# PANEL 3 — Cumulative % Uplift (middle-right)
# ═══════════════════════════════════════════════════════════════════════
ax_cpct = fig.add_subplot(gs[2, 1])
ax_cpct.set_xlim(left=months[0], right=x_right)

_pct_finals = []
for lam in RISK_AVERSIONS:
    cum_stoch = stoch_data[lam]['cum']
    cum_pct   = (cum_stoch - ss_cum) / ss_cum.abs() * 100
    cum_pct   = cum_pct.where(ss_cum.abs() > 1e4)
    ax_cpct.plot(months, cum_pct, color=LAMBDA_COLORS[lam], lw=1.8,
                 label=LAMBDA_LABELS[lam])
    final_pct = float(cum_pct.dropna().iloc[-1]) if len(cum_pct.dropna()) > 0 else 0.0
    _pct_finals.append((final_pct, lam))

# Spread end-of-line labels to avoid overlap
_pct_finals.sort(key=lambda x: x[0])
_n = len(_pct_finals)
_pt_offsets = [(i - (_n - 1) / 2) * 14 for i in range(_n)]
for (_val, lam), y_off in zip(_pct_finals, _pt_offsets):
    ax_cpct.annotate(f'{_val:+.1f}%',
                     xy=(months[-1], _val),
                     xytext=(8, y_off), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', color=LAMBDA_COLORS[lam],
                     ha='left', va='center', clip_on=False,
                     arrowprops=dict(arrowstyle='-', color=LAMBDA_COLORS[lam],
                                     lw=0.7, alpha=0.5))

ax_cpct.axhline(0, color='#90A4AE', lw=0.8, alpha=0.5)
ax_cpct.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.1f}%'))
ax_cpct.set_title('Cumulative Profit Uplift (%)', fontsize=12,
                  fontweight='bold', loc='left')
ax_cpct.tick_params(axis='both', labelsize=8.5)
ax_cpct.grid(True, alpha=0.12)
ax_cpct.legend(fontsize=8.5, framealpha=0.85, loc='upper left')

# ═══════════════════════════════════════════════════════════════════════
# PANEL 4 — Per-Window Profit Uplift, best λ only (full width)
# ═══════════════════════════════════════════════════════════════════════
ax_pct = fig.add_subplot(gs[3, :])

uplift_pct = ((wp_stoch[best_lam]['profit'] - wp_ss['profit'])
              / wp_ss['profit'].abs() * 100)
bar_colors = ['#43A047' if v >= 0 else '#E53935' for v in uplift_pct.values]
ax_pct.bar(wp_stoch[best_lam].index, uplift_pct,
           color=bar_colors, alpha=0.80, width=pd.Timedelta(days=135))
ax_pct.axhline(0, color='#546E7A', lw=0.8, alpha=0.5)

n_total = len(uplift_pct)
n_wins  = int((uplift_pct > 0).sum())
ax_pct.text(0.01, 0.95, f'Outperformed: {n_wins}/{n_total} windows',
            transform=ax_pct.transAxes, fontsize=9,
            color='#2E7D32', va='top', fontweight='bold')
ax_pct.text(0.01, 0.80, f'Underperformed: {n_total - n_wins}/{n_total} windows',
            transform=ax_pct.transAxes, fontsize=9,
            color='#C62828', va='top', fontweight='bold')

tick_dates = replan_dates_dt[::2]
ax_pct.set_xticks(tick_dates)
ax_pct.set_xticklabels([d.strftime("'%y") for d in tick_dates], fontsize=9)
ax_pct.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.0f}%'))
ax_pct.set_title(
    f'Per-Window Profit Uplift vs. Benchmark  ·  '
    f'{LAMBDA_LABELS[best_lam]} (λ={best_lam})',
    fontsize=12, fontweight='bold', loc='left')
ax_pct.set_ylabel('Uplift %', fontsize=10)
ax_pct.tick_params(axis='both', labelsize=9)
ax_pct.grid(True, alpha=0.12, axis='y')

# ── Footer ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.003,
         'Data: FRED API (WPU101704 · WPU1012 · IPG3311A2S)  ·  '
         'Model: Markov-Switching VAR (2 regimes) + Two-Stage Stochastic LP with CVaR  ·  '
         'Solver: HiGHS  ·  Scenarios: 3,000 → 300 via K-medoids',
         ha='center', va='bottom', fontsize=7.5, color='#9E9E9E', style='italic')

plt.savefig(RESULTS_DIR / 'executive_dashboard.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()
print(f"LinkedIn dashboard saved: {(RESULTS_DIR / 'executive_dashboard.png').resolve()}")
