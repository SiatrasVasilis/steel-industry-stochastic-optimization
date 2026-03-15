import sys, pickle
from pathlib import Path

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path('.').resolve()
sys.path.insert(0, str(_here.parent / 'src'))

import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from data.loader import DataLoader
from scenario import RegimeSwitchingGenerator
from optimization import StochasticOptimizationModel, SafetyStockModel, SafetyStockParams
from optimization.stochastic import ModelParameters
from params import RiskProfile

# ── Configuration ──────────────────────────────────────────────────────
BACKTEST_DATES = pd.date_range(start='2007-12', end='2023-12', freq='12MS').astype('str').tolist()
HORIZON        = 12
EXECUTE_MONTHS = 12
N_SCENARIOS    = 3_000
N_CLUSTERS     = 300
RISK_AVERSIONS = [0.0, 0.3, 0.7, 1.0]
CVAR_ALPHA     = 0.05
SEED           = 42

_start = pd.to_datetime(BACKTEST_DATES[0]).strftime('%Y%m')
_end   = pd.to_datetime(BACKTEST_DATES[-1]).strftime('%Y%m')
_freq  = f'{EXECUTE_MONTHS}M'
RESULTS_DIR = _here.parent / 'results' / f'rolling_replan_{_freq}_{_start}_{_end}'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Contract pricing as fractions of the current scrap cost (C)
CONTRACT_RATIOS = dict(
    c_fix_ratio     = 0.96,
    f_opt_ratio     = 0.04,
    basis_opt_ratio = -0.02,
    floor_opt_ratio = 0.75,
    cap_opt_ratio   = 1.50,
)

# Non-price model parameters (constant across windows)
STATIC_PARAMS = dict(
    x_fix_max     = 40_000.0,
    x_opt_max     = 20_000.0,
    Cap_base      = 42_000.0,
    Cap_flex      = 10_000.0,
    c_base        = 50.0,
    c_flex        = 80.0,
    alpha         = 1.05,
    h_rm          = 1.8,
    h_fg          = 5.0,
    pen_short_pct = 1.5,   # penalty = 150% of selling price (dynamic)
)

SS_POLICY = SafetyStockParams(
    service_level=0.95, fixed_pct=0.60,
    framework_pct=0.25, production_smoothing=True,
)

LAMBDA_LABELS = {0.0: 'Risk Neutral', 0.3: 'Moderate', 0.7: 'Conservative', 1.0: 'Full CVaR'}
LAMBDA_COLORS = {0.0: '#1565C0', 0.3: '#00897B', 0.7: '#F57F17', 1.0: '#C62828'}
SS_COLOR      = '#757575'


def _build_params(scrap_price, I_rm0, I_fg0):
    """Build ModelParameters with contract pricing relative to current scrap."""
    r = CONTRACT_RATIOS
    return ModelParameters(
        c_fix     = scrap_price * r['c_fix_ratio'],
        f_opt     = scrap_price * r['f_opt_ratio'],
        basis_opt = scrap_price * r['basis_opt_ratio'],
        floor_opt = scrap_price * r['floor_opt_ratio'],
        cap_opt   = scrap_price * r['cap_opt_ratio'],
        I_rm0     = I_rm0,
        I_fg0     = I_fg0,
        **STATIC_PARAMS,
    )


def _window_profit(bt):
    """Sum profit over the execution window."""
    return bt.timeline.iloc[:EXECUTE_MONTHS]['Profit'].sum()


# ── State initialisation ──────────────────────────────────────────────
INIT_REAL_DATA = {'P': 520, 'C': 130, 'D': 50_000}
INIT_I_RM0     = 5_000.0
INIT_I_FG0     = 5_000.0

stoch_state = {
    lam: {'I_rm0': INIT_I_RM0, 'I_fg0': INIT_I_FG0}
    for lam in RISK_AVERSIONS
}
ss_inv = {'I_rm0': INIT_I_RM0, 'I_fg0': INIT_I_FG0}
original_real_data = INIT_REAL_DATA.copy()

# Storage
all_monthly = {lam: [] for lam in RISK_AVERSIONS}
all_monthly['ss'] = []
window_profits = {lam: [] for lam in RISK_AVERSIONS}
window_profits['ss'] = []
market_history = []

# ── Download full FRED dataset once (avoid repeated API calls per window) ──
print("Downloading FRED data (once)...")
_fred_raw = DataLoader().load_from_fred(plot_data=False).get_raw_data()
print("FRED data downloaded.\n")

# ── Main backtest loop ────────────────────────────────────────────────
for date in BACKTEST_DATES:
    print(f"\n{'─'*60}\n  Replanning at  {date}\n{'─'*60}")

    loader = (
        DataLoader().load_from_dataframe(_fred_raw)
        .subset(n_observations=180, last_observation=date)
        .convert_to_real_prices(anchor_date=date, real_data=original_real_data)
        .compute_log_returns()
    )
    actual      = loader.get_future_data(last_observation=date)
    future_data = actual.rename(columns={'C': 'c_spot', 'P': 'p_sell'})

    # Record market data for dashboard
    for i in range(min(EXECUTE_MONTHS, len(actual))):
        market_history.append({
            'Date': actual.index[i],
            'P': float(actual['P'].iloc[i]),
            'C': float(actual['C'].iloc[i]),
            'D': float(actual['D'].iloc[i]),
        })

    # Scenario generation (once per window, no stress config)
    generator = RegimeSwitchingGenerator(n_regimes=2)
    generator.fit(loader)
    raw     = generator.generate(n_scenarios=N_SCENARIOS, horizon=HORIZON, seed=SEED)
    reduced = generator.reduce(raw, n_clusters=N_CLUSTERS)

    current_scrap = original_real_data['C']

    # ── Safety Stock benchmark ─────────────────────────────────────
    ss_params = _build_params(current_scrap, ss_inv['I_rm0'], ss_inv['I_fg0'])
    ss_model  = SafetyStockModel()
    ss_result = ss_model.run(
        scenarios=reduced.scenarios, prob=reduced.probabilities,
        variable_mapping={'C': 'c_spot', 'P': 'p_sell'},
        params=ss_params, policy=SS_POLICY,
    )
    ss_bt = ss_model.backtest(
        decisions=ss_result, actual_data=future_data,
        params=ss_params, policy=SS_POLICY,
    )
    win_ss = ss_bt.timeline.iloc[:EXECUTE_MONTHS].copy()
    win_ss['replan_date'] = date
    all_monthly['ss'].append(win_ss)
    window_profits['ss'].append({'replan_date': pd.to_datetime(date),
                                  'profit': _window_profit(ss_bt)})

    ss_inv['I_rm0'] = float(ss_bt.timeline['I_rm'].iloc[EXECUTE_MONTHS - 1])
    ss_inv['I_fg0'] = float(ss_bt.timeline['I_fg'].iloc[EXECUTE_MONTHS - 1])

    # ── Stochastic models (per λ) ──────────────────────────────────
    for lam in RISK_AVERSIONS:
        st = stoch_state[lam]
        params = _build_params(current_scrap, st['I_rm0'], st['I_fg0'])

        model  = StochasticOptimizationModel(solver='highs')
        result = model.run(
            scenarios=reduced.scenarios, prob=reduced.probabilities,
            variable_mapping={'C': 'c_spot', 'P': 'p_sell'},
            params=params,
            risk_profile=RiskProfile(risk_aversion=lam, cvar_alpha=CVAR_ALPHA),
        )
        bt = model.backtest(decisions=result, actual_data=future_data, params=params)

        win = bt.timeline.iloc[:EXECUTE_MONTHS].copy()
        win['replan_date'] = date
        all_monthly[lam].append(win)
        window_profits[lam].append({'replan_date': pd.to_datetime(date),
                                     'profit': _window_profit(bt)})

        st['I_rm0'] = float(bt.timeline['I_rm'].iloc[EXECUTE_MONTHS - 1])
        st['I_fg0'] = float(bt.timeline['I_fg'].iloc[EXECUTE_MONTHS - 1])

        print(f"    λ={lam:.1f}  window profit: {_window_profit(bt)/1e6:,.2f}M")

    # Update anchoring from actuals (shared across all models)
    original_real_data = {
        'P': float(actual['P'].iloc[EXECUTE_MONTHS - 1]),
        'C': float(actual['C'].iloc[EXECUTE_MONTHS - 1]),
        'D': float(actual['D'].iloc[EXECUTE_MONTHS - 1]),
    }


# ── Assemble results ──────────────────────────────────────────────────
market_df = pd.DataFrame(market_history).drop_duplicates('Date').set_index('Date').sort_index()

ss_monthly_df = pd.concat(all_monthly['ss']).sort_index()
ss_profits_s  = ss_monthly_df['Profit']
ss_cum        = ss_profits_s.cumsum()
months        = ss_profits_s.index

stoch_data = {}
for lam in RISK_AVERSIONS:
    df = pd.concat(all_monthly[lam]).sort_index()
    stoch_data[lam] = {
        'monthly': df,
        'profits': df['Profit'],
        'cum':     df['Profit'].cumsum(),
    }

wp_ss = pd.DataFrame(window_profits['ss']).set_index('replan_date')
wp_stoch = {}
for lam in RISK_AVERSIONS:
    wp_stoch[lam] = pd.DataFrame(window_profits[lam]).set_index('replan_date')

# Save results
with open(RESULTS_DIR / 'full_results_multi.pkl', 'wb') as f:
    pickle.dump({
        'stoch_data': {lam: {'monthly': d['monthly']} for lam, d in stoch_data.items()},
        'ss_monthly': ss_monthly_df,
        'market_df': market_df,
        'wp_ss': wp_ss,
        'wp_stoch': wp_stoch,
    }, f)

print(f"\nResults saved to {RESULTS_DIR.resolve()}")


# %%  ── LinkedIn Dashboard ────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

date_start      = BACKTEST_DATES[0][:7]
date_end        = BACKTEST_DATES[-1][:7]
replan_dates_dt = pd.to_datetime(BACKTEST_DATES)

best_lam     = max(RISK_AVERSIONS, key=lambda l: wp_stoch[l]['profit'].sum())
best_total   = wp_stoch[best_lam]['profit'].sum()
total_ss_val = wp_ss['profit'].sum()
best_vss     = best_total - total_ss_val
best_vss_pct = best_vss / abs(total_ss_val) * 100
best_wins    = int((wp_stoch[best_lam]['profit'] > wp_ss['profit']).sum())

ANNOTATION_EXT = 10  # months of x-axis padding for right-side value labels

_n_years = round((pd.to_datetime(BACKTEST_DATES[-1]) - pd.to_datetime(BACKTEST_DATES[0])).days / 365.25)

fig = plt.figure(figsize=(16, 13), facecolor='white')
gs  = gridspec.GridSpec(
    4, 2, figure=fig,
    height_ratios=[0.65, 0.92, 1.20, 0.78],
    hspace=0.52, wspace=0.32,
)

x_right = months[-1] + pd.DateOffset(months=ANNOTATION_EXT)

# ── Panel 0: KPI Banner (full width) ─────────────────────────────────
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

# ── Panel 1: Market Environment (full width) ─────────────────────────
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

# ── Panel 2: Cumulative EUR Advantage (middle-left) ───────────────────
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
    [(( stoch_data[l]['cum'] - ss_cum).iloc[-1] / 1e6, l) for l in RISK_AVERSIONS]
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

# ── Panel 3: Cumulative % Uplift (middle-right) ───────────────────────
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

# ── Panel 4: Per-Window Uplift, best λ only (full width) ─────────────
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

# ── Footer ────────────────────────────────────────────────────────────
fig.text(0.5, 0.003,
         'Data: FRED API (WPU101704 · WPU1012 · IPG3311A2S)  ·  '
         'Model: Markov-Switching VAR (2 regimes) + Two-Stage Stochastic LP with CVaR  ·  '
         'Solver: HiGHS  ·  Scenarios: 3,000 → 300 via K-medoids',
         ha='center', va='bottom', fontsize=7.5, color='#9E9E9E', style='italic')

plt.savefig(RESULTS_DIR / 'executive_dashboard.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()
print(f"Executive dashboard saved: {(RESULTS_DIR / 'executive_dashboard.png').resolve()}")

# %%  ── Summary Table ─────────────────────────────────────────────────
print('\n' + '=' * 90)
print('  BACKTEST SUMMARY')
print('=' * 90)

total_ss = wp_ss['profit'].sum()
for lam in RISK_AVERSIONS:
    total_stoch = wp_stoch[lam]['profit'].sum()
    vss = total_stoch - total_ss
    vss_pct = vss / abs(total_ss) * 100
    n_wins = (wp_stoch[lam]['profit'] > wp_ss['profit']).sum()
    print(f"  λ={lam:.1f} ({LAMBDA_LABELS[lam]:15s})  "
          f"Total: {total_stoch/1e6:>8,.1f}M  "
          f"VSS: {vss/1e6:>+7,.1f}M ({vss_pct:>+5.1f}%)  "
          f"Wins: {n_wins}/{len(BACKTEST_DATES)}")

print(f"\n  Safety Stock Benchmark:     Total: {total_ss/1e6:>8,.1f}M")
print('=' * 90)
