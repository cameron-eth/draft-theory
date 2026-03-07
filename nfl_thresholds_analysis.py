"""
NFL Year-1 Threshold Analysis
==============================
For WR, TE, RB, QB: what Year-1 performance thresholds predict career outcome tiers,
differentiated by draft capital (Rd 1, Rd 2-3, Rd 4-7)?

Approach:
  - Define career tiers per position based on CareerAV percentiles
  - For each draft capital tier, find Year-1 stat cutoffs that separate outcomes
  - Show probability tables: "Rd1 WR who gets X+ yards in Year 1 → P(WR1), P(WR2), ..."
"""

import pandas as pd
pd.options.future.infer_string = False

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import nflreadpy

OUT_DIR = '/Users/cam/Documents/Personal/data/'
MAX_RELIABLE_YEAR = 2014

print("=" * 70)
print("NFL YEAR-1 THRESHOLDS → CAREER OUTCOME TIERS")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. Load & link data (same pipeline as postdraft)
# ─────────────────────────────────────────────
print("\n[1] Loading data...")

combine_raw = nflreadpy.load_combine().to_pandas(use_pyarrow_extension_array=False)
combine = combine_raw.rename(columns={
    'season': 'Year', 'pfr_id': 'Pfr_ID', 'pos': 'Pos',
    'ht': 'Ht_str', 'wt': 'Wt', 'forty': 'Forty', 'bench': 'BenchReps',
    'vertical': 'Vertical', 'broad_jump': 'BroadJump', 'cone': 'Cone',
    'shuttle': 'Shuttle', 'draft_round': 'Round', 'draft_ovr': 'Pick',
    'draft_team': 'Team', 'player_name': 'Player',
})
for col in ['Wt', 'Forty', 'Round', 'Pick']:
    combine[col] = pd.to_numeric(combine[col], errors='coerce')

draft_raw = nflreadpy.load_draft_picks().to_pandas(use_pyarrow_extension_array=False)
draft_2000 = draft_raw[draft_raw['season'] >= 2000].copy()
valid_draft = draft_2000[draft_2000['pfr_player_id'].notna() & draft_2000['gsis_id'].notna()]
pfr_to_gsis = dict(zip(valid_draft['pfr_player_id'], valid_draft['gsis_id']))
pfr_to_wav = dict(zip(valid_draft['pfr_player_id'], valid_draft['w_av']))

combine['gsis_id'] = combine['Pfr_ID'].map(pfr_to_gsis)
combine['CareerAV'] = combine['Pfr_ID'].map(pfr_to_wav).fillna(0.0)

drafted = combine[(combine['Round'].notna()) & (combine['Year'] <= MAX_RELIABLE_YEAR)].copy()
drafted = drafted[drafted['gsis_id'].notna()].copy()

# Draft capital tiers
def draft_tier(rnd):
    if rnd == 1: return 'Rd 1'
    elif rnd <= 3: return 'Rd 2-3'
    else: return 'Rd 4-7'
drafted['DraftTier'] = drafted['Round'].apply(draft_tier)

print(f"    Drafted with gsis_id (2000-{MAX_RELIABLE_YEAR}): {len(drafted):,}")

# ─────────────────────────────────────────────
# 2. Load Year-1 stats
# ─────────────────────────────────────────────
print("\n[2] Loading seasonal stats...")
all_seasons = list(range(2000, 2018))
stats_all = nflreadpy.load_player_stats(seasons=all_seasons, summary_level='reg').to_pandas(use_pyarrow_extension_array=False)

STAT_COLS = ['games', 'passing_yards', 'passing_tds', 'passing_interceptions',
             'passing_epa', 'rushing_yards', 'rushing_tds', 'rushing_epa',
             'receiving_yards', 'receiving_tds', 'receptions', 'receiving_epa',
             'targets', 'fantasy_points_ppr', 'def_sacks', 'def_tackles_solo',
             'def_interceptions', 'completions', 'attempts']

for col in STAT_COLS:
    if col in stats_all.columns:
        stats_all[col] = pd.to_numeric(stats_all[col], errors='coerce').fillna(0)

# Build Year-1 stats for each player
print("    Computing Year-1 stats...")
records = []
for _, row in drafted.iterrows():
    gsis = row['gsis_id']
    draft_yr = row['Year']
    ps = stats_all[(stats_all['player_id'] == gsis) & (stats_all['season'] == draft_yr)]
    rec = {'gsis_id': gsis}
    for col in STAT_COLS:
        rec[f'Y1_{col}'] = ps[col].sum() if len(ps) > 0 else 0
    records.append(rec)

y1_df = pd.DataFrame(records)
drafted = drafted.merge(y1_df, on='gsis_id', how='left')
print(f"    Players with Year-1 game data: {(drafted['Y1_games'] > 0).sum():,}")

# ─────────────────────────────────────────────
# 3. Define position groups and career tiers
# ─────────────────────────────────────────────
POS_MAP = {
    'QB': 'QB', 'RB': 'RB', 'FB': 'RB',
    'WR': 'WR', 'TE': 'TE',
}
drafted['PosGroup'] = drafted['Pos'].map(POS_MAP)
skill = drafted[drafted['PosGroup'].isin(['WR', 'TE', 'RB', 'QB'])].copy()

# Career tiers defined by within-position AV percentiles
# Elite (top 15%), Starter (15-40%), Rotation/backup (40-70%), Bust (bottom 30%)
TIER_NAMES = ['Elite', 'Starter', 'Backup', 'Bust']
TIER_COLORS = ['#1a9850', '#91cf60', '#fee08b', '#d73027']

def assign_tier(row, thresholds):
    av = row['CareerAV']
    if av >= thresholds[0]: return 'Elite'
    elif av >= thresholds[1]: return 'Starter'
    elif av >= thresholds[2]: return 'Backup'
    else: return 'Bust'

print("\n[3] Career tier thresholds (AV):")
tier_thresholds = {}
for pos in ['WR', 'RB', 'TE', 'QB']:
    grp = skill[skill['PosGroup'] == pos]['CareerAV']
    p85, p60, p30 = grp.quantile(0.85), grp.quantile(0.60), grp.quantile(0.30)
    tier_thresholds[pos] = (p85, p60, p30)
    print(f"    {pos}: Elite ≥ {p85:.0f}  |  Starter ≥ {p60:.0f}  |  Backup ≥ {p30:.0f}  |  Bust < {p30:.0f}")

for pos in ['WR', 'RB', 'TE', 'QB']:
    mask = skill['PosGroup'] == pos
    skill.loc[mask, 'CareerTier'] = skill.loc[mask].apply(
        lambda r: assign_tier(r, tier_thresholds[pos]), axis=1
    )

# ─────────────────────────────────────────────
# 4. Position-specific configs
# ─────────────────────────────────────────────
POS_CONFIG = {
    'WR': {
        'primary_stat': 'Y1_receiving_yards',
        'stat_label': 'Year-1 Receiving Yards',
        'thresholds': [100, 250, 400, 600, 800],
        'secondary_stats': ['Y1_receptions', 'Y1_receiving_tds', 'Y1_games', 'Y1_targets'],
    },
    'TE': {
        'primary_stat': 'Y1_receiving_yards',
        'stat_label': 'Year-1 Receiving Yards',
        'thresholds': [50, 150, 300, 450, 600],
        'secondary_stats': ['Y1_receptions', 'Y1_receiving_tds', 'Y1_games'],
    },
    'RB': {
        'primary_stat': 'Y1_rushing_yards',
        'stat_label': 'Year-1 Rushing Yards',
        'thresholds': [50, 200, 400, 600, 800],
        'secondary_stats': ['Y1_rushing_tds', 'Y1_games', 'Y1_receptions'],
    },
    'QB': {
        'primary_stat': 'Y1_passing_yards',
        'stat_label': 'Year-1 Passing Yards',
        'thresholds': [0, 500, 1500, 2500, 3500],
        'secondary_stats': ['Y1_passing_tds', 'Y1_passing_interceptions', 'Y1_games', 'Y1_completions'],
    },
}

DRAFT_TIERS = ['Rd 1', 'Rd 2-3', 'Rd 4-7']

# ─────────────────────────────────────────────
# 5. Build threshold tables
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("YEAR-1 THRESHOLD → CAREER OUTCOME TABLES")
print("=" * 70)

all_tables = {}

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    primary = cfg['primary_stat']
    thresholds = cfg['thresholds']

    print(f"\n{'─' * 70}")
    print(f"  {pos} — Primary metric: {cfg['stat_label']}")
    print(f"{'─' * 70}")

    pos_tables = {}

    for dt in DRAFT_TIERS:
        grp = pos_df[pos_df['DraftTier'] == dt].copy()
        if len(grp) < 10:
            continue

        print(f"\n  {pos} | {dt}  (n={len(grp)})")
        print(f"  {'Y1 Stat Range':>20s}  {'n':>4s}  {'Avg AV':>7s}  {'Med AV':>7s}  {'%Elite':>7s}  {'%Start':>7s}  {'%Backup':>7s}  {'%Bust':>7s}")

        table_rows = []

        # Define bins from thresholds
        bins = [(-1, thresholds[0])] + \
               [(thresholds[i], thresholds[i+1]) for i in range(len(thresholds)-1)] + \
               [(thresholds[-1], 99999)]

        for lo, hi in bins:
            subset = grp[(grp[primary] > lo) & (grp[primary] <= hi)]
            if len(subset) < 3:
                continue

            label = f"{lo+1}-{hi}" if hi < 99999 else f"{lo+1}+"
            n = len(subset)
            avg_av = subset['CareerAV'].mean()
            med_av = subset['CareerAV'].median()

            tier_pcts = {}
            for tier in TIER_NAMES:
                tier_pcts[tier] = (subset['CareerTier'] == tier).mean() * 100

            print(f"  {label:>20s}  {n:4d}  {avg_av:7.1f}  {med_av:7.1f}  "
                  f"{tier_pcts['Elite']:6.0f}%  {tier_pcts['Starter']:6.0f}%  "
                  f"{tier_pcts['Backup']:6.0f}%  {tier_pcts['Bust']:6.0f}%")

            table_rows.append({
                'range': label, 'n': n, 'avg_av': avg_av, 'med_av': med_av,
                **{f'pct_{t}': tier_pcts[t] for t in TIER_NAMES}
            })

        # Overall for this draft tier
        n = len(grp)
        avg_av = grp['CareerAV'].mean()
        med_av = grp['CareerAV'].median()
        tier_pcts = {t: (grp['CareerTier'] == t).mean() * 100 for t in TIER_NAMES}
        print(f"  {'OVERALL':>20s}  {n:4d}  {avg_av:7.1f}  {med_av:7.1f}  "
              f"{tier_pcts['Elite']:6.0f}%  {tier_pcts['Starter']:6.0f}%  "
              f"{tier_pcts['Backup']:6.0f}%  {tier_pcts['Bust']:6.0f}%")

        pos_tables[dt] = table_rows

    all_tables[pos] = pos_tables

    # Secondary stat correlations for this position
    print(f"\n  {pos} secondary stat correlations with Career AV (all draft tiers):")
    for stat in cfg['secondary_stats']:
        sub = pos_df[[stat, 'CareerAV']].dropna().astype(float)
        if len(sub) > 20:
            r, p = stats.pearsonr(sub[stat], sub['CareerAV'])
            print(f"    {stat:30s}: r={r:.3f}  p={p:.4f}")

# ─────────────────────────────────────────────
# 6. Key threshold callouts
# ─────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("KEY THRESHOLD FINDINGS")
print(f"{'=' * 70}")

# Compute specific threshold hit rates
for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    primary = cfg['primary_stat']

    print(f"\n  {pos}:")

    for dt in DRAFT_TIERS:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 10:
            continue

        # Find the "breakout" threshold: Y1 stat level where >50% become Elite or Starter
        for thresh in cfg['thresholds']:
            above = grp[grp[primary] > thresh]
            below = grp[grp[primary] <= thresh]
            if len(above) >= 3 and len(below) >= 3:
                above_hit = ((above['CareerTier'] == 'Elite') | (above['CareerTier'] == 'Starter')).mean()
                below_hit = ((below['CareerTier'] == 'Elite') | (below['CareerTier'] == 'Starter')).mean()
                above_avg = above['CareerAV'].mean()
                below_avg = below['CareerAV'].mean()

                if above_hit >= 0.45 and above_hit > below_hit + 0.10:
                    print(f"    {dt}: >{thresh:,} {cfg['stat_label'].split('Year-1 ')[1]} → "
                          f"{above_hit*100:.0f}% hit (Elite+Starter), avg AV={above_avg:.0f}  "
                          f"(vs {below_hit*100:.0f}% / AV={below_avg:.0f} below)  "
                          f"[n={len(above)} above, {len(below)} below]")

# ─────────────────────────────────────────────
# 7. Games-played threshold (universal)
# ─────────────────────────────────────────────
print(f"\n  UNIVERSAL — Year-1 Games Played threshold:")
for pos in ['WR', 'RB', 'TE', 'QB']:
    pos_df = skill[skill['PosGroup'] == pos]
    for dt in DRAFT_TIERS:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 10:
            continue
        played_10 = grp[grp['Y1_games'] >= 10]
        played_lt10 = grp[grp['Y1_games'] < 10]
        if len(played_10) >= 3 and len(played_lt10) >= 3:
            hit_10 = ((played_10['CareerTier'] == 'Elite') | (played_10['CareerTier'] == 'Starter')).mean()
            hit_lt10 = ((played_lt10['CareerTier'] == 'Elite') | (played_lt10['CareerTier'] == 'Starter')).mean()
            print(f"    {pos} {dt}: ≥10 games → {hit_10*100:.0f}% hit (n={len(played_10)})  |  "
                  f"<10 games → {hit_lt10*100:.0f}% hit (n={len(played_lt10)})")

# ─────────────────────────────────────────────
# 8. Plots
# ─────────────────────────────────────────────
print("\n[4] Generating plots...")
sns.set_theme(style='whitegrid', palette='muted', font_scale=0.9)

fig, axes = plt.subplots(4, 3, figsize=(22, 24))
fig.subplots_adjust(hspace=0.55, wspace=0.30)

for row_idx, pos in enumerate(['WR', 'RB', 'TE', 'QB']):
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    primary = cfg['primary_stat']

    for col_idx, dt in enumerate(DRAFT_TIERS):
        ax = axes[row_idx, col_idx]
        grp = pos_df[pos_df['DraftTier'] == dt].copy()

        if len(grp) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Scatter: Y1 stat vs CareerAV, colored by tier
        for tier, color in zip(TIER_NAMES, TIER_COLORS):
            tier_data = grp[grp['CareerTier'] == tier]
            ax.scatter(tier_data[primary], tier_data['CareerAV'],
                      alpha=0.5, s=25, color=color, label=tier, edgecolors='white', linewidths=0.3)

        # Trend line
        valid = grp[[primary, 'CareerAV']].dropna().astype(float)
        if len(valid) > 10:
            m, b = np.polyfit(valid[primary], valid['CareerAV'], 1)
            x_line = np.linspace(valid[primary].min(), valid[primary].max(), 100)
            ax.plot(x_line, m*x_line + b, color='black', lw=1.5, linestyle='--', alpha=0.7)

            r, _ = stats.pearsonr(valid[primary], valid['CareerAV'])
            ax.text(0.95, 0.95, f'r={r:.2f}\nn={len(grp)}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Threshold lines
        for thresh in cfg['thresholds'][1:-1]:
            ax.axvline(thresh, color='gray', lw=0.7, linestyle=':', alpha=0.5)

        ax.set_xlabel(cfg['stat_label'] if row_idx == 3 else '')
        ax.set_ylabel('Career AV' if col_idx == 0 else '')
        ax.set_title(f'{pos} — {dt}')

        if row_idx == 0 and col_idx == 2:
            ax.legend(fontsize=7, loc='upper left')

plt.suptitle(f'Year-1 Production → Career AV by Position & Draft Capital\n(2000-{MAX_RELIABLE_YEAR} draft classes)',
             fontsize=15, fontweight='bold', y=1.01)

out_path = OUT_DIR + 'nfl_thresholds_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"    Saved: {out_path}")

# ─────────────────────────────────────────────
# 9. Stacked bar chart: outcome probabilities
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))
fig2.subplots_adjust(hspace=0.45, wspace=0.30)

for idx, pos in enumerate(['WR', 'RB', 'TE', 'QB']):
    ax = axes2[idx // 2, idx % 2]
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    primary = cfg['primary_stat']

    # Build bars for each draft tier + Y1 performance bucket
    bar_data = []
    bar_labels = []
    for dt in DRAFT_TIERS:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 8:
            continue

        # Split into low/med/high Y1 production within this draft tier
        t_lo = cfg['thresholds'][1]
        t_hi = cfg['thresholds'][3] if len(cfg['thresholds']) > 3 else cfg['thresholds'][-1]

        for label_suffix, mask in [
            (f'<{t_lo}', grp[primary] <= t_lo),
            (f'{t_lo}-{t_hi}', (grp[primary] > t_lo) & (grp[primary] <= t_hi)),
            (f'>{t_hi}', grp[primary] > t_hi),
        ]:
            sub = grp[mask]
            if len(sub) < 3:
                continue
            pcts = [(sub['CareerTier'] == t).mean() * 100 for t in TIER_NAMES]
            bar_data.append(pcts)
            bar_labels.append(f'{dt}\n{label_suffix}')

    if not bar_data:
        continue

    bar_data = np.array(bar_data)
    x = np.arange(len(bar_labels))
    bottom = np.zeros(len(bar_labels))

    for i, (tier, color) in enumerate(zip(TIER_NAMES, TIER_COLORS)):
        vals = bar_data[:, i]
        bars = ax.bar(x, vals, bottom=bottom, color=color, edgecolor='white',
                      linewidth=0.5, label=tier, width=0.7)
        # Label significant segments
        for j, v in enumerate(vals):
            if v >= 12:
                ax.text(x[j], bottom[j] + v/2, f'{v:.0f}%',
                       ha='center', va='center', fontsize=7, fontweight='bold')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=0)
    ax.set_ylabel('% of Players')
    ax.set_title(f'{pos}: Career Outcome by Draft Tier + Year-1 {cfg["stat_label"].split("Year-1 ")[1]}')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc='upper right')

plt.suptitle(f'Career Outcome Probabilities: Draft Capital × Year-1 Production\n(2000-{MAX_RELIABLE_YEAR})',
             fontsize=14, fontweight='bold', y=1.01)

out_path2 = OUT_DIR + 'nfl_thresholds_stacked.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"    Saved: {out_path2}")

print("\nDone.")
