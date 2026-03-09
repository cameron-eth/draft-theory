"""
NFL Peak Outcome Tiers — EXPANDED EDITION
==========================================
Absolute stat floor tier definitions:

  WR1: 1000+ rec yds AND (60+ catches OR 6+ rec TDs)
  WR2: 600+ rec yds AND (40+ catches OR 4+ rec TDs)
  WR3: 300+ rec yds
  Depth/Out: below WR3

  RB1: 1000+ rush yds AND (60+ touches OR 6+ rush TDs)
  RB2: 600+ rush yds AND (4+ rush TDs OR 150+ carries)
  RB3: 300+ rush yds
  Depth/Out: below RB3

  TE1: 700+ rec yds AND (50+ catches OR 5+ rec TDs)
  TE2: 400+ rec yds AND (30+ catches OR 3+ rec TDs)
  Depth/Out: below TE2

  QB1: 4000+ pass yds AND 25+ pass TDs
  QB2: 3000+ pass yds AND 15+ pass TDs
  Backup/Out: below QB2

Draft classes 2000-2021, stats through 2025.
"""

import pandas as pd
pd.options.future.infer_string = False

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import nflreadpy

OUT_DIR = '/Users/cam/Documents/Personal/data/'
MAX_DRAFT_YEAR = 2021   # give at least 4 seasons of career data
MAX_STAT_YEAR  = 2025

print("=" * 90)
print("PEAK OUTCOME TIERS — EXPANDED (absolute stat floors)")
print(f"Draft classes 2000-{MAX_DRAFT_YEAR}, stats through {MAX_STAT_YEAR}")
print("=" * 90)

# ═══════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════
print("\n[1] Loading data...")
combine_raw = nflreadpy.load_combine().to_pandas(use_pyarrow_extension_array=False)
combine = combine_raw.rename(columns={
    'season': 'Year', 'pfr_id': 'Pfr_ID', 'pos': 'Pos',
    'ht': 'Ht', 'wt': 'Wt', 'forty': 'Forty', 'bench': 'BenchReps',
    'vertical': 'Vertical', 'broad_jump': 'BroadJump', 'cone': 'Cone',
    'shuttle': 'Shuttle', 'draft_round': 'Round', 'draft_ovr': 'Pick',
    'player_name': 'Player',
})
for col in ['Round', 'Pick', 'Wt', 'Forty']:
    combine[col] = pd.to_numeric(combine[col], errors='coerce')

draft_raw = nflreadpy.load_draft_picks().to_pandas(use_pyarrow_extension_array=False)
draft_2000 = draft_raw[draft_raw['season'] >= 2000].copy()
valid_draft = draft_2000[draft_2000['pfr_player_id'].notna() & draft_2000['gsis_id'].notna()]
pfr_to_gsis = dict(zip(valid_draft['pfr_player_id'], valid_draft['gsis_id']))

combine['gsis_id'] = combine['Pfr_ID'].map(pfr_to_gsis)

drafted = combine[(combine['Round'].notna()) & (combine['Year'] >= 2000) & (combine['Year'] <= MAX_DRAFT_YEAR)].copy()
drafted = drafted[drafted['gsis_id'].notna()].copy()

# ── Draft tier functions ──
def draft_tier_3(rnd):
    if rnd == 1: return 'Rd 1'
    elif rnd <= 3: return 'Rd 2-3'
    else: return 'Rd 4+'

def draft_tier_fine(rnd, pick):
    if pd.notna(pick) and pick <= 10: return 'Top 10'
    elif rnd == 1: return 'Rd 1 (11-32)'
    elif rnd == 2: return 'Rd 2'
    elif rnd == 3: return 'Rd 3'
    elif rnd <= 5: return 'Rd 4-5'
    else: return 'Rd 6-7'

drafted['DraftTier'] = drafted['Round'].apply(draft_tier_3)
drafted['DraftTierFine'] = drafted.apply(lambda r: draft_tier_fine(r['Round'], r['Pick']), axis=1)

POS_MAP = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}
drafted['PosGroup'] = drafted['Pos'].map(POS_MAP)
skill = drafted[drafted['PosGroup'].isin(['WR', 'TE', 'RB', 'QB'])].copy()
print(f"    Skill position players (2000-{MAX_DRAFT_YEAR}): {len(skill):,}")

# ── Speed tier for WR/RB ──
def speed_tier(forty):
    if pd.isna(forty): return 'No 40'
    elif forty <= 4.39: return '≤4.39 (Elite)'
    elif forty <= 4.49: return '4.40-4.49 (Fast)'
    elif forty <= 4.59: return '4.50-4.59 (Avg)'
    else: return '4.60+ (Slow)'

skill['SpeedTier'] = skill['Forty'].apply(speed_tier)

# ── Size tier ──
def size_tier_wr(wt):
    if pd.isna(wt): return 'No Wt'
    elif wt >= 215: return '215+ (Big)'
    elif wt >= 200: return '200-214'
    else: return '<200 (Small)'

def size_tier_rb(wt):
    if pd.isna(wt): return 'No Wt'
    elif wt >= 225: return '225+ (Big)'
    elif wt >= 210: return '210-224'
    else: return '<210 (Small)'

skill.loc[skill['PosGroup'] == 'WR', 'SizeTier'] = skill.loc[skill['PosGroup'] == 'WR', 'Wt'].apply(size_tier_wr)
skill.loc[skill['PosGroup'] == 'RB', 'SizeTier'] = skill.loc[skill['PosGroup'] == 'RB', 'Wt'].apply(size_tier_rb)

# ═══════════════════════════════════════════════
# 2. LOAD ALL SEASONAL STATS
# ═══════════════════════════════════════════════
print("\n[2] Loading seasonal stats...")
all_seasons = list(range(2000, MAX_STAT_YEAR + 1))
stats_all = nflreadpy.load_player_stats(seasons=all_seasons, summary_level='reg').to_pandas(use_pyarrow_extension_array=False)

STAT_COLS = ['games', 'passing_yards', 'passing_tds', 'passing_interceptions',
             'rushing_yards', 'rushing_tds', 'carries',
             'receiving_yards', 'receiving_tds', 'receptions', 'targets',
             'fantasy_points_ppr', 'completions', 'attempts']

for col in STAT_COLS:
    if col in stats_all.columns:
        stats_all[col] = pd.to_numeric(stats_all[col], errors='coerce').fillna(0)

gsis_to_pos = dict(zip(skill['gsis_id'], skill['PosGroup']))
stats_all['PosGroup'] = stats_all['player_id'].map(gsis_to_pos)
stats_skill = stats_all[stats_all['PosGroup'].notna()].copy()

for col in ['receptions', 'receiving_tds', 'rushing_tds', 'carries', 'passing_tds']:
    if col not in stats_skill.columns:
        stats_skill[col] = 0
    else:
        stats_skill[col] = pd.to_numeric(stats_skill[col], errors='coerce').fillna(0)

print(f"    Total player-seasons: {len(stats_skill):,}")

# ═══════════════════════════════════════════════
# 3. CLASSIFY SEASONS → PEAK TIER
# ═══════════════════════════════════════════════
print("\n[3] Classifying peak tiers...")

def classify_season_tier(row):
    pos = row['PosGroup']
    rec_yds = row.get('receiving_yards', 0) or 0
    rush_yds = row.get('rushing_yards', 0) or 0
    pass_yds = row.get('passing_yards', 0) or 0
    rec = row.get('receptions', 0) or 0
    rec_tds = row.get('receiving_tds', 0) or 0
    rush_tds = row.get('rushing_tds', 0) or 0
    pass_tds = row.get('passing_tds', 0) or 0
    rush_att = row.get('carries', 0) or 0
    touches = rush_att + rec

    if pos == 'WR':
        if rec_yds >= 1000 and (rec >= 60 or rec_tds >= 6): return 'WR1'
        elif rec_yds >= 600 and (rec >= 40 or rec_tds >= 4): return 'WR2'
        elif rec_yds >= 300: return 'WR3'
        else: return 'Depth/Out'
    elif pos == 'RB':
        if rush_yds >= 1000 and (touches >= 60 or rush_tds >= 6): return 'RB1'
        elif rush_yds >= 600 and (rush_tds >= 4 or rush_att >= 150): return 'RB2'
        elif rush_yds >= 300: return 'RB3'
        else: return 'Depth/Out'
    elif pos == 'TE':
        if rec_yds >= 700 and (rec >= 50 or rec_tds >= 5): return 'TE1'
        elif rec_yds >= 400 and (rec >= 30 or rec_tds >= 3): return 'TE2'
        else: return 'Depth/Out'
    elif pos == 'QB':
        if pass_yds >= 4000 and pass_tds >= 25: return 'QB1'
        elif pass_yds >= 3000 and pass_tds >= 15: return 'QB2'
        else: return 'Backup/Out'
    return 'Depth/Out'

TIER_ORDER = {
    'WR1': 1, 'WR2': 2, 'WR3': 3,
    'RB1': 1, 'RB2': 2, 'RB3': 3,
    'TE1': 1, 'TE2': 2,
    'QB1': 1, 'QB2': 2,
    'Depth/Out': 99, 'Backup/Out': 99,
}

stats_skill['season_tier'] = stats_skill.apply(classify_season_tier, axis=1)
stats_skill['tier_order'] = stats_skill['season_tier'].map(TIER_ORDER)

best_season_idx = stats_skill.groupby('player_id')['tier_order'].idxmin()
peak_seasons = stats_skill.loc[best_season_idx].copy()

peak_df = peak_seasons[['player_id', 'season', 'season_tier',
                         'receiving_yards', 'rushing_yards', 'passing_yards',
                         'receptions', 'receiving_tds', 'rushing_tds', 'passing_tds',
                         'fantasy_points_ppr', 'games']].copy()
peak_df = peak_df.rename(columns={
    'player_id': 'gsis_id', 'season_tier': 'PeakTier', 'season': 'peak_season',
    'receiving_yards': 'peak_rec_yds', 'rushing_yards': 'peak_rush_yds',
    'passing_yards': 'peak_pass_yds', 'receptions': 'peak_rec',
    'receiving_tds': 'peak_rec_tds', 'rushing_tds': 'peak_rush_tds',
    'passing_tds': 'peak_pass_tds', 'fantasy_points_ppr': 'peak_fpppr',
    'games': 'peak_games',
})

seasons_played = stats_skill.groupby('player_id')['season'].nunique().reset_index()
seasons_played.columns = ['gsis_id', 'total_seasons']
peak_df = peak_df.merge(seasons_played, on='gsis_id', how='left')

# Count how many tier-1 seasons each player had
tier1_counts = stats_skill[stats_skill['tier_order'] == 1].groupby('player_id').size().reset_index(name='tier1_seasons')
tier1_counts = tier1_counts.rename(columns={'player_id': 'gsis_id'})
peak_df = peak_df.merge(tier1_counts, on='gsis_id', how='left')
peak_df['tier1_seasons'] = peak_df['tier1_seasons'].fillna(0).astype(int)

skill = skill.merge(peak_df, on='gsis_id', how='left')
skill['PeakTier'] = skill['PeakTier'].fillna(
    skill['PosGroup'].apply(lambda p: 'Backup/Out' if p == 'QB' else 'Depth/Out')
)
skill['tier1_seasons'] = skill['tier1_seasons'].fillna(0).astype(int)
print(f"    Players with peak tier data: {len(peak_df):,}")

# ═══════════════════════════════════════════════
# 4. LOAD YEAR-1 AND YEAR-2 STATS
# ═══════════════════════════════════════════════
print("\n[4] Computing Year-1 and Year-2 stats...")

# Build lookup dict for speed: (player_id, season) → stat dict
stats_lookup = {}
for _, row in stats_all[stats_all['PosGroup'].notna()].iterrows():
    key = (row['player_id'], row['season'])
    stats_lookup[key] = {col: row[col] for col in STAT_COLS if col in stats_all.columns}

y_records = []
for _, row in skill.iterrows():
    gsis = row['gsis_id']
    draft_yr = row['Year']
    rec = {'gsis_id': gsis}

    # Year 1
    y1 = stats_lookup.get((gsis, draft_yr), {})
    for col in STAT_COLS:
        rec[f'Y1_{col}'] = y1.get(col, 0)

    # Year 2
    y2 = stats_lookup.get((gsis, draft_yr + 1), {})
    for col in STAT_COLS:
        rec[f'Y2_{col}'] = y2.get(col, 0)

    y_records.append(rec)

y_df = pd.DataFrame(y_records)
skill = skill.merge(y_df, on='gsis_id', how='left')

# ═══════════════════════════════════════════════
# 5. TIER DISTRIBUTION OVERVIEW
# ═══════════════════════════════════════════════
print("\n" + "=" * 90)
print("TIER DISTRIBUTION BY POSITION")
print("=" * 90)

for pos in ['WR', 'RB', 'TE', 'QB']:
    grp = skill[skill['PosGroup'] == pos]
    print(f"\n  {pos} (n={len(grp)}):")
    tier_counts = grp['PeakTier'].value_counts()
    for tier, count in sorted(tier_counts.items(), key=lambda x: TIER_ORDER.get(x[0], 99)):
        pct = count / len(grp) * 100
        print(f"    {tier:12s}: {count:4d}  ({pct:5.1f}%)")

# ═══════════════════════════════════════════════
# 6. HELPER: print a table block
# ═══════════════════════════════════════════════
def print_table(pos_df, tiers, stat_col, stat_label, bins, bin_labels, group_label, min_n=3):
    """Print one tier-likelihood table and return rows for plotting."""
    n_tiers = len(tiers)
    tier_header = '  '.join(f'{t:>10s}' for t in tiers) + '   Hit Rate'
    print(f"\n  {group_label}  (n={len(pos_df)})")
    print(f"  {stat_label:>22s}  {'n':>5s}  {tier_header}")
    print(f"  {'─' * 22}  {'─' * 5}  {'─' * (12 * n_tiers + 12)}")

    plot_rows = []
    for (lo, hi), label in zip(bins, bin_labels):
        subset = pos_df[(pos_df[stat_col] > lo) & (pos_df[stat_col] <= hi)]
        if len(subset) < min_n:
            continue
        n = len(subset)
        tier_pcts = []
        for tier in tiers:
            pct = (subset['PeakTier'] == tier).mean() * 100
            tier_pcts.append(pct)
        # Hit rate = tier1 + tier2 combined
        hit_rate = sum(tier_pcts[:2])
        tier_str = '  '.join(f'{p:9.0f}%' for p in tier_pcts)
        print(f"  {label:>22s}  {n:5d}  {tier_str}  {hit_rate:8.0f}%")
        plot_rows.append({'label': label, 'n': n, 'tiers': tier_pcts, 'hit_rate': hit_rate})

    # Overall
    n = len(pos_df)
    tier_pcts = [(pos_df['PeakTier'] == t).mean() * 100 for t in tiers]
    hit_rate = sum(tier_pcts[:2])
    tier_str = '  '.join(f'{p:9.0f}%' for p in tier_pcts)
    print(f"  {'─' * 22}  {'─' * 5}  {'─' * (12 * n_tiers + 12)}")
    print(f"  {'OVERALL':>22s}  {n:5d}  {tier_str}  {hit_rate:8.0f}%")

    return plot_rows


def print_examples(pos_df, tiers, n_examples=4):
    """Print named player examples for each tier."""
    for tier in tiers:
        t_df = pos_df[pos_df['PeakTier'] == tier]
        # Sort by peak stat (descending) for best examples
        if 'peak_rec_yds' in t_df.columns:
            t_df = t_df.sort_values('peak_rec_yds', ascending=False)
        elif 'peak_rush_yds' in t_df.columns:
            t_df = t_df.sort_values('peak_rush_yds', ascending=False)
        elif 'peak_pass_yds' in t_df.columns:
            t_df = t_df.sort_values('peak_pass_yds', ascending=False)
        examples = t_df.head(n_examples)
        names = ', '.join(
            f"{r['Player']} ('{int(r['Year']) % 100} R{int(r['Round'])})"
            for _, r in examples.iterrows()
        )
        print(f"    {tier:12s}: {names}")


# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
POS_CONFIG = {
    'WR': {
        'y1_stat': 'Y1_receiving_yards', 'y2_stat': 'Y2_receiving_yards',
        'label': 'Rec Yards',
        'bins': [(-1, 100), (100, 300), (300, 500), (500, 750), (750, 99999)],
        'bin_labels': ['0-100', '101-300', '301-500', '501-750', '750+'],
        'tiers': ['WR1', 'WR2', 'WR3', 'Depth/Out'],
    },
    'RB': {
        'y1_stat': 'Y1_rushing_yards', 'y2_stat': 'Y2_rushing_yards',
        'label': 'Rush Yards',
        'bins': [(-1, 100), (100, 350), (350, 600), (600, 900), (900, 99999)],
        'bin_labels': ['0-100', '101-350', '351-600', '601-900', '900+'],
        'tiers': ['RB1', 'RB2', 'RB3', 'Depth/Out'],
    },
    'TE': {
        'y1_stat': 'Y1_receiving_yards', 'y2_stat': 'Y2_receiving_yards',
        'label': 'Rec Yards',
        'bins': [(-1, 75), (75, 200), (200, 400), (400, 99999)],
        'bin_labels': ['0-75', '76-200', '201-400', '400+'],
        'tiers': ['TE1', 'TE2', 'Depth/Out'],
    },
    'QB': {
        'y1_stat': 'Y1_passing_yards', 'y2_stat': 'Y2_passing_yards',
        'label': 'Pass Yards',
        'bins': [(-1, 0), (0, 1000), (1000, 2500), (2500, 3500), (3500, 99999)],
        'bin_labels': ['Sat (0)', '1-1000', '1001-2500', '2501-3500', '3500+'],
        'tiers': ['QB1', 'QB2', 'Backup/Out'],
    },
}

DRAFT_TIERS_3 = ['Rd 1', 'Rd 2-3', 'Rd 4+']
DRAFT_TIERS_FINE = ['Top 10', 'Rd 1 (11-32)', 'Rd 2', 'Rd 3', 'Rd 4-5', 'Rd 6-7']

# ═══════════════════════════════════════════════
# 7. FULL TABLES — YEAR 1, by 3-tier draft capital
# ═══════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION A: YEAR-1 PRODUCTION × DRAFT CAPITAL (3 tiers)")
print("Hit Rate = Tier1 + Tier2 combined")
print("=" * 90)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    tiers = cfg['tiers']

    print(f"\n{'━' * 90}")
    if pos == 'WR':
        print(f"  WR  |  WR1: 1000+ yds & (60+ rec or 6+ TDs)  |  WR2: 600+ yds & (40+ rec or 4+ TDs)  |  WR3: 300+ yds")
    elif pos == 'RB':
        print(f"  RB  |  RB1: 1000+ yds & (60+ touches or 6+ TDs)  |  RB2: 600+ yds & (4+ TDs or 150+ car)  |  RB3: 300+ yds")
    elif pos == 'TE':
        print(f"  TE  |  TE1: 700+ yds & (50+ rec or 5+ TDs)  |  TE2: 400+ yds & (30+ rec or 3+ TDs)")
    elif pos == 'QB':
        print(f"  QB  |  QB1: 4000+ yds & 25+ TDs  |  QB2: 3000+ yds & 15+ TDs")
    print(f"{'━' * 90}")

    for dt in DRAFT_TIERS_3:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 5: continue
        print_table(grp, tiers, cfg['y1_stat'], f'Rookie {cfg["label"]}',
                    cfg['bins'], cfg['bin_labels'], f'{pos} | {dt}')

    # All rounds
    print_table(pos_df, tiers, cfg['y1_stat'], f'Rookie {cfg["label"]}',
                cfg['bins'], cfg['bin_labels'], f'{pos} | ALL ROUNDS')

    # Examples
    print(f"\n  Example players:")
    print_examples(pos_df, tiers)

# ═══════════════════════════════════════════════
# 8. FINE-GRAINED DRAFT SPLITS — YEAR 1
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION B: YEAR-1 PRODUCTION × DRAFT CAPITAL (6 tiers: Top 10 / Rd1 11-32 / Rd2 / Rd3 / Rd4-5 / Rd6-7)")
print("=" * 90)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    tiers = cfg['tiers']

    print(f"\n{'━' * 90}")
    print(f"  {pos}")
    print(f"{'━' * 90}")

    for dt in DRAFT_TIERS_FINE:
        grp = pos_df[pos_df['DraftTierFine'] == dt]
        if len(grp) < 8: continue
        print_table(grp, tiers, cfg['y1_stat'], f'Rookie {cfg["label"]}',
                    cfg['bins'], cfg['bin_labels'], f'{pos} | {dt}')

# ═══════════════════════════════════════════════
# 9. YEAR-2 PRODUCTION TABLES
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION C: YEAR-2 PRODUCTION × DRAFT CAPITAL")
print("(Does Year-2 production change the picture?)")
print("=" * 90)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    # Only players with at least 2 seasons of data
    pos_df_y2 = pos_df[pos_df['Year'] <= MAX_DRAFT_YEAR - 1].copy()
    tiers = cfg['tiers']

    print(f"\n{'━' * 90}")
    print(f"  {pos} — Year-2 Stats")
    print(f"{'━' * 90}")

    for dt in DRAFT_TIERS_3:
        grp = pos_df_y2[pos_df_y2['DraftTier'] == dt]
        if len(grp) < 5: continue
        print_table(grp, tiers, cfg['y2_stat'], f'Year-2 {cfg["label"]}',
                    cfg['bins'], cfg['bin_labels'], f'{pos} | {dt} (Year 2)')

# ═══════════════════════════════════════════════
# 10. COMBINE METRICS — 40 TIME × DRAFT TIER
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION D: 40-YARD DASH × DRAFT CAPITAL (WR & RB)")
print("=" * 90)

SPEED_TIERS = ['≤4.39 (Elite)', '4.40-4.49 (Fast)', '4.50-4.59 (Avg)', '4.60+ (Slow)']

for pos in ['WR', 'RB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    tiers = cfg['tiers']

    print(f"\n{'━' * 90}")
    print(f"  {pos} — 40-Yard Dash")
    print(f"{'━' * 90}")

    for dt in DRAFT_TIERS_3:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 5: continue

        n_tiers = len(tiers)
        tier_header = '  '.join(f'{t:>10s}' for t in tiers) + '   Hit Rate'
        print(f"\n  {pos} | {dt}  (n={len(grp)})")
        print(f"  {'40-Yard Dash':>22s}  {'n':>5s}  {tier_header}")
        print(f"  {'─' * 22}  {'─' * 5}  {'─' * (12 * n_tiers + 12)}")

        for st in SPEED_TIERS + ['No 40']:
            subset = grp[grp['SpeedTier'] == st]
            if len(subset) < 3: continue
            n = len(subset)
            tier_pcts = [(subset['PeakTier'] == t).mean() * 100 for t in tiers]
            hit_rate = sum(tier_pcts[:2])
            tier_str = '  '.join(f'{p:9.0f}%' for p in tier_pcts)
            print(f"  {st:>22s}  {n:5d}  {tier_str}  {hit_rate:8.0f}%")

        # Overall
        n = len(grp)
        tier_pcts = [(grp['PeakTier'] == t).mean() * 100 for t in tiers]
        hit_rate = sum(tier_pcts[:2])
        tier_str = '  '.join(f'{p:9.0f}%' for p in tier_pcts)
        print(f"  {'─' * 22}  {'─' * 5}  {'─' * (12 * n_tiers + 12)}")
        print(f"  {'OVERALL':>22s}  {n:5d}  {tier_str}  {hit_rate:8.0f}%")

# ═══════════════════════════════════════════════
# 11. SIZE × DRAFT TIER  (WR & RB)
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION E: SIZE (WEIGHT) × DRAFT CAPITAL (WR & RB)")
print("=" * 90)

WR_SIZE_TIERS = ['<200 (Small)', '200-214', '215+ (Big)']
RB_SIZE_TIERS = ['<210 (Small)', '210-224', '225+ (Big)']

for pos, size_tiers in [('WR', WR_SIZE_TIERS), ('RB', RB_SIZE_TIERS)]:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    tiers = cfg['tiers']

    print(f"\n{'━' * 90}")
    print(f"  {pos} — Weight")
    print(f"{'━' * 90}")

    for dt in DRAFT_TIERS_3:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 5: continue

        n_tiers = len(tiers)
        tier_header = '  '.join(f'{t:>10s}' for t in tiers) + '   Hit Rate'
        print(f"\n  {pos} | {dt}  (n={len(grp)})")
        print(f"  {'Weight':>22s}  {'n':>5s}  {tier_header}")
        print(f"  {'─' * 22}  {'─' * 5}  {'─' * (12 * n_tiers + 12)}")

        for st in size_tiers + ['No Wt']:
            subset = grp[grp['SizeTier'] == st]
            if len(subset) < 3: continue
            n = len(subset)
            tier_pcts = [(subset['PeakTier'] == t).mean() * 100 for t in tiers]
            hit_rate = sum(tier_pcts[:2])
            tier_str = '  '.join(f'{p:9.0f}%' for p in tier_pcts)
            print(f"  {st:>22s}  {n:5d}  {tier_str}  {hit_rate:8.0f}%")

        n = len(grp)
        tier_pcts = [(grp['PeakTier'] == t).mean() * 100 for t in tiers]
        hit_rate = sum(tier_pcts[:2])
        tier_str = '  '.join(f'{p:9.0f}%' for p in tier_pcts)
        print(f"  {'─' * 22}  {'─' * 5}  {'─' * (12 * n_tiers + 12)}")
        print(f"  {'OVERALL':>22s}  {n:5d}  {tier_str}  {hit_rate:8.0f}%")

# ═══════════════════════════════════════════════
# 12. MULTI-YEAR HIT RATES — Tier-1 seasons count
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION F: SUSTAINABILITY — # of Tier-1 Seasons by Draft Capital")
print("(How many WR1/RB1/TE1/QB1 seasons did players who hit Tier 1 actually produce?)")
print("=" * 90)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    top_tier = cfg['tiers'][0]
    t1_players = pos_df[pos_df['PeakTier'] == top_tier].copy()

    if len(t1_players) < 3: continue

    print(f"\n  {pos} — Among players who reached {top_tier} (n={len(t1_players)}):")
    print(f"    {top_tier} seasons  |  Count  |  Pct   |  Examples")
    print(f"    {'─' * 70}")

    for nseas in sorted(t1_players['tier1_seasons'].unique()):
        grp = t1_players[t1_players['tier1_seasons'] == nseas]
        pct = len(grp) / len(t1_players) * 100
        examples = grp.head(3)
        names = ', '.join(f"{r['Player']} ('{int(r['Year'])%100} R{int(r['Round'])})" for _, r in examples.iterrows())
        label = f'{int(nseas):d}' if nseas < 8 else f'{int(nseas)}+'
        print(f"    {label:>14s}  |  {len(grp):5d}  | {pct:5.1f}%  |  {names}")

    avg = t1_players['tier1_seasons'].mean()
    med = t1_players['tier1_seasons'].median()
    print(f"    Mean: {avg:.1f}  |  Median: {med:.0f}")

# ═══════════════════════════════════════════════
# 13. NAMED EXAMPLES BY CELL (Rd 1 WR with 750+ rookie yds → who?)
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION G: NAMED PLAYERS BY DRAFT TIER × ROOKIE PRODUCTION BIN")
print("=" * 90)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos]
    tiers = cfg['tiers']
    top_tier = tiers[0]

    print(f"\n{'━' * 90}")
    print(f"  {pos}")
    print(f"{'━' * 90}")

    for dt in DRAFT_TIERS_3:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 5: continue

        print(f"\n  {pos} | {dt}")
        for (lo, hi), label in zip(cfg['bins'], cfg['bin_labels']):
            subset = grp[(grp[cfg['y1_stat']] > lo) & (grp[cfg['y1_stat']] <= hi)]
            if len(subset) < 1: continue

            # Show who became each tier
            hits = subset[subset['PeakTier'] == top_tier]
            misses = subset[subset['PeakTier'] == tiers[-1]]

            hit_names = ', '.join(
                f"{r['Player']}" for _, r in hits.head(4).iterrows()
            ) if len(hits) > 0 else '(none)'

            miss_names = ', '.join(
                f"{r['Player']}" for _, r in misses.head(3).iterrows()
            ) if len(misses) > 0 else '(none)'

            print(f"    Rookie {cfg['label']} {label:>8s}  (n={len(subset):3d})  "
                  f"→ {top_tier}: {hit_names}")
            if len(misses) > 0:
                print(f"    {'':>38s}  → {tiers[-1]}: {miss_names}")

# ═══════════════════════════════════════════════
# 14. CROSS-TAB: YEAR-1 × YEAR-2 → PEAK TIER (WR only, biggest group)
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SECTION H: YEAR-1 × YEAR-2 CROSS-TAB → PEAK OUTCOME (WR, Rd 1-3)")
print("(Does Year-2 confirm or override Year-1 signal?)")
print("=" * 90)

wr_early = skill[(skill['PosGroup'] == 'WR') &
                 (skill['DraftTier'].isin(['Rd 1', 'Rd 2-3'])) &
                 (skill['Year'] <= MAX_DRAFT_YEAR - 1)].copy()

Y1_BINS = [(-1, 300), (300, 600), (600, 99999)]
Y1_LABELS = ['Y1 ≤300', 'Y1 301-600', 'Y1 600+']
Y2_BINS = [(-1, 300), (300, 600), (600, 99999)]
Y2_LABELS = ['Y2 ≤300', 'Y2 301-600', 'Y2 600+']

print(f"\n  WR Rd 1-3 (n={len(wr_early)})")
tier_header = '  '.join(f'{t:>8s}' for t in ['WR1', 'WR2', 'WR3', 'Depth'])
print(f"  {'Y1 \\ Y2':>20s}  {'n':>4s}  {tier_header}  {'Hit%':>6s}")
print(f"  {'─' * 20}  {'─' * 4}  {'─' * 40}  {'─' * 6}")

for (y1lo, y1hi), y1label in zip(Y1_BINS, Y1_LABELS):
    for (y2lo, y2hi), y2label in zip(Y2_BINS, Y2_LABELS):
        subset = wr_early[
            (wr_early['Y1_receiving_yards'] > y1lo) & (wr_early['Y1_receiving_yards'] <= y1hi) &
            (wr_early['Y2_receiving_yards'] > y2lo) & (wr_early['Y2_receiving_yards'] <= y2hi)
        ]
        if len(subset) < 3: continue
        n = len(subset)
        pcts = [(subset['PeakTier'] == t).mean() * 100 for t in ['WR1', 'WR2', 'WR3', 'Depth/Out']]
        hit = pcts[0] + pcts[1]
        tier_str = '  '.join(f'{p:7.0f}%' for p in pcts)
        combo = f'{y1label} / {y2label}'
        print(f"  {combo:>20s}  {n:4d}  {tier_str}  {hit:5.0f}%")

# Same for RB
print(f"\n  RB Rd 1-3")
rb_early = skill[(skill['PosGroup'] == 'RB') &
                 (skill['DraftTier'].isin(['Rd 1', 'Rd 2-3'])) &
                 (skill['Year'] <= MAX_DRAFT_YEAR - 1)].copy()

Y1R_BINS = [(-1, 350), (350, 700), (700, 99999)]
Y1R_LABELS = ['Y1 ≤350', 'Y1 351-700', 'Y1 700+']
Y2R_BINS = [(-1, 350), (350, 700), (700, 99999)]
Y2R_LABELS = ['Y2 ≤350', 'Y2 351-700', 'Y2 700+']

print(f"  (n={len(rb_early)})")
tier_header = '  '.join(f'{t:>8s}' for t in ['RB1', 'RB2', 'RB3', 'Depth'])
print(f"  {'Y1 \\ Y2':>20s}  {'n':>4s}  {tier_header}  {'Hit%':>6s}")
print(f"  {'─' * 20}  {'─' * 4}  {'─' * 40}  {'─' * 6}")

for (y1lo, y1hi), y1label in zip(Y1R_BINS, Y1R_LABELS):
    for (y2lo, y2hi), y2label in zip(Y2R_BINS, Y2R_LABELS):
        subset = rb_early[
            (rb_early['Y1_rushing_yards'] > y1lo) & (rb_early['Y1_rushing_yards'] <= y1hi) &
            (rb_early['Y2_rushing_yards'] > y2lo) & (rb_early['Y2_rushing_yards'] <= y2hi)
        ]
        if len(subset) < 3: continue
        n = len(subset)
        pcts = [(subset['PeakTier'] == t).mean() * 100 for t in ['RB1', 'RB2', 'RB3', 'Depth/Out']]
        hit = pcts[0] + pcts[1]
        tier_str = '  '.join(f'{p:7.0f}%' for p in pcts)
        combo = f'{y1label} / {y2label}'
        print(f"  {combo:>20s}  {n:4d}  {tier_str}  {hit:5.0f}%")

# ═══════════════════════════════════════════════
# 15. PLOTS
# ═══════════════════════════════════════════════
print(f"\n[PLOTS] Generating...")
sns.set_theme(style='whitegrid', font_scale=0.85)

POS_COLORS = {
    'WR': {'WR1': '#1a9850', 'WR2': '#91cf60', 'WR3': '#fee08b', 'Depth/Out': '#d73027'},
    'RB': {'RB1': '#1a9850', 'RB2': '#91cf60', 'RB3': '#fee08b', 'Depth/Out': '#d73027'},
    'TE': {'TE1': '#1a9850', 'TE2': '#91cf60', 'Depth/Out': '#d73027'},
    'QB': {'QB1': '#1a9850', 'QB2': '#91cf60', 'Backup/Out': '#d73027'},
}

fig, axes = plt.subplots(4, 3, figsize=(22, 24))
fig.subplots_adjust(hspace=0.55, wspace=0.28)

for row_idx, pos in enumerate(['WR', 'RB', 'TE', 'QB']):
    cfg = POS_CONFIG[pos]
    tiers = cfg['tiers']
    colors = [POS_COLORS[pos][t] for t in tiers]
    pos_df = skill[skill['PosGroup'] == pos]

    for col_idx, dt in enumerate(DRAFT_TIERS_3):
        ax = axes[row_idx, col_idx]
        grp = pos_df[pos_df['DraftTier'] == dt]

        labels_list, n_list, tier_data = [], [], []
        for (lo, hi), label in zip(cfg['bins'], cfg['bin_labels']):
            subset = grp[(grp[cfg['y1_stat']] > lo) & (grp[cfg['y1_stat']] <= hi)]
            if len(subset) < 3: continue
            labels_list.append(label)
            n_list.append(len(subset))
            tier_data.append([(subset['PeakTier'] == t).mean() * 100 for t in tiers])

        if not labels_list:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{pos} — {dt}')
            continue

        x = np.arange(len(labels_list))
        bottom = np.zeros(len(labels_list))
        for tier_idx, (tier, color) in enumerate(zip(tiers, colors)):
            vals = np.array([td[tier_idx] for td in tier_data])
            ax.bar(x, vals, bottom=bottom, color=color, edgecolor='white',
                   linewidth=0.5, label=tier, width=0.65)
            for j, v in enumerate(vals):
                if v >= 8:
                    ax.text(x[j], bottom[j] + v/2, f'{v:.0f}%',
                           ha='center', va='center', fontsize=7, fontweight='bold',
                           color='white' if color in ['#1a9850', '#d73027'] else 'black')
            bottom += vals

        for j, nn in enumerate(n_list):
            ax.text(x[j], 102, f'n={nn}', ha='center', va='bottom', fontsize=6, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, fontsize=7, rotation=15)
        ax.set_ylabel('% of Players' if col_idx == 0 else '')
        ax.set_ylim(0, 115)
        ax.set_title(f'{pos} — {dt}  (n={len(grp)})', fontweight='bold')
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

plt.suptitle(f'Peak Career Outcome by Draft Capital + Rookie Production\n'
             f'(2000-{MAX_DRAFT_YEAR} draft classes, absolute stat floors)',
             fontsize=14, fontweight='bold', y=1.01)

out_path = OUT_DIR + 'nfl_peak_tiers.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"    Saved: {out_path}")

# ── Second plot: fine-grained draft tiers for WR and RB ──
fig2, axes2 = plt.subplots(2, 6, figsize=(36, 12))
fig2.subplots_adjust(hspace=0.45, wspace=0.25)

for row_idx, pos in enumerate(['WR', 'RB']):
    cfg = POS_CONFIG[pos]
    tiers = cfg['tiers']
    colors = [POS_COLORS[pos][t] for t in tiers]
    pos_df = skill[skill['PosGroup'] == pos]

    for col_idx, dt in enumerate(DRAFT_TIERS_FINE):
        ax = axes2[row_idx, col_idx]
        grp = pos_df[pos_df['DraftTierFine'] == dt]

        labels_list, n_list, tier_data = [], [], []
        for (lo, hi), label in zip(cfg['bins'], cfg['bin_labels']):
            subset = grp[(grp[cfg['y1_stat']] > lo) & (grp[cfg['y1_stat']] <= hi)]
            if len(subset) < 3: continue
            labels_list.append(label)
            n_list.append(len(subset))
            tier_data.append([(subset['PeakTier'] == t).mean() * 100 for t in tiers])

        if not labels_list:
            ax.text(0.5, 0.5, f'n={len(grp)}\nInsuff.', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f'{pos} — {dt}')
            continue

        x = np.arange(len(labels_list))
        bottom = np.zeros(len(labels_list))
        for tier_idx, (tier, color) in enumerate(zip(tiers, colors)):
            vals = np.array([td[tier_idx] for td in tier_data])
            ax.bar(x, vals, bottom=bottom, color=color, edgecolor='white',
                   linewidth=0.5, label=tier, width=0.65)
            for j, v in enumerate(vals):
                if v >= 10:
                    ax.text(x[j], bottom[j] + v/2, f'{v:.0f}%',
                           ha='center', va='center', fontsize=7, fontweight='bold',
                           color='white' if color in ['#1a9850', '#d73027'] else 'black')
            bottom += vals

        for j, nn in enumerate(n_list):
            ax.text(x[j], 102, f'n={nn}', ha='center', va='bottom', fontsize=6, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, fontsize=6, rotation=20)
        ax.set_ylim(0, 115)
        ax.set_title(f'{pos} — {dt}  (n={len(grp)})', fontweight='bold', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('% of Players')
            ax.legend(fontsize=6, loc='upper right')

fig2.suptitle(f'WR & RB Peak Outcome — Fine Draft Splits × Rookie Production\n'
              f'(2000-{MAX_DRAFT_YEAR})',
              fontsize=14, fontweight='bold')

out_path2 = OUT_DIR + 'nfl_peak_tiers_fine.png'
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"    Saved: {out_path2}")

print("\n" + "=" * 90)
print("DONE — all sections complete.")
print("=" * 90)
