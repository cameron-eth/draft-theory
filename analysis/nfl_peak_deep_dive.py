"""
NFL Peak Tiers — DEEP DIVE WITH PLAYER EXAMPLES
================================================
For every draft tier × rookie production bucket:
  - Full player list with peak stats
  - Hits vs Misses with context
  - Surprise outcomes (late bloomers, busts)
  - Historical comp profiles
"""

import pandas as pd
pd.options.future.infer_string = False

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nflreadpy

MAX_DRAFT_YEAR = 2021
MAX_STAT_YEAR  = 2025

print("=" * 100)
print("NFL PEAK TIERS — DEEP DIVE WITH PLAYER EXAMPLES")
print(f"Draft classes 2000-{MAX_DRAFT_YEAR}, stats through {MAX_STAT_YEAR}")
print("=" * 100)

# ═══════════════════════════════════════════════
# 1. LOAD & LINK DATA (same pipeline as main script)
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

def draft_tier_3(rnd):
    if rnd == 1: return 'Rd 1'
    elif rnd <= 3: return 'Rd 2-3'
    else: return 'Rd 4+'
drafted['DraftTier'] = drafted['Round'].apply(draft_tier_3)

POS_MAP = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}
drafted['PosGroup'] = drafted['Pos'].map(POS_MAP)
skill = drafted[drafted['PosGroup'].isin(['WR', 'TE', 'RB', 'QB'])].copy()
print(f"    Skill players: {len(skill):,}")

# ═══════════════════════════════════════════════
# 2. LOAD STATS
# ═══════════════════════════════════════════════
print("[2] Loading stats...")
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

# ═══════════════════════════════════════════════
# 3. CLASSIFY PEAK TIERS
# ═══════════════════════════════════════════════
print("[3] Classifying peak tiers...")

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

# Count tier-1 seasons per player
tier1_by_player = stats_skill[stats_skill['tier_order'] == 1].groupby('player_id').agg(
    t1_count=('season', 'count'),
    t1_seasons_list=('season', list),
).reset_index().rename(columns={'player_id': 'gsis_id'})

# Best season per player
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
peak_df = peak_df.merge(tier1_by_player, on='gsis_id', how='left')
peak_df['t1_count'] = peak_df['t1_count'].fillna(0).astype(int)

skill = skill.merge(peak_df, on='gsis_id', how='left')
skill['PeakTier'] = skill['PeakTier'].fillna(
    skill['PosGroup'].apply(lambda p: 'Backup/Out' if p == 'QB' else 'Depth/Out')
)
skill['t1_count'] = skill['t1_count'].fillna(0).astype(int)

# ═══════════════════════════════════════════════
# 4. YEAR-1 AND YEAR-2 STATS
# ═══════════════════════════════════════════════
print("[4] Computing Year-1 and Year-2 stats...")
stats_lookup = {}
for _, row in stats_all[stats_all['PosGroup'].notna()].iterrows():
    key = (row['player_id'], row['season'])
    stats_lookup[key] = {col: row[col] for col in STAT_COLS if col in stats_all.columns}

y_records = []
for _, row in skill.iterrows():
    gsis = row['gsis_id']
    draft_yr = row['Year']
    rec = {'gsis_id': gsis}
    for yr_label, yr_offset in [('Y1', 0), ('Y2', 1)]:
        ys = stats_lookup.get((gsis, draft_yr + yr_offset), {})
        for col in STAT_COLS:
            rec[f'{yr_label}_{col}'] = ys.get(col, 0)
    y_records.append(rec)

y_df = pd.DataFrame(y_records)
skill = skill.merge(y_df, on='gsis_id', how='left')

# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════
def _i(v):
    """Safe int conversion for NaN values."""
    try:
        if pd.isna(v): return 0
        return int(v)
    except (ValueError, TypeError):
        return 0

def fmt_wr(r):
    peak = f"{_i(r.get('peak_rec_yds',0))} yds / {_i(r.get('peak_rec',0))} rec / {_i(r.get('peak_rec_tds',0))} TDs"
    y1 = f"Y1: {_i(r.get('Y1_receiving_yards',0))} yds"
    draft = f"'{_i(r['Year'])%100} Rd{_i(r['Round'])} #{_i(r['Pick'])}"
    t1n = f" ({_i(r['t1_count'])}x WR1)" if r.get('t1_count', 0) > 0 else ""
    return f"{r['Player']:25s}  {draft:15s}  {y1:15s}  Peak: {peak}{t1n}"

def fmt_rb(r):
    peak = f"{_i(r.get('peak_rush_yds',0))} yds / {_i(r.get('peak_rush_tds',0))} TDs"
    y1 = f"Y1: {_i(r.get('Y1_rushing_yards',0))} yds"
    draft = f"'{_i(r['Year'])%100} Rd{_i(r['Round'])} #{_i(r['Pick'])}"
    t1n = f" ({_i(r['t1_count'])}x RB1)" if r.get('t1_count', 0) > 0 else ""
    return f"{r['Player']:25s}  {draft:15s}  {y1:15s}  Peak: {peak}{t1n}"

def fmt_te(r):
    peak = f"{_i(r.get('peak_rec_yds',0))} yds / {_i(r.get('peak_rec',0))} rec / {_i(r.get('peak_rec_tds',0))} TDs"
    y1 = f"Y1: {_i(r.get('Y1_receiving_yards',0))} yds"
    draft = f"'{_i(r['Year'])%100} Rd{_i(r['Round'])} #{_i(r['Pick'])}"
    t1n = f" ({_i(r['t1_count'])}x TE1)" if r.get('t1_count', 0) > 0 else ""
    return f"{r['Player']:25s}  {draft:15s}  {y1:15s}  Peak: {peak}{t1n}"

def fmt_qb(r):
    peak = f"{_i(r.get('peak_pass_yds',0))} yds / {_i(r.get('peak_pass_tds',0))} TDs"
    y1 = f"Y1: {_i(r.get('Y1_passing_yards',0))} yds"
    draft = f"'{_i(r['Year'])%100} Rd{_i(r['Round'])} #{_i(r['Pick'])}"
    t1n = f" ({_i(r['t1_count'])}x QB1)" if r.get('t1_count', 0) > 0 else ""
    return f"{r['Player']:25s}  {draft:15s}  {y1:18s}  Peak: {peak}{t1n}"

FMT = {'WR': fmt_wr, 'RB': fmt_rb, 'TE': fmt_te, 'QB': fmt_qb}

POS_CONFIG = {
    'WR': {
        'y1_stat': 'Y1_receiving_yards', 'y2_stat': 'Y2_receiving_yards',
        'label': 'Rec Yards',
        'bins': [(-1, 100), (100, 300), (300, 500), (500, 750), (750, 99999)],
        'bin_labels': ['0-100', '101-300', '301-500', '501-750', '750+'],
        'tiers': ['WR1', 'WR2', 'WR3', 'Depth/Out'],
        'peak_sort': 'peak_rec_yds',
    },
    'RB': {
        'y1_stat': 'Y1_rushing_yards', 'y2_stat': 'Y2_rushing_yards',
        'label': 'Rush Yards',
        'bins': [(-1, 100), (100, 350), (350, 600), (600, 900), (900, 99999)],
        'bin_labels': ['0-100', '101-350', '351-600', '601-900', '900+'],
        'tiers': ['RB1', 'RB2', 'RB3', 'Depth/Out'],
        'peak_sort': 'peak_rush_yds',
    },
    'TE': {
        'y1_stat': 'Y1_receiving_yards', 'y2_stat': 'Y2_receiving_yards',
        'label': 'Rec Yards',
        'bins': [(-1, 75), (75, 200), (200, 400), (400, 99999)],
        'bin_labels': ['0-75', '76-200', '201-400', '400+'],
        'tiers': ['TE1', 'TE2', 'Depth/Out'],
        'peak_sort': 'peak_rec_yds',
    },
    'QB': {
        'y1_stat': 'Y1_passing_yards', 'y2_stat': 'Y2_passing_yards',
        'label': 'Pass Yards',
        'bins': [(-1, 0), (0, 1000), (1000, 2500), (2500, 3500), (3500, 99999)],
        'bin_labels': ['Sat (0)', '1-1000', '1001-2500', '2501-3500', '3500+'],
        'tiers': ['QB1', 'QB2', 'Backup/Out'],
        'peak_sort': 'peak_pass_yds',
    },
}

DRAFT_TIERS = ['Rd 1', 'Rd 2-3', 'Rd 4+']

# ═══════════════════════════════════════════════
# 5. FULL ROSTER FOR EVERY CELL
# ═══════════════════════════════════════════════
print("\n" + "=" * 100)
print("SECTION 1: EVERY PLAYER — BY DRAFT TIER × ROOKIE PRODUCTION × PEAK TIER")
print("=" * 100)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    tiers = cfg['tiers']
    fmt = FMT[pos]
    top_tier = tiers[0]
    bot_tier = tiers[-1]

    print(f"\n{'━' * 100}")
    print(f"  {pos}")
    print(f"{'━' * 100}")

    for dt in DRAFT_TIERS:
        grp = pos_df[pos_df['DraftTier'] == dt]
        if len(grp) < 3: continue

        for (lo, hi), label in zip(cfg['bins'], cfg['bin_labels']):
            subset = grp[(grp[cfg['y1_stat']] > lo) & (grp[cfg['y1_stat']] <= hi)]
            if len(subset) < 1: continue

            # Tier counts
            tier_pcts = {}
            for t in tiers:
                cnt = (subset['PeakTier'] == t).sum()
                pct = cnt / len(subset) * 100
                tier_pcts[t] = (cnt, pct)

            tier_summary = ' | '.join(f'{t}: {cnt}({pct:.0f}%)' for t, (cnt, pct) in tier_pcts.items())

            print(f"\n  ┌─ {pos} | {dt} | Rookie {cfg['label']} {label}  (n={len(subset)})")
            print(f"  │  {tier_summary}")
            print(f"  │")

            # HITS — became top tier
            hits = subset[subset['PeakTier'] == top_tier].sort_values(cfg['peak_sort'], ascending=False)
            if len(hits) > 0:
                print(f"  │  ✅ Became {top_tier}:")
                for _, r in hits.iterrows():
                    print(f"  │     {fmt(r)}")

            # SECOND TIER
            t2 = subset[subset['PeakTier'] == tiers[1]].sort_values(cfg['peak_sort'], ascending=False)
            if len(t2) > 0:
                print(f"  │  🟡 Became {tiers[1]}:")
                for _, r in t2.head(5).iterrows():
                    print(f"  │     {fmt(r)}")
                if len(t2) > 5:
                    print(f"  │     ... and {len(t2)-5} more")

            # THIRD TIER (WR/RB only)
            if len(tiers) >= 4:
                t3 = subset[subset['PeakTier'] == tiers[2]].sort_values(cfg['peak_sort'], ascending=False)
                if len(t3) > 0:
                    print(f"  │  🟠 Became {tiers[2]}:")
                    for _, r in t3.head(4).iterrows():
                        print(f"  │     {fmt(r)}")
                    if len(t3) > 4:
                        print(f"  │     ... and {len(t3)-4} more")

            # BUSTS — became depth/out
            busts = subset[subset['PeakTier'] == bot_tier]
            if len(busts) > 0:
                print(f"  │  ❌ {bot_tier}:")
                for _, r in busts.head(5).iterrows():
                    print(f"  │     {fmt(r)}")
                if len(busts) > 5:
                    print(f"  │     ... and {len(busts)-5} more")

            print(f"  └─")

# ═══════════════════════════════════════════════
# 6. SURPRISE OUTCOMES
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 2: BIGGEST SURPRISES — LATE BLOOMERS & UNEXPECTED BUSTS")
print("=" * 100)

for pos in ['WR', 'RB', 'TE', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    tiers = cfg['tiers']
    top_tier = tiers[0]
    bot_tier = tiers[-1]
    fmt = FMT[pos]

    print(f"\n{'━' * 100}")
    print(f"  {pos}")
    print(f"{'━' * 100}")

    # Late-round hits: Rd 4+ who became tier 1
    print(f"\n  🏆 LATE-ROUND GOLD: Rd 4+ picks who became {top_tier}")
    late_hits = pos_df[(pos_df['DraftTier'] == 'Rd 4+') & (pos_df['PeakTier'] == top_tier)].copy()
    late_hits = late_hits.sort_values(cfg['peak_sort'], ascending=False)
    for _, r in late_hits.iterrows():
        print(f"     {fmt(r)}")
    if len(late_hits) == 0:
        print(f"     (none)")

    # Slow starters who became tier 1: low Y1, high peak
    if pos == 'WR':
        y1_thresh = 200
    elif pos == 'RB':
        y1_thresh = 200
    elif pos == 'TE':
        y1_thresh = 75
    elif pos == 'QB':
        y1_thresh = 500

    print(f"\n  🐌 SLOW STARTERS → {top_tier}: Rookie {cfg['label']} ≤{y1_thresh} but peaked at {top_tier}")
    slow_hits = pos_df[(pos_df[cfg['y1_stat']] <= y1_thresh) & (pos_df['PeakTier'] == top_tier)].copy()
    slow_hits = slow_hits.sort_values(cfg['peak_sort'], ascending=False)
    for _, r in slow_hits.iterrows():
        y2_col = cfg['y2_stat']
        y2_val = int(r.get(y2_col, 0))
        extra = f"  → Y2: {y2_val} {cfg['label'].lower()}"
        print(f"     {fmt(r)}{extra}")
    if len(slow_hits) == 0:
        print(f"     (none)")

    # High-capital busts: Rd 1 who became depth/out
    print(f"\n  💀 FIRST-ROUND BUSTS: Rd 1 picks who peaked at {bot_tier}")
    rd1_busts = pos_df[(pos_df['DraftTier'] == 'Rd 1') & (pos_df['PeakTier'] == bot_tier)].copy()
    rd1_busts = rd1_busts.sort_values('Pick')
    for _, r in rd1_busts.iterrows():
        print(f"     {fmt(r)}")
    if len(rd1_busts) == 0:
        print(f"     (none)")

    # Hot starts that fizzled: strong Y1 but ended up in bottom tier
    if pos == 'WR':
        y1_hot = 500
    elif pos == 'RB':
        y1_hot = 500
    elif pos == 'TE':
        y1_hot = 300
    elif pos == 'QB':
        y1_hot = 2000

    print(f"\n  📉 HOT START, BAD ENDING: Rookie {cfg['label']} >{y1_hot} but peaked at {bot_tier}")
    hot_busts = pos_df[(pos_df[cfg['y1_stat']] > y1_hot) & (pos_df['PeakTier'] == bot_tier)].copy()
    hot_busts = hot_busts.sort_values(cfg['y1_stat'], ascending=False)
    for _, r in hot_busts.iterrows():
        print(f"     {fmt(r)}")
    if len(hot_busts) == 0:
        print(f"     (none)")

    # Sustainability kings: most tier-1 seasons
    print(f"\n  👑 SUSTAINABILITY KINGS: Most {top_tier} seasons")
    kings = pos_df[pos_df['t1_count'] > 0].sort_values('t1_count', ascending=False).head(15)
    for _, r in kings.iterrows():
        print(f"     {int(r['t1_count']):2d}x {top_tier}  —  {fmt(r)}")

# ═══════════════════════════════════════════════
# 7. YEAR-1 → YEAR-2 TRAJECTORY STORIES
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 3: YEAR-1 → YEAR-2 TRAJECTORIES — BREAKOUTS & COLLAPSES")
print("=" * 100)

for pos in ['WR', 'RB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[(skill['PosGroup'] == pos) & (skill['Year'] <= MAX_DRAFT_YEAR - 1)].copy()
    tiers = cfg['tiers']
    top_tier = tiers[0]
    fmt = FMT[pos]
    y1_col = cfg['y1_stat']
    y2_col = cfg['y2_stat']

    print(f"\n{'━' * 100}")
    print(f"  {pos}")
    print(f"{'━' * 100}")

    # BREAKOUT: Y2 >> Y1 (big jump)
    pos_df['y_delta'] = pos_df[y2_col] - pos_df[y1_col]
    pos_df['y_ratio'] = pos_df[y2_col] / (pos_df[y1_col] + 1)  # avoid div/0

    print(f"\n  🚀 YEAR-2 BREAKOUTS: Biggest Y1→Y2 jump (min Y2 > 500 {cfg['label'].lower()})")
    breakouts = pos_df[(pos_df[y2_col] > 500) & (pos_df['y_delta'] > 200)].sort_values('y_delta', ascending=False).head(20)
    for _, r in breakouts.iterrows():
        y1v = int(r[y1_col])
        y2v = int(r[y2_col])
        delta = int(r['y_delta'])
        print(f"     {r['Player']:25s}  '{int(r['Year'])%100} Rd{int(r['Round'])} #{int(r['Pick']):3d}  "
              f"Y1: {y1v:5d} → Y2: {y2v:5d}  (+{delta})  Peak: {r['PeakTier']}")

    # COLLAPSE: Y1 >> Y2 (big drop)
    print(f"\n  📉 YEAR-2 COLLAPSES: Biggest Y1→Y2 drop (min Y1 > 500 {cfg['label'].lower()})")
    collapses = pos_df[(pos_df[y1_col] > 500) & (pos_df['y_delta'] < -200)].sort_values('y_delta').head(20)
    for _, r in collapses.iterrows():
        y1v = int(r[y1_col])
        y2v = int(r[y2_col])
        delta = int(r['y_delta'])
        print(f"     {r['Player']:25s}  '{int(r['Year'])%100} Rd{int(r['Round'])} #{int(r['Pick']):3d}  "
              f"Y1: {y1v:5d} → Y2: {y2v:5d}  ({delta})  Peak: {r['PeakTier']}")

# ═══════════════════════════════════════════════
# 8. COMP FINDER: "Players like X"
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 4: HISTORICAL COMP PROFILES")
print("Show all players matching a specific profile → what happened")
print("=" * 100)

# Define compelling profiles to look up
profiles = [
    # (label, pos, draft_tier, y1_stat_col, y1_lo, y1_hi, extra_filter_desc, extra_filter_fn)
    ("Rd 1 WR, 700+ rookie rec yds", 'WR', 'Rd 1', 'Y1_receiving_yards', 700, 99999, None, None),
    ("Rd 1 WR, under 300 rookie rec yds", 'WR', 'Rd 1', 'Y1_receiving_yards', -1, 300, None, None),
    ("Rd 2-3 WR, 500+ rookie rec yds", 'WR', 'Rd 2-3', 'Y1_receiving_yards', 500, 99999, None, None),
    ("Rd 4+ WR, 500+ rookie rec yds", 'WR', 'Rd 4+', 'Y1_receiving_yards', 500, 99999, None, None),
    ("Rd 1 RB, 900+ rookie rush yds", 'RB', 'Rd 1', 'Y1_rushing_yards', 900, 99999, None, None),
    ("Rd 1 RB, under 200 rookie rush yds", 'RB', 'Rd 1', 'Y1_rushing_yards', -1, 200, None, None),
    ("Rd 2-3 RB, 600+ rookie rush yds", 'RB', 'Rd 2-3', 'Y1_rushing_yards', 600, 99999, None, None),
    ("Rd 4+ RB, 500+ rookie rush yds", 'RB', 'Rd 4+', 'Y1_rushing_yards', 500, 99999, None, None),
    ("Rd 1 TE, any production", 'TE', 'Rd 1', 'Y1_receiving_yards', -1, 99999, None, None),
    ("Rd 2-3 TE, 300+ rookie rec yds", 'TE', 'Rd 2-3', 'Y1_receiving_yards', 300, 99999, None, None),
    ("Rd 4+ TE, 200+ rookie rec yds", 'TE', 'Rd 4+', 'Y1_receiving_yards', 200, 99999, None, None),
    ("Rd 1 QB, 3000+ rookie pass yds", 'QB', 'Rd 1', 'Y1_passing_yards', 3000, 99999, None, None),
    ("Rd 1 QB, sat rookie year (0 pass yds)", 'QB', 'Rd 1', 'Y1_passing_yards', -1, 0, None, None),
    ("Rd 1 QB, under 1000 rookie pass yds", 'QB', 'Rd 1', 'Y1_passing_yards', 0, 1000, None, None),
    ("Day 3 QB, 1000+ rookie pass yds", 'QB', 'Rd 4+', 'Y1_passing_yards', 1000, 99999, None, None),
    ("Fast WR (≤4.45) drafted Rd 1", 'WR', 'Rd 1', 'Y1_receiving_yards', -1, 99999,
     "40 ≤ 4.45", lambda df: df[df['Forty'] <= 4.45]),
    ("Slow WR (4.55+) drafted Rd 1-3", 'WR', None, 'Y1_receiving_yards', -1, 99999,
     "40 ≥ 4.55 & Rd 1-3", lambda df: df[(df['Forty'] >= 4.55) & (df['DraftTier'].isin(['Rd 1', 'Rd 2-3']))]),
    ("Big RB (225+) drafted Rd 1-2", 'RB', None, 'Y1_rushing_yards', -1, 99999,
     "Wt ≥ 225 & Rd 1-2", lambda df: df[(df['Wt'] >= 225) & (df['Round'] <= 2)]),
    ("Small RB (<200) any round", 'RB', None, 'Y1_rushing_yards', -1, 99999,
     "Wt < 200", lambda df: df[df['Wt'] < 200]),
]

for label, pos, dt, y1_col, y1_lo, y1_hi, extra_desc, extra_fn in profiles:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    fmt = FMT[pos]
    tiers = cfg['tiers']

    if dt is not None:
        subset = pos_df[pos_df['DraftTier'] == dt]
    else:
        subset = pos_df

    subset = subset[(subset[y1_col] > y1_lo) & (subset[y1_col] <= y1_hi)]

    if extra_fn is not None:
        subset = extra_fn(subset)

    if len(subset) == 0: continue

    # Tier counts
    tier_pcts = {}
    for t in tiers:
        cnt = (subset['PeakTier'] == t).sum()
        pct = cnt / len(subset) * 100
        tier_pcts[t] = (cnt, pct)

    tier_summary = ' | '.join(f'{t}: {cnt}({pct:.0f}%)' for t, (cnt, pct) in tier_pcts.items())
    hit_rate = sum(cnt for t, (cnt, pct) in list(tier_pcts.items())[:2])
    hit_pct = hit_rate / len(subset) * 100

    print(f"\n  ╔═ {label}  (n={len(subset)})  Hit rate: {hit_pct:.0f}%")
    print(f"  ║  {tier_summary}")
    print(f"  ║")

    subset_sorted = subset.sort_values(cfg['peak_sort'], ascending=False)
    for _, r in subset_sorted.iterrows():
        tier_icon = '✅' if r['PeakTier'] == tiers[0] else ('🟡' if r['PeakTier'] == tiers[1] else ('🟠' if len(tiers) > 3 and r['PeakTier'] == tiers[2] else '❌'))
        print(f"  ║  {tier_icon} [{r['PeakTier']:10s}]  {fmt(r)}")

    print(f"  ╚═")

# ═══════════════════════════════════════════════
# 9. DRAFT CLASS SNAPSHOTS — Best and Worst Classes
# ═══════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SECTION 5: DRAFT CLASS RANKINGS — HIT RATE BY YEAR")
print("=" * 100)

for pos in ['WR', 'RB', 'QB']:
    cfg = POS_CONFIG[pos]
    pos_df = skill[skill['PosGroup'] == pos].copy()
    tiers = cfg['tiers']
    top_tier = tiers[0]
    fmt = FMT[pos]

    print(f"\n{'━' * 100}")
    print(f"  {pos} — Hit Rate by Draft Class (Tier 1 + Tier 2)")
    print(f"{'━' * 100}")

    class_data = []
    for yr in range(2000, MAX_DRAFT_YEAR + 1):
        yr_df = pos_df[pos_df['Year'] == yr]
        if len(yr_df) < 3: continue
        t1 = (yr_df['PeakTier'] == tiers[0]).sum()
        t2 = (yr_df['PeakTier'] == tiers[1]).sum()
        hit_pct = (t1 + t2) / len(yr_df) * 100
        t1_names = ', '.join(yr_df[yr_df['PeakTier'] == tiers[0]]['Player'].tolist()[:4])
        class_data.append((yr, len(yr_df), t1, t2, hit_pct, t1_names))

    class_data.sort(key=lambda x: -x[4])

    print(f"\n  {'Year':>6s}  {'n':>4s}  {tiers[0]:>5s}  {tiers[1]:>5s}  {'Hit%':>6s}  {tiers[0]} Names")
    print(f"  {'─' * 6}  {'─' * 4}  {'─' * 5}  {'─' * 5}  {'─' * 6}  {'─' * 60}")
    for yr, n, t1, t2, hit_pct, names in class_data:
        print(f"  {yr:6d}  {n:4d}  {t1:5d}  {t2:5d}  {hit_pct:5.0f}%  {names}")

print("\n" + "=" * 100)
print("DONE.")
print("=" * 100)
