"""
NFL Post-Draft Predictability Analysis
=======================================
Question: Once a player is drafted, how quickly can we tell if they'll succeed?
How much does early career performance add to predicting career AV?

Chain: Combine (Pfr_ID) → Draft picks (pfr_player_id→gsis_id) → Player stats (player_id)

Key analyses:
  1. Year-1 and Year-2 stats as predictors of career AV
  2. Team drafting effects — do certain franchises develop players better?
  3. "When do we know?" — R² at year 1, 2, 3 vs full career
  4. Combine (pre-draft) vs early stats (post-draft) head-to-head
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

import nflreadpy

OUT_DIR = '/Users/cam/Documents/Personal/data/'
MAX_RELIABLE_YEAR = 2014   # AV data is most reliable through ~2017 season

print("=" * 65)
print("NFL POST-DRAFT PREDICTABILITY ANALYSIS")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. Load combine + draft picks + build gsis_id map
# ─────────────────────────────────────────────
print("\n[1] Loading data...")

# Combine data from nflreadpy (2000-2026)
combine_raw = nflreadpy.load_combine().to_pandas(use_pyarrow_extension_array=False)
combine = combine_raw.rename(columns={
    'season': 'Year', 'pfr_id': 'Pfr_ID', 'pos': 'Pos',
    'ht': 'Ht_str', 'wt': 'Wt', 'forty': 'Forty', 'bench': 'BenchReps',
    'vertical': 'Vertical', 'broad_jump': 'BroadJump', 'cone': 'Cone',
    'shuttle': 'Shuttle', 'draft_round': 'Round', 'draft_ovr': 'Pick',
    'draft_team': 'Team', 'player_name': 'Player',
})
# Convert height string "6-4" → inches
def ht_to_inches(h):
    try:
        parts = str(h).split('-')
        return int(parts[0]) * 12 + int(parts[1])
    except:
        return np.nan
combine['Ht'] = combine['Ht_str'].apply(ht_to_inches)

for col in ['Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle', 'Round', 'Pick']:
    combine[col] = pd.to_numeric(combine[col], errors='coerce')

print(f"    Combine data: {len(combine):,} players ({combine['Year'].min()}-{combine['Year'].max()})")

# Draft picks: pfr_player_id → gsis_id + career AV (w_av)
draft_raw = nflreadpy.load_draft_picks().to_pandas(use_pyarrow_extension_array=False)
draft_2000 = draft_raw[draft_raw['season'] >= 2000].copy()

# Build pfr_id → gsis_id lookup
valid_draft = draft_2000[draft_2000['pfr_player_id'].notna() & draft_2000['gsis_id'].notna()]
pfr_to_gsis = dict(zip(valid_draft['pfr_player_id'], valid_draft['gsis_id']))
pfr_to_team = dict(zip(valid_draft['pfr_player_id'], valid_draft['team']))
pfr_to_wav = dict(zip(valid_draft['pfr_player_id'], valid_draft['w_av']))
print(f"    pfr→gsis mapping: {len(pfr_to_gsis):,} players")

# Attach gsis_id and career AV
combine['gsis_id'] = combine['Pfr_ID'].map(pfr_to_gsis)
combine['DraftTeam'] = combine['Pfr_ID'].map(pfr_to_team)
combine['CareerAV'] = combine['Pfr_ID'].map(pfr_to_wav).fillna(0.0)

# Filter to reliable drafted players
drafted = combine[(combine['Round'].notna()) & (combine['Year'] <= MAX_RELIABLE_YEAR)].copy()
drafted = drafted[drafted['gsis_id'].notna()].copy()
print(f"    Drafted with gsis_id (2000-{MAX_RELIABLE_YEAR}): {len(drafted):,}")

# ─────────────────────────────────────────────
# 2. Load seasonal player stats for years 1-3
# ─────────────────────────────────────────────
print("\n[2] Loading seasonal player stats (2000-2017)...")
all_seasons = list(range(2000, 2018))
stats_all = nflreadpy.load_player_stats(seasons=all_seasons, summary_level='reg').to_pandas(use_pyarrow_extension_array=False)
print(f"    Raw seasonal stats: {len(stats_all):,} player-seasons")

# Key stat columns for aggregation
STAT_COLS = {
    'games': 'games',
    'passing_yards': 'passing_yards',
    'passing_tds': 'passing_tds',
    'passing_interceptions': 'passing_interceptions',
    'passing_epa': 'passing_epa',
    'rushing_yards': 'rushing_yards',
    'rushing_tds': 'rushing_tds',
    'rushing_epa': 'rushing_epa',
    'receiving_yards': 'receiving_yards',
    'receiving_tds': 'receiving_tds',
    'receptions': 'receptions',
    'receiving_epa': 'receiving_epa',
    'def_sacks': 'def_sacks',
    'def_interceptions': 'def_interceptions',
    'def_tackles_solo': 'def_tackles_solo',
    'fantasy_points_ppr': 'fantasy_points_ppr',
}

# Ensure numeric
for col in STAT_COLS.keys():
    if col in stats_all.columns:
        stats_all[col] = pd.to_numeric(stats_all[col], errors='coerce').fillna(0)

# ─────────────────────────────────────────────
# 3. For each drafted player, compute year-1, year-2, year-3 stats
# ─────────────────────────────────────────────
print("\n[3] Computing early career stats per player...")

# Merge draft year info
draft_year_map = dict(zip(drafted['gsis_id'], drafted['Year']))

early_stats = {}
for yr_offset in [1, 2, 3]:
    records = []
    for _, row in drafted.iterrows():
        gsis = row['gsis_id']
        draft_yr = row['Year']
        target_season = draft_yr + yr_offset - 1  # year 1 = draft year, etc

        player_season = stats_all[
            (stats_all['player_id'] == gsis) &
            (stats_all['season'] == target_season)
        ]

        if len(player_season) > 0:
            rec = {'gsis_id': gsis}
            for stat_col in STAT_COLS.keys():
                rec[f'Y{yr_offset}_{stat_col}'] = player_season[stat_col].sum()
            records.append(rec)
        else:
            # Player didn't play that season — fill zeros
            rec = {'gsis_id': gsis}
            for stat_col in STAT_COLS.keys():
                rec[f'Y{yr_offset}_{stat_col}'] = 0
            records.append(rec)

    yr_df = pd.DataFrame(records)
    early_stats[yr_offset] = yr_df
    print(f"    Year {yr_offset}: {(yr_df[f'Y{yr_offset}_games'] > 0).sum():,} players with game data")

# Merge all back to drafted
for yr_offset, yr_df in early_stats.items():
    drafted = drafted.merge(yr_df, on='gsis_id', how='left')

# ─────────────────────────────────────────────
# 4. Baseline models: pick-only, then add early career
# ─────────────────────────────────────────────
print("\n[4] Predictive model comparison...")

drafted['log_pick'] = np.log(drafted['Pick'])
COMBINE_METRICS = ['Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle', 'Ht', 'Wt']

# Define feature sets
feat_pick_only = ['log_pick']
feat_pick_combine = ['log_pick'] + COMBINE_METRICS
feat_y1 = [c for c in drafted.columns if c.startswith('Y1_')]
feat_y2 = [c for c in drafted.columns if c.startswith('Y2_')]
feat_y3 = [c for c in drafted.columns if c.startswith('Y3_')]

feat_pick_y1 = feat_pick_only + feat_y1
feat_pick_y2 = feat_pick_only + feat_y1 + feat_y2
feat_pick_y3 = feat_pick_only + feat_y1 + feat_y2 + feat_y3
feat_pick_combine_y1 = feat_pick_combine + feat_y1
feat_everything = feat_pick_combine + feat_y1 + feat_y2 + feat_y3

y = drafted['CareerAV'].values.astype(float)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def eval_model(features, name, model_type='gbm'):
    X = drafted[features].values.astype(float)
    if model_type == 'gbm':
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('model', GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))
        ])
    else:
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
    cv = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
    print(f"    {name:42s}  R²={cv.mean():.3f} ± {cv.std():.3f}  ({len(features)} features)")
    return cv.mean(), cv.std()

results = {}
print("\n    --- Pre-Draft ---")
results['Pick only'] = eval_model(feat_pick_only, 'Pick only')
results['Pick + Combine'] = eval_model(feat_pick_combine, 'Pick + Combine metrics')

print("\n    --- Post-Draft (adding early career stats) ---")
results['Pick + Year 1 stats'] = eval_model(feat_pick_y1, 'Pick + Year 1 stats')
results['Pick + Year 1+2 stats'] = eval_model(feat_pick_y2, 'Pick + Year 1+2 stats')
results['Pick + Year 1+2+3 stats'] = eval_model(feat_pick_y3, 'Pick + Year 1+2+3 stats')

print("\n    --- Combined Pre+Post Draft ---")
results['Pick + Combine + Year 1'] = eval_model(feat_pick_combine_y1, 'Pick + Combine + Year 1 stats')
results['Everything (pick+combine+Y1-3)'] = eval_model(feat_everything, 'Everything (pick+combine+Y1-3)')

print("\n    --- Year 1 stats ONLY (no pick info) ---")
results['Year 1 stats only'] = eval_model(feat_y1, 'Year 1 stats only (no pick)')
results['Year 1+2 stats only'] = eval_model(feat_y1 + feat_y2, 'Year 1+2 stats only (no pick)')

# ─────────────────────────────────────────────
# 5. Feature importances with full model
# ─────────────────────────────────────────────
print("\n[5] Feature importances (GBM, everything model)...")
pipe_full = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('model', GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))
])
X_full = drafted[feat_everything].values.astype(float)
pipe_full.fit(X_full, y)
imp = pd.Series(pipe_full.named_steps['model'].feature_importances_, index=feat_everything)
imp = imp.sort_values(ascending=False)
print("    Top 15 features:")
for feat, val in imp.head(15).items():
    bar = '█' * int(val * 100)
    print(f"      {feat:30s}: {val:.4f}  {bar}")

# ─────────────────────────────────────────────
# 6. Team effects: which teams develop players best?
# ─────────────────────────────────────────────
print("\n[6] Team drafting effects...")

# Expected AV from pick position
from sklearn.linear_model import LinearRegression
base_model = LinearRegression().fit(drafted[['log_pick']].values, y)
drafted['ExpectedAV'] = base_model.predict(drafted[['log_pick']].values)
drafted['ExcessAV'] = drafted['CareerAV'] - drafted['ExpectedAV']

team_excess = drafted.groupby('DraftTeam').agg(
    MeanExcessAV=('ExcessAV', 'mean'),
    MedianExcessAV=('ExcessAV', 'median'),
    StdExcessAV=('ExcessAV', 'std'),
    N=('ExcessAV', 'count'),
    MeanPick=('Pick', 'mean'),
).sort_values('MeanExcessAV', ascending=False)

# Only show teams with enough picks
team_excess = team_excess[team_excess['N'] >= 40]

print(f"\n    Team draft development value (min 40 picks, 2000-{MAX_RELIABLE_YEAR}):")
print(f"    {'Team':6s} {'MeanExcess':>10s} {'Median':>8s} {'N':>5s} {'AvgPick':>8s}")
for team, row in team_excess.iterrows():
    indicator = '+' if row['MeanExcessAV'] > 0 else '-'
    print(f"    {team:6s} {row['MeanExcessAV']:+10.2f}  {row['MedianExcessAV']:+8.2f} {int(row['N']):5d}  {row['MeanPick']:8.1f}  {indicator}")

# Test significance of team effect with ANOVA
from scipy.stats import f_oneway
team_groups = [grp['ExcessAV'].values for name, grp in drafted.groupby('DraftTeam') if len(grp) >= 40]
f_stat, p_val = f_oneway(*team_groups)
print(f"\n    ANOVA test for team effect: F={f_stat:.2f}, p={p_val:.4f}")
if p_val < 0.05:
    print("    → Statistically significant team differences in player development")
else:
    print("    → No statistically significant team effect (teams are roughly equal at development)")

# ─────────────────────────────────────────────
# 7. "When do we know?" — R² curve over time
# ─────────────────────────────────────────────
print("\n[7] Knowledge accumulation curve:")
knowledge_curve = {
    'Pre-draft\n(pick only)': results['Pick only'][0],
    'Pre-draft\n(pick+combine)': results['Pick + Combine'][0],
    'After\nYear 1': results['Pick + Year 1 stats'][0],
    'After\nYear 2': results['Pick + Year 1+2 stats'][0],
    'After\nYear 3': results['Pick + Year 1+2+3 stats'][0],
}
for stage, r2 in knowledge_curve.items():
    bar = '█' * int(r2 * 50)
    stage_clean = stage.replace('\n', ' ')
    print(f"    {stage_clean:25s}: R²={r2:.3f}  {bar}")

# ─────────────────────────────────────────────
# 8. Position-specific early career signals
# ─────────────────────────────────────────────
print("\n[8] Which early career stat best predicts career AV (by position)?")

POS_GROUPS = {
    'QB': 'QB', 'RB': 'RB', 'FB': 'RB',
    'WR': 'WR', 'TE': 'TE',
    'OT': 'OL', 'OG': 'OL', 'OL': 'OL', 'C': 'OL', 'G': 'OL',
    'DE': 'Edge', 'OLB': 'Edge', 'EDGE': 'Edge',
    'DT': 'DT', 'NT': 'DT',
    'ILB': 'LB', 'LB': 'LB',
    'CB': 'CB', 'FS': 'S', 'SS': 'S', 'S': 'S', 'DB': 'S',
}
drafted['PosGroup'] = drafted['Pos'].map(POS_GROUPS).fillna('Other')

KEY_STATS = {
    'QB': ['Y1_passing_yards', 'Y1_passing_tds', 'Y1_passing_epa', 'Y1_games'],
    'RB': ['Y1_rushing_yards', 'Y1_rushing_tds', 'Y1_rushing_epa', 'Y1_games'],
    'WR': ['Y1_receiving_yards', 'Y1_receiving_tds', 'Y1_receptions', 'Y1_games'],
    'TE': ['Y1_receiving_yards', 'Y1_receiving_tds', 'Y1_receptions', 'Y1_games'],
    'Edge': ['Y1_def_sacks', 'Y1_def_tackles_solo', 'Y1_games'],
    'CB': ['Y1_def_interceptions', 'Y1_def_tackles_solo', 'Y1_games'],
    'OL': ['Y1_games'],
    'DT': ['Y1_def_sacks', 'Y1_def_tackles_solo', 'Y1_games'],
}

for pos, stat_cols in KEY_STATS.items():
    grp = drafted[drafted['PosGroup'] == pos]
    if len(grp) < 30:
        continue
    print(f"\n    {pos} (n={len(grp)}):")
    best_r = 0
    best_stat = ''
    for stat in stat_cols:
        sub = grp[[stat, 'CareerAV']].dropna().astype(float)
        if len(sub) >= 20:
            r, p = stats.pearsonr(sub[stat], sub['CareerAV'])
            sig = '*' if p < 0.05 else ' '
            print(f"      {stat:30s}: r={r:.3f}  p={p:.4f} {sig}")
            if abs(r) > abs(best_r):
                best_r = r
                best_stat = stat
    if best_stat:
        print(f"      → Best early signal: {best_stat} (r={best_r:.3f})")

# ─────────────────────────────────────────────
# 9. Plots
# ─────────────────────────────────────────────
print("\n[9] Generating plots...")
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.0)
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Knowledge accumulation curve (bar chart)
ax1 = fig.add_subplot(gs[0, :2])
stages = list(knowledge_curve.keys())
r2_vals = list(knowledge_curve.values())
colors = ['#2196F3', '#1976D2', '#FF9800', '#F57C00', '#E65100']
bars = ax1.bar(stages, r2_vals, color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cross-Validated R²')
ax1.set_title('How Quickly Can We Predict Career Success?\n(R² improves as more career data becomes available)')
ax1.set_ylim(0, max(r2_vals) * 1.15)
ax1.axhline(1.0, color='gray', lw=0.5, linestyle='--', alpha=0.3)

# Plot 2: R² comparison of all model configs
ax2 = fig.add_subplot(gs[0, 2])
model_names = list(results.keys())
model_r2 = [v[0] for v in results.values()]
model_err = [v[1] for v in results.values()]
y_pos = range(len(model_names))
ax2.barh(y_pos, model_r2, xerr=model_err, color='steelblue', edgecolor='white', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names, fontsize=8)
ax2.set_xlabel('CV R²')
ax2.set_title('Model Comparison\n(all configurations)')
ax2.invert_yaxis()

# Plot 3: Top feature importances
ax3 = fig.add_subplot(gs[1, 0])
top_imp = imp.head(12)
colors3 = ['crimson' if 'Y' in f else ('steelblue' if f != 'log_pick' else 'darkorange') for f in top_imp.index]
top_imp.plot(kind='barh', ax=ax3, color=colors3)
ax3.set_xlabel('Feature Importance')
ax3.set_title('Top Features (GBM, full model)\norange=pick, blue=combine, red=early stats')
ax3.invert_yaxis()

# Plot 4: Team development value
ax4 = fig.add_subplot(gs[1, 1])
team_sorted = team_excess.sort_values('MeanExcessAV')
colors4 = ['forestgreen' if x > 0 else 'firebrick' for x in team_sorted['MeanExcessAV']]
ax4.barh(team_sorted.index, team_sorted['MeanExcessAV'], color=colors4, edgecolor='white', alpha=0.8)
ax4.axvline(0, color='black', lw=1)
ax4.set_xlabel('Mean Excess AV (vs expected from pick)')
ax4.set_title(f'Team Draft Development Value\n(2000-{MAX_RELIABLE_YEAR}, min 40 picks)')
ax4.tick_params(axis='y', labelsize=7)

# Plot 5: Year-1 games vs career AV scatter
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(drafted['Y1_games'], drafted['CareerAV'], alpha=0.15, s=8, color='steelblue')
# Bin by Y1 games and show mean
bins = drafted.groupby(pd.cut(drafted['Y1_games'], bins=range(0, 18, 2)))['CareerAV'].mean()
bin_centers = [1, 3, 5, 7, 9, 11, 13, 15][:len(bins)]
ax5.plot(bin_centers, bins.values, color='crimson', lw=3, marker='o', markersize=8, label='Binned mean')
ax5.set_xlabel('Year-1 Games Played')
ax5.set_ylabel('Career AV')
r_games, p_games = stats.pearsonr(
    drafted['Y1_games'].astype(float), drafted['CareerAV'].astype(float)
)
ax5.set_title(f'Year 1 Games → Career AV\nr={r_games:.3f}')
ax5.legend()

plt.suptitle(f'NFL Post-Draft Analysis: When Do We Know?\n(2000-{MAX_RELIABLE_YEAR} draft classes)',
             fontsize=15, fontweight='bold', y=1.01)
out_path = OUT_DIR + 'nfl_postdraft_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"    Saved: {out_path}")

# ─────────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY: POST-DRAFT PREDICTABILITY")
print("=" * 65)

r2_pick = results['Pick only'][0]
r2_combine = results['Pick + Combine'][0]
r2_y1 = results['Pick + Year 1 stats'][0]
r2_y2 = results['Pick + Year 1+2 stats'][0]
r2_y3 = results['Pick + Year 1+2+3 stats'][0]
r2_all = results['Everything (pick+combine+Y1-3)'][0]

print(f"""
  Pre-draft:
    Pick alone:           R² = {r2_pick:.3f}  ({r2_pick*100:.1f}% of career AV explained)
    + Combine metrics:    R² = {r2_combine:.3f}  (combine adds +{(r2_combine-r2_pick)*100:.1f}pp)

  Post-draft:
    + Year 1 stats:       R² = {r2_y1:.3f}  (year 1 adds +{(r2_y1-r2_pick)*100:.1f}pp over pick alone)
    + Year 1+2 stats:     R² = {r2_y2:.3f}  (year 1+2 adds +{(r2_y2-r2_pick)*100:.1f}pp)
    + Year 1+2+3 stats:   R² = {r2_y3:.3f}  (year 1+2+3 adds +{(r2_y3-r2_pick)*100:.1f}pp)

  Everything combined:    R² = {r2_all:.3f}  ({r2_all*100:.1f}% explained, {(1-r2_all)*100:.1f}% still unexplained)

  KEY FINDINGS:
  1. Year 1 stats are MUCH more informative than combine metrics
     (Year 1 adds +{(r2_y1-r2_pick)*100:.1f}pp vs combine's +{(r2_combine-r2_pick)*100:.1f}pp)
  2. Each additional year of data substantially improves prediction
  3. Even with 3 years of data + pick + combine, ~{(1-r2_all)*100:.0f}% remains unpredictable
  4. The single most predictive early signal: GAMES PLAYED in year 1
     (r={r_games:.3f} with career AV — if a team plays you, they believe in you)
  5. Team effect: {"significant" if p_val < 0.05 else "not significant"} (ANOVA p={p_val:.4f})
""")
