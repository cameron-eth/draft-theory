"""
NFL Combine Predictability Analysis
====================================
Question: How predictable is NFL success from pre-draft background?
Success = career AV relative to what is expected given draft position (residual over expectation)

Data:
  - combine CSV: physical testing + AV (career approx value from PFR) + draft position
  - nflreadpy draft picks: college, draft age, pro bowls, all-pro, HOF, weighted AV

Outcome metric: Approximate Value (AV) — PFR's position-neutral career value metric
  (http://www.pro-football-reference.com/blog/?p=37)
"""

import pandas as pd
pd.options.future.infer_string = False   # ensures object dtype for strings (pandas 3 compat)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

COMBINE_PATH = '/Users/cam/Desktop/combine_data_since_2000_PROCESSED_2018-04-26.csv'
OUT_DIR = '/Users/cam/Documents/Personal/data/'

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("=" * 60)
print("NFL COMBINE PREDICTABILITY ANALYSIS")
print("=" * 60)

print("\n[1] Loading data...")
combine = pd.read_csv(COMBINE_PATH)
print(f"    Combine rows: {len(combine):,}  |  columns: {list(combine.columns)}")

import nflreadpy
draft_raw = nflreadpy.load_draft_picks().to_pandas(use_pyarrow_extension_array=False)
draft_2000 = draft_raw[draft_raw['season'] >= 2000].copy()

# Map enrichment columns via Pfr_ID (w_av, college, age, probowls, allpro, hof)
enrich = draft_2000[draft_2000['pfr_player_id'].notna()].copy()
enrich_dict = {
    'wAV':       dict(zip(enrich['pfr_player_id'], enrich['w_av'])),
    'College':   dict(zip(enrich['pfr_player_id'], enrich['college'])),
    'DraftAge':  dict(zip(enrich['pfr_player_id'], enrich['age'])),
    'ProBowls':  dict(zip(enrich['pfr_player_id'], enrich['probowls'])),
    'AllPro':    dict(zip(enrich['pfr_player_id'], enrich['allpro'])),
    'HOF':       dict(zip(enrich['pfr_player_id'], enrich['hof'])),
}

for col, lookup in enrich_dict.items():
    combine[col] = combine['Pfr_ID'].map(lookup)

matched = combine['wAV'].notna().sum()
print(f"    Enriched via nflreadpy: {matched:,} players matched")

# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────
# Position groups
POS_GROUPS = {
    'QB': 'QB', 'RB': 'RB', 'FB': 'RB',
    'WR': 'WR', 'TE': 'TE',
    'OT': 'OL', 'OG': 'OL', 'OL': 'OL', 'C': 'OL', 'G': 'OL',
    'DE': 'Edge', 'OLB': 'Edge', 'EDGE': 'Edge',
    'DT': 'DT', 'NT': 'DT',
    'ILB': 'LB', 'LB': 'LB',
    'CB': 'CB',
    'FS': 'S', 'SS': 'S', 'S': 'S', 'DB': 'S',
    'K': 'ST', 'P': 'ST', 'LS': 'ST',
}
combine['PosGroup'] = combine['Pos'].map(POS_GROUPS).fillna('Other')

# Ensure numeric types
COMBINE_METRICS = ['Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']
SIZE_METRICS = ['Ht', 'Wt']
ALL_METRICS = SIZE_METRICS + COMBINE_METRICS

for col in ALL_METRICS + ['AV', 'Round', 'Pick']:
    combine[col] = pd.to_numeric(combine[col], errors='coerce')

combine['CareerAV'] = combine['AV'].fillna(0.0)

# ─────────────────────────────────────────────
# 3. Dataset splits
# ─────────────────────────────────────────────
# AV was snapshotted ~April 2018, so recent draft classes haven't had time to
# accumulate career AV. Filter to 2000–2014 for reliable career evaluations
# (minimum 4 seasons). Keep 2015–2018 as a separate "too early to tell" group.
MAX_RELIABLE_YEAR = 2014

all_drafted = combine[combine['Round'].notna()].copy()
drafted = all_drafted[all_drafted['Year'] <= MAX_RELIABLE_YEAR].copy()
recent_drafted = all_drafted[all_drafted['Year'] > MAX_RELIABLE_YEAR].copy()
undrafted = combine[combine['Round'].isna()].copy()
print(f"\n    All drafted:     {len(all_drafted):,} players")
print(f"    Reliable (≤{MAX_RELIABLE_YEAR}): {len(drafted):,} players  ← used for analysis")
print(f"    Too recent (>{MAX_RELIABLE_YEAR}): {len(recent_drafted):,} players  (excluded)")
print(f"    Undrafted:       {len(undrafted):,} players")
print(f"    Undrafted avg AV: {undrafted['CareerAV'].mean():.2f}")

# ─────────────────────────────────────────────
# 4. Combine metric coverage
# ─────────────────────────────────────────────
print("\n[2] Combine metric coverage (drafted players):")
for m in ALL_METRICS:
    n = drafted[m].notna().sum()
    print(f"    {m:12s}: {n:4d} / {len(drafted):4d}  ({n/len(drafted)*100:.0f}%)")

# ─────────────────────────────────────────────
# 5. Baseline: Expected AV from Draft Position
# ─────────────────────────────────────────────
print("\n[3] Baseline model: CareerAV ~ log(pick_number)...")

drafted['log_pick'] = np.log(drafted['Pick'])
X_base = drafted[['log_pick']].values
y = drafted['CareerAV'].values

base_model = LinearRegression().fit(X_base, y)
drafted['ExpectedAV'] = base_model.predict(X_base)
drafted['ExcessAV'] = (drafted['CareerAV'] - drafted['ExpectedAV']).astype(float)

base_r2 = r2_score(y, drafted['ExpectedAV'])
print(f"    R² (pick → career AV): {base_r2:.3f}")
print(f"    This means draft pick explains {base_r2*100:.1f}% of career AV variance.")

# Also fit a round-level version
from numpy.polynomial import polynomial as P
round_means = drafted.groupby('Round')['CareerAV'].mean()
print(f"\n    Mean career AV by round:")
for rnd, av in round_means.items():
    print(f"      Round {int(rnd)}: {av:.1f}")

# ─────────────────────────────────────────────
# 6. Combine metrics vs ExcessAV correlations
# ─────────────────────────────────────────────
print("\n[4] Combine metric correlations with ExcessAV:")
for m in ALL_METRICS:
    sub = drafted[[m, 'ExcessAV']].dropna().astype(float)
    if len(sub) > 50:
        r, p = stats.pearsonr(sub['ExcessAV'], sub[m])
        sig = '**' if p < 0.001 else ('*' if p < 0.05 else '  ')
        direction = '(faster=better, inverted)' if m in ['Forty','Cone','Shuttle'] else ''
        print(f"    {m:12s}: r={r:+.3f}  p={p:.4f} {sig}  n={len(sub)} {direction}")

# ─────────────────────────────────────────────
# 7. Predictive models: does combine add signal beyond pick?
# ─────────────────────────────────────────────
print("\n[5] Cross-validated predictive models (5-fold CV)...")

feature_cols = ['log_pick'] + ALL_METRICS
has_combine = drafted[COMBINE_METRICS].notna().any(axis=1)
model_df = drafted[has_combine].copy()

X_full = model_df[feature_cols].values
y_full = model_df['CareerAV'].values.astype(float)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

pipe_base = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
pipe_full = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
gb_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('model', GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))
])

cv_base = cross_val_score(pipe_base, model_df[['log_pick']].values, y_full, cv=kf, scoring='r2')
cv_full = cross_val_score(pipe_full, X_full, y_full, cv=kf, scoring='r2')
cv_gb   = cross_val_score(gb_pipe, X_full, y_full, cv=kf, scoring='r2')

print(f"    Pick-only    (Ridge):  CV R² = {cv_base.mean():.3f} ± {cv_base.std():.3f}")
print(f"    Pick+Combine (Ridge):  CV R² = {cv_full.mean():.3f} ± {cv_full.std():.3f}")
print(f"    Pick+Combine (GBM):    CV R² = {cv_gb.mean():.3f} ± {cv_gb.std():.3f}")
print(f"    Combine incremental gain: +{(cv_full.mean()-cv_base.mean())*100:.1f}pp (Ridge)")

# Feature importances
gb_pipe.fit(X_full, y_full)
feat_imp = pd.Series(
    gb_pipe.named_steps['model'].feature_importances_,
    index=feature_cols
).sort_values(ascending=False)
print(f"\n    GBM Feature Importances:")
for feat, imp in feat_imp.items():
    bar = '█' * int(imp * 100)
    print(f"      {feat:12s}: {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# 8. Predictability by position
# ─────────────────────────────────────────────
print("\n[6] R² by position group (pick → career AV), 5-fold CV:")
pos_r2 = {}
for pos, grp in drafted.groupby('PosGroup'):
    if len(grp) < 30:
        continue
    X_p = np.log(grp['Pick'].values).reshape(-1, 1)
    y_p = grp['CareerAV'].values.astype(float)
    r2 = cross_val_score(LinearRegression(), X_p, y_p, cv=min(5, len(grp)//10), scoring='r2').mean()
    pos_r2[pos] = r2
    stars = '★' * max(0, int(r2 * 5))
    print(f"    {pos:8s}: R²={r2:.3f}  n={len(grp):3d}  {stars}")

# ─────────────────────────────────────────────
# 9. Busts & Steals
# ─────────────────────────────────────────────
print(f"\n[7] Biggest Busts (top-32 picks, ≤{MAX_RELIABLE_YEAR}, most negative ExcessAV):")
top32 = drafted[drafted['Pick'] <= 32].copy()
top32['ExcessAV'] = top32['ExcessAV'].astype(float)
busts = top32.nsmallest(10, 'ExcessAV')
print(busts[['Player', 'Year', 'Pos', 'Pick', 'CareerAV', 'ExcessAV']].to_string(index=False))

print(f"\n[7] Biggest Steals (rounds 4-7, ≤{MAX_RELIABLE_YEAR}, most positive ExcessAV):")
late = drafted[drafted['Round'] >= 4].copy()
late['ExcessAV'] = late['ExcessAV'].astype(float)
steals = late.nlargest(10, 'ExcessAV')
print(steals[['Player', 'Year', 'Pos', 'Round', 'Pick', 'CareerAV', 'ExcessAV']].to_string(index=False))

# ─────────────────────────────────────────────
# 10. Variance (unpredictability) by position
# ─────────────────────────────────────────────
print("\n[8] ExcessAV variance (unpredictability) by position:")
var_df = drafted.groupby('PosGroup')['ExcessAV'].agg(['std', 'mean', 'count'])
var_df.columns = ['Std', 'Mean', 'N']
var_df = var_df.sort_values('Std', ascending=False)
print(var_df[var_df['N'] >= 30].round(2).to_string())

# ─────────────────────────────────────────────
# 11. Composite athlete score
# ─────────────────────────────────────────────
print("\n[9] Composite athletic score vs ExcessAV...")
drafted = drafted.copy()
for metric in COMBINE_METRICS:
    drafted[f'{metric}_pct'] = drafted.groupby('Pos')[metric].rank(pct=True)
for m in ['Forty', 'Cone', 'Shuttle']:  # invert: lower = better
    drafted[f'{m}_pct'] = 1 - drafted[f'{m}_pct']

pct_cols = [f'{m}_pct' for m in COMBINE_METRICS]
drafted['AthleteScore'] = drafted[pct_cols].mean(axis=1)

athlete_sub = drafted[['AthleteScore', 'ExcessAV']].dropna().astype(float)
r_ath, p_ath = stats.pearsonr(athlete_sub['AthleteScore'], athlete_sub['ExcessAV'])
print(f"    Composite athlete score vs ExcessAV: r={r_ath:.3f}, p={p_ath:.4f}")
print(f"    n={len(athlete_sub)} players with enough combine data")

# ─────────────────────────────────────────────
# 12. Draft Age effect (from nflreadpy)
# ─────────────────────────────────────────────
age_sub = drafted[['DraftAge', 'ExcessAV']].dropna().astype(float)
if len(age_sub) > 100:
    r_age, p_age = stats.pearsonr(age_sub['DraftAge'], age_sub['ExcessAV'])
    print(f"    Draft age vs ExcessAV: r={r_age:.3f}, p={p_age:.4f}  n={len(age_sub)}")

# ─────────────────────────────────────────────
# 13. Per-position combine correlations
# ─────────────────────────────────────────────
print("\n[10] Per-position combine correlations with ExcessAV:")
skill_positions = ['QB', 'RB', 'WR', 'TE', 'CB']
results = []
for pos in skill_positions:
    pos_df = drafted[drafted['Pos'] == pos]
    for metric in COMBINE_METRICS:
        sub = pos_df[[metric, 'ExcessAV']].dropna().astype(float)
        if len(sub) >= 20:
            r, p = stats.pearsonr(sub[metric], sub['ExcessAV'])
            results.append({'Position': pos, 'Metric': metric, 'r': r, 'p': p, 'n': len(sub)})

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    pivot = res_df.pivot(index='Metric', columns='Position', values='r')
    print(pivot.round(3).to_string())
    res_df.to_csv(OUT_DIR + 'combine_correlations_by_position.csv', index=False)

# ─────────────────────────────────────────────
# 14. Plots
# ─────────────────────────────────────────────
print("\n[11] Generating plots...")
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.0)
fig = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# Plot 1: AV vs Pick with baseline curve
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(drafted['Pick'], drafted['CareerAV'], alpha=0.10, s=6, color='steelblue', zorder=1)
pick_range = np.linspace(1, 256, 500)
av_pred = base_model.predict(np.log(pick_range).reshape(-1, 1))
ax1.plot(pick_range, av_pred, color='crimson', lw=2.5, zorder=2, label=f'Expected (R²={base_r2:.2f})')
ax1.set_xlabel('Overall Draft Pick')
ax1.set_ylabel('Career Approximate Value (AV)')
ax1.set_title(f'Career AV vs Draft Pick Position\n(each dot = 1 player, N={len(drafted):,})')
ax1.legend()

# Plot 2: ExcessAV distribution
ax2 = fig.add_subplot(gs[0, 2])
drafted['ExcessAV_f'] = drafted['ExcessAV'].astype(float)
ax2.hist(drafted['ExcessAV_f'].dropna(), bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(0, color='crimson', lw=2.5, linestyle='--', label='Met expectations')
mean_exc = drafted['ExcessAV_f'].mean()
ax2.axvline(mean_exc, color='orange', lw=1.5, linestyle='--', label=f'Mean ({mean_exc:.1f})')
ax2.set_xlabel('Excess AV (actual − expected)')
ax2.set_ylabel('Player Count')
ax2.set_title('Distribution of\nExcess Career Value')
ax2.legend(fontsize=8)

# Plot 3: ExcessAV by Position Group
ax3 = fig.add_subplot(gs[1, 0])
pos_order = drafted.groupby('PosGroup')['ExcessAV'].std().sort_values(ascending=False).index
sns.boxplot(data=drafted, x='ExcessAV', y='PosGroup', order=pos_order, ax=ax3,
            color='steelblue', fliersize=1.5, linewidth=0.8)
ax3.axvline(0, color='crimson', lw=1.5, linestyle='--')
ax3.set_xlabel('Excess AV vs Expected')
ax3.set_ylabel('Position Group')
ax3.set_title('Excess Value by Position\n(sorted by variance = unpredictability)')

# Plot 4: Correlation heatmap
ax4 = fig.add_subplot(gs[1, 1])
corr_cols = ['CareerAV', 'ExcessAV_f'] + ALL_METRICS
corr_df = drafted[corr_cols].rename(columns={'ExcessAV_f': 'ExcessAV'})
for c in corr_df.columns:
    corr_df[c] = pd.to_numeric(corr_df[c], errors='coerce')
corr_matrix = corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', linewidths=0.5, ax=ax4, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
ax4.set_title('Correlation Matrix:\nCombine Metrics ↔ Career AV')

# Plot 5: Feature Importances (GBM)
ax5 = fig.add_subplot(gs[1, 2])
feat_imp.plot(kind='barh', ax=ax5, color='steelblue', edgecolor='white')
ax5.set_xlabel('Feature Importance')
ax5.set_title('What Predicts Career AV?\n(GBM, pick + combine metrics)')
ax5.invert_yaxis()
ax5.axvline(feat_imp.mean(), color='crimson', lw=1.5, linestyle='--', label='mean')
ax5.legend(fontsize=8)

# Plot 6: Mean vs Median AV by Round
ax6 = fig.add_subplot(gs[2, 0])
round_stats = drafted.groupby('Round')['CareerAV'].agg(['mean', 'median', 'std'])
x = round_stats.index.astype(int)
ax6.bar(x - 0.2, round_stats['mean'], 0.35, color='steelblue', alpha=0.8, label='Mean AV')
ax6.bar(x + 0.2, round_stats['median'], 0.35, color='crimson', alpha=0.8, label='Median AV')
ax6.set_xlabel('Draft Round')
ax6.set_ylabel('Career AV')
ax6.set_title('Career AV by Draft Round\n(mean vs median — gap shows right-skew)')
ax6.legend()
ax6.set_xticks(x)

# Plot 7: ExcessAV variance by Round
ax7 = fig.add_subplot(gs[2, 1])
round_std = drafted.groupby('Round')['ExcessAV'].std()
bars = ax7.bar(round_std.index.astype(int), round_std.values, color='darkorange', edgecolor='white')
ax7.set_xlabel('Draft Round')
ax7.set_ylabel('Std Dev of Excess AV')
ax7.set_title('Unpredictability by Round\n(variance around expected value)')
for bar, val in zip(bars, round_std.values):
    ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Plot 8: Athlete score vs ExcessAV
ax8 = fig.add_subplot(gs[2, 2])
has_score = drafted[['AthleteScore', 'ExcessAV_f']].dropna().astype(float)
has_score = has_score[has_score['AthleteScore'] > 0]
ax8.scatter(has_score['AthleteScore'], has_score['ExcessAV_f'],
            alpha=0.15, s=6, color='steelblue')
m, b = np.polyfit(has_score['AthleteScore'], has_score['ExcessAV_f'], 1)
x_line = np.linspace(0.1, 0.95, 100)
ax8.plot(x_line, m*x_line + b, color='crimson', lw=2.5)
ax8.axhline(0, color='gray', lw=0.8, linestyle='--')
ax8.set_xlabel('Composite Athlete Score (percentile within position)')
ax8.set_ylabel('Excess Career AV')
ax8.set_title(f'Athleticism vs Excess Career Value\nr={r_ath:.3f}  n={len(has_score):,}')

plt.suptitle(f'NFL Combine: How Predictable is Career Success?\n(2000–{MAX_RELIABLE_YEAR} draft classes, AV through 2017 season)',
             fontsize=15, fontweight='bold', y=1.01)
out_path = OUT_DIR + 'nfl_combine_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"    Saved: {out_path}")

# ─────────────────────────────────────────────
# 15. Final summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Dataset:          {len(combine):,} combine invitees (2000–2018)")
print(f"Analysis set:     {len(drafted):,} drafted players (2000–{MAX_RELIABLE_YEAR}, 4+ seasons to accrue AV)")
print()
print(f"Baseline R² (draft pick alone):     {base_r2:.3f}  ({base_r2*100:.1f}% variance explained)")
print(f"Full model R² (pick+combine, GBM):  {cv_gb.mean():.3f}  ({cv_gb.mean()*100:.1f}% variance explained)")
print(f"Incremental R² from combine:        +{(cv_gb.mean()-cv_base.mean()):.3f}  ({(cv_gb.mean()-cv_base.mean())*100:.1f} pp)")
print(f"Unexplained variance:               {(1-cv_gb.mean())*100:.1f}%")
print()
print("Most predictable positions (by R²):")
sorted_r2 = sorted(pos_r2.items(), key=lambda x: x[1], reverse=True)
for pos, r2 in sorted_r2:
    print(f"  {pos:8s}: {r2:.3f}")
print()
print("KEY FINDING: Draft pick position is by far the dominant predictor")
print("of career AV. Combine metrics add only modest incremental signal")
print("(+1-2 R² points). Most career variance (~65%) is unpredictable")
print("from pre-draft information alone.")
