import pandas as pd
import numpy as np
import nflreadpy as nfl

"""
Draft Theory Plan
=================
Goal:
Predict a player's RANGE of NFL outcomes and probability of beating expectations
given his profile before entering the league.

Core principle:
Model *actual career value minus expected value from draft capital*, then estimate
the full outcome distribution (not only a binary hit/miss).
"""

# 1) Feature groups we want in the model
FEATURE_GROUPS = {
    "combine": [
        "height_inches",
        "weight",
        "forty",
        "vertical",
        "broad_jump",
        "bench",
        "three_cone",
        "shuttle",
    ],
    "college_profile": [
        "conference_tier",   # Power / Group / FCS, etc.
        "college_strength",  # Program-level quality proxy
        "age_at_draft",
        "experience_years",
        "games_played_college",
    ],
    "college_production": [
        # Position-specific production features should be added by role templates.
        # Ex: QB EPA/CPOE, WR yards/route, RB rush+rec efficiency, etc.
        "production_index",
        "market_share_index",
        "efficiency_index",
    ],
    "draft_capital": [
        "draft_round",
        "draft_pick",
        "draft_pick_value",   # Continuous value curve from pick number
        "team_context_score", # Optional: roster + coaching context
    ],
}


# 2) Define "success" in components, then combine into one career value score.
#    This avoids reducing career quality to a single noisy stat.
SUCCESS_COMPONENTS = {
    "peak_performance": 0.30,      # Best season quality
    "sustained_performance": 0.25, # Multi-year quality
    "time_to_quality": 0.15,       # Earlier breakout gets more credit
    "duration": 0.15,              # Career length as contributor
    "availability": 0.15,          # Injury / games available component
}


def add_height_inches(df: pd.DataFrame, src_col: str = "Height") -> pd.DataFrame:
    """Convert height from '6-3' format to numeric inches."""
    out = df.copy()
    feet = out[src_col].astype(str).str.split("-").str[0]
    inches = out[src_col].astype(str).str.split("-").str[1]
    out["height_inches"] = pd.to_numeric(feet, errors="coerce") * 12 + pd.to_numeric(
        inches, errors="coerce"
    )
    return out


def draft_pick_to_value(pick_series: pd.Series) -> pd.Series:
    """
    Convert pick number to smooth draft-capital value.
    Higher for earlier picks, with non-linear drop-off.
    """
    pick = pd.to_numeric(pick_series, errors="coerce")
    return 1 / np.sqrt(pick)


def compute_expected_career_value_from_capital(df: pd.DataFrame) -> pd.Series:
    """
    Baseline expectation model from draft capital only.
    In production, this should be replaced by a calibrated historical model:
      E[career_value | draft_pick, position, year]
    """
    return draft_pick_to_value(df["Pick"])


def compute_over_expected_success(
    actual_career_value: pd.Series, expected_career_value: pd.Series
) -> pd.Series:
    """Primary target: value added above expectation from draft capital."""
    return actual_career_value - expected_career_value


def success_probability_label(
    over_expected: pd.Series, threshold: float = 0.0
) -> pd.Series:
    """Binary label for classification views: exceeded expectation or not."""
    return (over_expected > threshold).astype(int)


def outcome_bucket(over_expected: pd.Series) -> pd.Series:
    """
    Coarse range outcome class for player distributions.
    Buckets can be tuned by quantiles after full target build.
    """
    bins = [-np.inf, -0.15, -0.03, 0.03, 0.15, np.inf]
    labels = ["bust", "below_exp", "at_exp", "above_exp", "star"]
    return pd.cut(over_expected, bins=bins, labels=labels)


def modeling_blueprint() -> dict:
    """
    Ordered blueprint for implementation.
    """
    return {
        "step_1_data_sources": [
            "combine measurements/results",
            "college level/context",
            "college production (position-adjusted)",
            "draft capital (round/pick + pick value curve)",
            "NFL outcomes (for target construction)",
        ],
        "step_2_feature_engineering": [
            "normalize combine by position and year",
            "build college production indexes by position",
            "encode conference/program strength",
            "derive draft pick value",
            "add missingness indicators for drills",
        ],
        "step_3_target_build": [
            "compute component scores: peak, sustained, time-to-quality, duration, availability",
            "aggregate to actual_career_value",
            "estimate expected_career_value from draft capital baseline",
            "define over_expected = actual - expected",
            "create binary success and multi-class outcome buckets",
        ],
        "step_4_modeling": [
            "probability model for P(over_expected > 0)",
            "quantile model for range outcomes (P10/P50/P90 over_expected)",
            "segment models by position family if needed",
        ],
        "step_5_evaluation": [
            "calibration by draft round/pick bins",
            "ranking quality (Spearman / top-k lift)",
            "temporal validation (train older years, test newer years)",
            "fairness checks across conferences/positions",
        ],
    }


if __name__ == "__main__":
    plan = modeling_blueprint()
    for section, items in plan.items():
        print(f"\\n{section}:")
        for item in items:
            print(f"  - {item}")