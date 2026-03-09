"""Build the first WR modeling table from the merged prospect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean WR modeling table from the merged prospect dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/processed/prospect_dataset.csv",
        help="Path to the merged prospect dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/processed/wr_model_table.csv",
        help="Output CSV path for the WR modeling table.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2021,
        help="Latest draft year to include for more mature career outcomes.",
    )
    return parser.parse_args()


def draft_pick_value(pick_series):
    pick = pick_series.astype(float)
    return 1 / np.sqrt(pick)


def fit_expected_wav_from_pick(df):
    x = np.log(df["Pick"].astype(float).values)
    y = df["w_av"].astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    expected = intercept + slope * x
    return expected, intercept, slope


def main() -> None:
    import pandas as pd

    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    df = pd.read_csv(input_path)
    wr = df[df["Pos"].astype(str).str.upper() == "WR"].copy()

    wr["Year"] = pd.to_numeric(wr["Year"], errors="coerce")
    wr["Pick"] = pd.to_numeric(wr["Pick"], errors="coerce")
    wr["Round"] = pd.to_numeric(wr["Round"], errors="coerce")
    wr["w_av"] = pd.to_numeric(wr["w_av"], errors="coerce")

    wr = wr[
        (wr["Drafted"].astype(str).str.lower() == "true")
        & wr["Pick"].notna()
        & wr["w_av"].notna()
        & (wr["Year"] <= args.max_year)
        & (wr["college_match_score"] >= 6.5)
        & wr["college_index"].notna()
        & ~wr["college_match_reason"].astype(str).str.startswith("low_confidence:")
    ].copy()

    wr["draft_pick_value"] = draft_pick_value(wr["Pick"])
    wr["combine_missing_40yd"] = wr["40yd"].isna().astype(int)
    wr["combine_missing_vertical"] = wr["Vertical"].isna().astype(int)
    wr["combine_missing_broad_jump"] = wr["Broad Jump"].isna().astype(int)
    wr["combine_missing_bench"] = wr["Bench"].isna().astype(int)
    wr["combine_missing_3cone"] = wr["3Cone"].isna().astype(int)
    wr["combine_missing_shuttle"] = wr["Shuttle"].isna().astype(int)

    expected_wav, intercept, slope = fit_expected_wav_from_pick(wr)
    wr["expected_w_av_from_pick"] = expected_wav
    wr["over_expected_w_av"] = wr["w_av"] - wr["expected_w_av_from_pick"]
    wr["beat_pick_expectation"] = (wr["over_expected_w_av"] > 0).astype(int)

    # First-pass WR-specific college features from the CFBD wide table.
    if "cfbd_career_receiving_yds" in wr.columns and "cfbd_seasons_played" in wr.columns:
        wr["cfbd_career_receiving_yds_per_season"] = (
            wr["cfbd_career_receiving_yds"] / wr["cfbd_seasons_played"].replace(0, np.nan)
        )
    if "cfbd_final_receiving_rec" in wr.columns and "cfbd_final_receiving_yds" in wr.columns:
        wr["cfbd_final_yards_per_reception"] = (
            wr["cfbd_final_receiving_yds"] / wr["cfbd_final_receiving_rec"].replace(0, np.nan)
        )

    output_cols = [
        "Year",
        "Player",
        "School",
        "Pos",
        "Round",
        "Pick",
        "draft_pick_value",
        "w_av",
        "expected_w_av_from_pick",
        "over_expected_w_av",
        "beat_pick_expectation",
        "college_match_score",
        "college_match_reason",
        "height_inches",
        "Weight",
        "40yd",
        "Vertical",
        "Bench",
        "Broad Jump",
        "3Cone",
        "Shuttle",
        "combine_missing_40yd",
        "combine_missing_vertical",
        "combine_missing_broad_jump",
        "combine_missing_bench",
        "combine_missing_3cone",
        "combine_missing_shuttle",
        "cfbd_seasons_played",
        "cfbd_final_conference",
        "cfbd_team_talent",
        "cfbd_career_receiving_rec",
        "cfbd_career_receiving_yds",
        "cfbd_career_receiving_td",
        "cfbd_career_receiving_ypr",
        "cfbd_career_receiving_yds_per_season",
        "cfbd_final_receiving_rec",
        "cfbd_final_receiving_yds",
        "cfbd_final_receiving_td",
        "cfbd_final_receiving_ypr",
        "cfbd_final_yards_per_reception",
    ]
    output_cols = [col for col in output_cols if col in wr.columns]

    wr = wr[output_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wr.to_csv(output_path, index=False)

    print(f"Wrote WR model table to {output_path}")
    print(f"Rows: {len(wr):,}")
    print(f"Columns: {len(wr.columns):,}")
    print(f"Draft classes: {int(wr['Year'].min())}-{int(wr['Year'].max())}")
    print(f"Mean wAV: {wr['w_av'].mean():.2f}")
    print(f"Mean over-expected wAV: {wr['over_expected_w_av'].mean():.2f}")
    print(f"Beat expectation rate: {wr['beat_pick_expectation'].mean():.1%}")
    print(f"Expected wAV formula: {intercept:.3f} + ({slope:.3f} * log(Pick))")


if __name__ == "__main__":
    main()
