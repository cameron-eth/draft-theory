"""Audit combine-to-college matching quality in the merged prospect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect match quality in the merged prospect dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/processed/prospect_dataset.csv",
        help="Path to the merged prospect dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=15,
        help="Number of sample unmatched / low-confidence rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    df = pd.read_csv(input_path)

    print(f"Dataset: {input_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    if "college_match_score" in df.columns:
        print("\nMatch score distribution:")
        print(df["college_match_score"].describe().to_string())
        print(f"Strong matches (>= 9.0): {(df['college_match_score'] >= 9).sum():,}")
        print(f"Acceptable matches (>= 6.5): {(df['college_match_score'] >= 6.5).sum():,}")
        print(f"Weak / unmatched (< 6.5): {(df['college_match_score'] < 6.5).sum():,}")

    if "college_match_reason" in df.columns:
        print("\nTop match reasons:")
        print(df["college_match_reason"].value_counts(dropna=False).head(12).to_string())

    useful_cols = [
        col
        for col in [
            "Player",
            "School",
            "Pos",
            "Year",
            "Pick",
            "college_match_score",
            "college_match_reason",
            "player",
            "team_college",
            "cfbd_final_year",
        ]
        if col in df.columns
    ]

    unmatched = df[df.get("college_match_reason", "").fillna("") == "no_name_match"]
    if len(unmatched):
        print(f"\nSample unmatched rows ({min(args.sample_size, len(unmatched))} shown):")
        print(unmatched[useful_cols].head(args.sample_size).to_string(index=False))

    low_conf = df[df.get("college_match_reason", "").fillna("").astype(str).str.startswith("low_confidence:")]
    if len(low_conf):
        print(f"\nSample low-confidence rows ({min(args.sample_size, len(low_conf))} shown):")
        print(low_conf[useful_cols].head(args.sample_size).to_string(index=False))

    school_vs_name_only = df[
        df.get("college_match_reason", "").fillna("").astype(str).eq("exact_name,position")
    ]
    if len(school_vs_name_only):
        print(f"\nName+position only matches ({len(school_vs_name_only):,} total):")
        print(school_vs_name_only[useful_cols].head(args.sample_size).to_string(index=False))


if __name__ == "__main__":
    main()
