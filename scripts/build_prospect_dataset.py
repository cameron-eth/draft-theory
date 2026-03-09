"""Build the merged Draft Theory prospect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from draft_theory import build_prospect_dataset


DEFAULT_INPUT = REPO_ROOT / "data/raw/nfl_combine_2010_to_2023.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/processed/prospect_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the merged prospect dataset from combine, college, and NFL outcome data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the raw combine CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for the merged prospect dataset.",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help="Optional first college season to pull from CFBD.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Optional last college season to pull from CFBD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    seasons = None
    if args.start_season is not None and args.end_season is not None:
        seasons = list(range(args.start_season, args.end_season + 1))

    df = build_prospect_dataset(
        combine_csv_path=input_path,
        seasons=seasons,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote merged dataset to {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    matched = (
        df["college_match_score"].notna().sum()
        if "college_match_score" in df.columns
        else 0
    )
    print(f"Rows with college match metadata: {matched:,}")


if __name__ == "__main__":
    main()
