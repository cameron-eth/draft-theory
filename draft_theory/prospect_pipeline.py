"""Unified prospect dataset builder for Draft Theory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from .cfbd_client import build_cfbd_apis, fetch_player_season_stats, fetch_team_talent
from .matching import build_match_table


SKILL_POSITIONS = {"WR", "QB", "TE", "RB", "FB"}


def _require_library(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{name}'. Install project requirements first."
        ) from exc


def load_combine_csv(path: str | Path):
    pd = _require_library("pandas")
    df = pd.read_csv(path)
    if "Pos" in df.columns:
        df = df[df["Pos"].astype(str).str.upper().isin(SKILL_POSITIONS)].copy()
    if "Height" in df.columns:
        height_split = df["Height"].astype(str).str.split("-", n=1, expand=True)
        if height_split.shape[1] == 2:
            df["height_inches"] = (
                pd.to_numeric(height_split[0], errors="coerce") * 12
                + pd.to_numeric(height_split[1], errors="coerce")
            )
    return df


def enrich_with_nfl_outcomes(combine_df):
    """
    Attach draft outcome fields from nflreadpy draft data when available.

    The local combine CSV already has some draft fields, but nflreadpy gives us
    stronger outcome columns like weighted AV, Pro Bowls, All-Pro, and HOF.
    """

    pd = _require_library("pandas")
    nflreadpy = _require_library("nflreadpy")

    draft_raw = nflreadpy.load_draft_picks().to_pandas(use_pyarrow_extension_array=False)
    name_col = _resolve_column(draft_raw.columns, ["player_name", "pfr_player_name"])
    year_col = _resolve_column(draft_raw.columns, ["season", "year"])
    if not all([name_col, year_col]):
        raise ValueError(
            "nflreadpy draft data is missing the expected player-name or season columns."
        )

    draft_raw["match_name"] = draft_raw[name_col].astype(str).str.lower().str.strip()
    draft_raw["match_year"] = pd.to_numeric(draft_raw[year_col], errors="coerce")

    combine = combine_df.copy()
    combine["match_name"] = combine["Player"].astype(str).str.lower().str.strip()
    combine["match_year"] = pd.to_numeric(combine["Year"], errors="coerce")

    keep_cols = [
        "match_name",
        "match_year",
        "pfr_player_id",
        "gsis_id",
        "w_av",
        "probowls",
        "allpro",
        "hof",
        "age",
        "college",
        "team",
    ]
    available_cols = [col for col in keep_cols if col in draft_raw.columns]
    enriched = combine.merge(
        draft_raw[available_cols],
        on=["match_name", "match_year"],
        how="left",
        suffixes=("", "_nfl"),
    )
    return enriched


def summarize_college_stats(college_df):
    """
    Collapse CFBD player-season rows to a prospect-level table.

    The function is schema-tolerant and only aggregates columns that exist.
    """

    pd = _require_library("pandas")
    if college_df.empty:
        return college_df.copy()

    name_col = _resolve_column(college_df.columns, ["player", "name", "player_name"])
    school_col = _resolve_column(college_df.columns, ["team", "school"])
    pos_col = _resolve_column(college_df.columns, ["position", "pos"])
    year_col = _resolve_column(college_df.columns, ["year", "season"])
    if not all([name_col, school_col, pos_col, year_col]):
        raise ValueError(
            "College stats DataFrame is missing one of the required identity columns."
        )

    df = college_df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    conference_col = _resolve_column(df.columns, ["conference"])

    category_col = _resolve_column(df.columns, ["category"])
    stat_type_col = _resolve_column(df.columns, ["statType", "stat_type"])
    stat_col = _resolve_column(df.columns, ["stat"])
    if all([category_col, stat_type_col, stat_col]):
        df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
        df["metric_name"] = (
            df[category_col].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
            + "_"
            + df[stat_type_col].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        )
        index_cols = [name_col, school_col, pos_col, year_col]
        if conference_col:
            index_cols.append(conference_col)
        df = (
            df.pivot_table(
                index=index_cols,
                columns="metric_name",
                values=stat_col,
                aggfunc="sum",
            )
            .reset_index()
        )
        df.columns.name = None
        conference_col = _resolve_column(df.columns, ["conference"])

    numeric_cols = [
        col
        for col in df.columns
        if col not in {name_col, school_col, pos_col, year_col, conference_col}
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    key_cols = [name_col, school_col, pos_col]
    latest_cols = key_cols + [year_col] + numeric_cols
    if conference_col:
        latest_cols.append(conference_col)

    latest_idx = df.groupby(key_cols)[year_col].idxmax()
    latest = df.loc[latest_idx, latest_cols].copy()
    latest = latest.rename(
        columns={col: f"cfbd_final_{col}" for col in [year_col] + numeric_cols + ([conference_col] if conference_col else [])}
    )

    grouped = df.groupby(key_cols, dropna=False)
    career = grouped[numeric_cols].sum(min_count=1).reset_index()
    career = career.rename(columns={col: f"cfbd_career_{col}" for col in numeric_cols})

    seasons_played = grouped[year_col].nunique().reset_index(name="cfbd_seasons_played")

    summary = career.merge(seasons_played, on=key_cols, how="left")
    summary = summary.merge(
        latest,
        on=key_cols,
        how="left",
    )
    return summary


def add_team_context(college_summary_df, team_talent_df):
    """Join CFBD team talent onto the summarized college table using final season + team."""

    pd = _require_library("pandas")
    if college_summary_df.empty or team_talent_df.empty:
        return college_summary_df.copy()

    team_col = _resolve_column(team_talent_df.columns, ["school", "team"])
    year_col = _resolve_column(team_talent_df.columns, ["year", "season"])
    talent_col = _resolve_column(team_talent_df.columns, ["talent"])
    if not all([team_col, year_col, talent_col]):
        return college_summary_df.copy()

    out = college_summary_df.copy()
    school_col = _resolve_column(out.columns, ["team", "school"])
    final_year_col = _resolve_column(out.columns, ["cfbd_final_year"])
    if not all([school_col, final_year_col]):
        return out

    talent = team_talent_df[[team_col, year_col, talent_col]].copy()
    talent = talent.rename(
        columns={
            team_col: school_col,
            year_col: final_year_col,
            talent_col: "cfbd_team_talent",
        }
    )
    talent[final_year_col] = pd.to_numeric(talent[final_year_col], errors="coerce")
    out[final_year_col] = pd.to_numeric(out[final_year_col], errors="coerce")
    return out.merge(talent, on=[school_col, final_year_col], how="left")


def merge_combine_and_college(combine_df, college_summary_df):
    pd = _require_library("pandas")
    match_df = build_match_table(combine_df, college_summary_df)

    if match_df["college_index"].notna().sum() == 0:
        combine = combine_df.copy()
        combine["college_match_score"] = 0.0
        combine["college_match_reason"] = "unmatched"
        return combine

    matched_rows = college_summary_df.reset_index().rename(columns={"index": "college_index"})
    out = (
        combine_df.reset_index()
        .merge(match_df, left_on="index", right_on="combine_index", how="left")
        .merge(matched_rows, on="college_index", how="left", suffixes=("", "_college"))
        .drop(columns=["index", "combine_index"], errors="ignore")
        .rename(
            columns={
                "score": "college_match_score",
                "reason": "college_match_reason",
            }
        )
    )
    return out


def build_prospect_dataset(
    combine_csv_path: str | Path,
    seasons: Optional[Sequence[int]] = None,
    positions: Sequence[str] = ("WR", "QB", "TE", "RB"),
    include_nfl_outcomes: bool = True,
):
    """
    Build the first canonical prospect table:
    combine + draft capital + college production/context + NFL outcomes.
    """

    pd = _require_library("pandas")
    combine_df = load_combine_csv(combine_csv_path)
    if include_nfl_outcomes:
        combine_df = enrich_with_nfl_outcomes(combine_df)

    if seasons is None:
        year_series = pd.to_numeric(combine_df["Year"], errors="coerce").dropna()
        min_year = int(year_series.min()) - 4
        max_year = int(year_series.max()) - 1
        seasons = list(range(min_year, max_year + 1))

    cfbd_apis = build_cfbd_apis()
    college_stats = fetch_player_season_stats(cfbd_apis, seasons=seasons, positions=positions)
    college_summary = summarize_college_stats(college_stats)
    team_talent = fetch_team_talent(cfbd_apis, seasons=seasons)
    college_summary = add_team_context(college_summary, team_talent)

    return merge_combine_and_college(combine_df, college_summary)


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None
