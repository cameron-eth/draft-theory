"""Prospect matching utilities for combine/draft rows and college stats."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional


POSITION_FAMILIES = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
}

SCHOOL_ALIASES = {
    "lsu": "louisiana state",
    "ole miss": "mississippi",
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami florida": "miami",
    "usc": "southern california",
    "ucf": "central florida",
    "uconn": "connecticut",
    "pitt": "pittsburgh",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "ucla": "california los angeles",
    "ul lafayette": "louisiana",
    "louisiana lafayette": "louisiana",
    "la monroe": "louisiana monroe",
    "umass": "massachusetts",
    "north carolina state": "nc state",
    "ohio st": "ohio state",
    "iowa st": "iowa state",
    "penn st": "penn state",
    "west michigan": "western michigan",
    "kansas st": "kansas state",
    "oklahoma st": "oklahoma state",
    "boston col": "boston college",
    "texas el paso": "utep",
    "michigan st": "michigan state",
    "arkansas st": "arkansas state",
    "oregon st": "oregon state",
    "arizona st": "arizona state",
    "san diego st": "san diego state",
    "florida st": "florida state",
    "kent st": "kent state",
    "fresno st": "fresno state",
    "south dakota st": "south dakota state",
    "appalachian st": "appalachian state",
    "mississippi st": "mississippi state",
    "san jose st": "san jose state",
    "east washington": "eastern washington",
    "east illinois": "eastern illinois",
    "texas san antonio": "utsa",
    "ala birmingham": "uab",
    "hawai i": "hawaii",
    "hawaii": "hawaii",
    "wayne state mi": "wayne state",
    "north dakota st": "north dakota state",
    "south dakota st": "south dakota state",
    "florida atlantic": "florida atlantic",
    "california davis": "uc davis",
}


@dataclass(frozen=True)
class MatchResult:
    combine_index: int
    college_index: Optional[int]
    score: float
    reason: str


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_school(value: object) -> str:
    school = normalize_text(value)
    return SCHOOL_ALIASES.get(school, school)


def normalize_player_name(value: object) -> str:
    name = normalize_text(value)
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    parts = [part for part in name.split() if part not in suffixes]
    return " ".join(parts)


def position_family(value: object) -> str:
    return POSITION_FAMILIES.get(normalize_text(value).upper(), "")


def _first_token(value: str) -> str:
    return value.split()[0] if value else ""


def _last_token(value: str) -> str:
    return value.split()[-1] if value else ""


def match_score(
    combine_name: str,
    combine_school: str,
    combine_pos: str,
    combine_year: int,
    college_name: str,
    college_school: str,
    college_pos: str,
    college_year: int,
) -> tuple[float, str]:
    score = 0.0
    reasons = []

    if combine_name == college_name:
        score += 6.0
        reasons.append("exact_name")
    else:
        if _last_token(combine_name) and _last_token(combine_name) == _last_token(college_name):
            score += 2.5
            reasons.append("last_name")
        if _first_token(combine_name) and _first_token(combine_name) == _first_token(college_name):
            score += 1.5
            reasons.append("first_name")

    if combine_school and combine_school == college_school:
        score += 3.0
        reasons.append("school")

    if combine_pos and combine_pos == college_pos:
        score += 2.0
        reasons.append("position")

    if college_year == combine_year - 1:
        score += 1.5
        reasons.append("final_season")
    elif college_year in {combine_year - 2, combine_year - 3}:
        score += 0.5
        reasons.append("season_window")

    return score, ",".join(reasons) if reasons else "none"


def build_match_table(combine_df, college_df):
    """
    Return one best CFBD candidate per combine row.

    Matching is intentionally conservative: exact name + school/position/year
    signals dominate, and weak candidates are left unmatched.
    """

    pd = __import__("pandas")

    combine = combine_df.copy()
    college = college_df.copy()

    combine["match_name"] = combine["Player"].map(normalize_player_name)
    combine["match_school"] = combine["School"].map(normalize_school)
    combine["match_pos"] = combine["Pos"].map(position_family)
    combine["match_year"] = pd.to_numeric(combine["Year"], errors="coerce")

    name_col = _resolve_column(college.columns, ["player", "name", "player_name"])
    school_col = _resolve_column(college.columns, ["team", "school"])
    pos_col = _resolve_column(college.columns, ["position", "pos", "position_x", "position_y"])
    year_col = _resolve_column(college.columns, ["year", "season", "cfbd_final_year", "cfbd_final_season"])
    if not all([name_col, school_col, pos_col, year_col]):
        raise ValueError(
            "College stats DataFrame is missing one of the required columns: "
            "player/name, team/school, position/pos, year/season."
        )

    college["match_name"] = college[name_col].map(normalize_player_name)
    college["match_school"] = college[school_col].map(normalize_school)
    college["match_pos"] = college[pos_col].map(position_family)
    college["match_year"] = pd.to_numeric(college[year_col], errors="coerce")

    results = []
    college_groups = college.groupby("match_name", dropna=False)

    for combine_idx, row in combine.iterrows():
        candidates = college_groups.get_group(row["match_name"]) if row["match_name"] in college_groups.groups else college.iloc[0:0]

        if candidates.empty:
            results.append(MatchResult(combine_idx, None, 0.0, "no_name_match"))
            continue

        best_index = None
        best_score = float("-inf")
        best_reason = "none"

        for college_idx, candidate in candidates.iterrows():
            score, reason = match_score(
                combine_name=row["match_name"],
                combine_school=row["match_school"],
                combine_pos=row["match_pos"],
                combine_year=int(row["match_year"]) if pd.notna(row["match_year"]) else 0,
                college_name=candidate["match_name"],
                college_school=candidate["match_school"],
                college_pos=candidate["match_pos"],
                college_year=int(candidate["match_year"]) if pd.notna(candidate["match_year"]) else 0,
            )
            if score > best_score:
                best_index = college_idx
                best_score = score
                best_reason = reason

        has_school_support = "school" in best_reason
        has_time_support = "final_season" in best_reason or "season_window" in best_reason

        if best_score < 8.5 or (not has_school_support and not has_time_support):
            results.append(MatchResult(combine_idx, None, best_score, f"low_confidence:{best_reason}"))
        else:
            results.append(MatchResult(combine_idx, best_index, best_score, best_reason))

    return pd.DataFrame([result.__dict__ for result in results])


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None
