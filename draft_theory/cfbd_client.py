"""Helpers for authenticated access to the CFBD API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional


def _require_library(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{name}'. Install project requirements first."
        ) from exc


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'. "
            f"Set it in your shell or in a local .env file that is not committed."
        )
    return value


@dataclass(frozen=True)
class CFBDApis:
    """Thin wrapper around CFBD API configuration."""

    base_url: str
    headers: dict


def build_cfbd_apis(api_key: Optional[str] = None) -> CFBDApis:
    """
    Build authenticated CFBD API config.

    We intentionally use direct HTTP requests here instead of the `cfbd` Python
    package because the current `cfbd` and `nflreadpy` dependency trees conflict
    on their pydantic major versions.
    """

    token = api_key or get_required_env("CFBD_API_KEY")
    return CFBDApis(
        base_url="https://api.collegefootballdata.com",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )


def _object_to_dict(record: object) -> dict:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if hasattr(record, "to_str"):
        # Fallback for some generated OpenAPI models that do not serialize cleanly.
        return record.__dict__
    return {
        key: value
        for key, value in getattr(record, "__dict__", {}).items()
        if not key.startswith("_")
    }


def records_to_frame(records: Iterable[object]):
    """Convert CFBD model objects to a pandas DataFrame."""

    pd = _require_library("pandas")
    return pd.DataFrame([_object_to_dict(record) for record in records])


def _get_json(api: CFBDApis, endpoint: str, params: Optional[dict] = None):
    requests = _require_library("requests")
    response = requests.get(
        f"{api.base_url}{endpoint}",
        headers=api.headers,
        params=params or {},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_player_season_stats(
    api: CFBDApis,
    seasons: Iterable[int],
    positions: Optional[Iterable[str]] = None,
):
    """
    Pull player-season college stats across multiple years.

    The generated client shape can vary slightly across versions, so this keeps
    the fetch logic thin and returns a concatenated pandas DataFrame.
    """

    pd = _require_library("pandas")
    frames = []
    position_filter = {pos.upper() for pos in positions or []}

    for season in seasons:
        records = _get_json(
            api,
            "/stats/player/season",
            params={"year": season},
        )
        season_df = records_to_frame(records)
        if season_df.empty:
            continue

        if "year" not in season_df.columns:
            season_df["year"] = season

        if position_filter:
            pos_col = _resolve_column(season_df.columns, ["position", "pos"])
            if pos_col:
                season_df = season_df[
                    season_df[pos_col].astype(str).str.upper().isin(position_filter)
                ].copy()

        frames.append(season_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def fetch_team_talent(api: CFBDApis, seasons: Iterable[int]):
    """Pull CFBD team talent scores, one row per team-season."""

    pd = _require_library("pandas")
    frames = []

    for season in seasons:
        records = _get_json(
            api,
            "/talent",
            params={"year": season},
        )
        season_df = records_to_frame(records)
        if season_df.empty:
            continue
        if "year" not in season_df.columns:
            season_df["year"] = season
        frames.append(season_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None
