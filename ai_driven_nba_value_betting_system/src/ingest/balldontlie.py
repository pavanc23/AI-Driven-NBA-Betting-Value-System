from __future__ import annotations

"""
Lightweight loader for NBA game data.

For the resume-friendly demo we default to bundled sample data so everything
runs offline. A stub for the balldontlie API is included if you want to swap
in live data later (free, no key required).
"""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


SAMPLE_GAMES_PATH = Path("data/sample/games_sample.csv")


def load_sample_games(path: Path = SAMPLE_GAMES_PATH) -> pd.DataFrame:
    """Load bundled sample games and add helper columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team_won"] = (df["home_score"] > df["away_score"]).astype(int)
    df["game_id"] = range(len(df))
    return df


def fetch_games_from_api(
    team_ids: Iterable[int],
    seasons: Iterable[int],
    per_page: int = 100,
    max_pages: int = 50,
    base_url: str = "https://api.balldontlie.io/v1/games",
    timeout: int = 10,
) -> pd.DataFrame:
    """
    Optional helper to pull real data from balldontlie.
    Returns a DataFrame aligned with the sample schema.

    Note: This is not wired into the CLI by default to keep the demo fully
    offline. Enable and point to your desired seasons if you want fresher data.
    """
    rows = []
    for season in seasons:
        for team_id in team_ids:
            page = 1
            while page <= max_pages:
                resp = requests.get(
                    base_url,
                    params={
                        "team_ids[]": team_id,
                        "seasons[]": season,
                        "per_page": per_page,
                        "page": page,
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data", [])
                if not data:
                    break
                for game in data:
                    rows.append(
                        {
                            "date": game["date"][:10],
                            "home_team": game["home_team"]["full_name"],
                            "away_team": game["visitor_team"]["full_name"],
                            "home_score": game["home_team_score"],
                            "away_score": game["visitor_team_score"],
                        }
                    )
                if payload.get("meta", {}).get("next_page") is None:
                    break
                page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["home_team_won"] = (df["home_score"] > df["away_score"]).astype(int)
    df["game_id"] = range(len(df))
    return df


def save_games(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

