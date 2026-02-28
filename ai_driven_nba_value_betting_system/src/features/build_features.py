from __future__ import annotations

"""
Feature engineering for team-level win probability.

We use rolling team form (win pct, point differential) and rest days.
Defaults target the home-team win label.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


def make_team_long(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Return long per-team rows with rolling form stats."""
    home = df[
        ["game_id", "date", "home_team", "away_team", "home_score", "away_score", "home_team_won"]
    ].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["won"] = home["home_team_won"]
    home["point_diff"] = home["home_score"] - home["away_score"]
    home["role"] = "home"

    away = df[
        ["game_id", "date", "home_team", "away_team", "home_score", "away_score", "home_team_won"]
    ].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["won"] = 1 - away["home_team_won"]
    away["point_diff"] = away["away_score"] - away["home_score"]
    away["role"] = "away"

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"])

    long_df["rest_days"] = (
        long_df.groupby("team")["date"].diff().dt.days.fillna(3).clip(lower=0)
    )

    long_df["win_rolling"] = (
        long_df.groupby("team")["won"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
        .fillna(0.5)
    )
    long_df["pd_rolling"] = (
        long_df.groupby("team")["point_diff"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
        .fillna(0.0)
    )
    return long_df


def build_game_features(df: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create game-level features (home/away splits) and return both the feature
    frame and the team-long frame for reuse in scoring future fixtures.
    """
    long_df = make_team_long(df, window=window)
    home_feat = (
        long_df[long_df["role"] == "home"]
        .set_index("game_id")[["win_rolling", "pd_rolling", "rest_days"]]
        .add_prefix("home_")
    )
    away_feat = (
        long_df[long_df["role"] == "away"]
        .set_index("game_id")[["win_rolling", "pd_rolling", "rest_days"]]
        .add_prefix("away_")
    )

    features = df.set_index("game_id")[["date", "home_team", "away_team", "home_team_won"]]
    features = features.join(home_feat).join(away_feat).reset_index()
    features = features.rename(columns={"home_team_won": "target_home_win"})
    return features, long_df


def features_for_matchup(
    team_long: pd.DataFrame, matchup_date: pd.Timestamp, home_team: str, away_team: str
) -> pd.DataFrame:
    """
    Build a single-row feature frame for an upcoming game using the most recent
    rolling stats available before the matchup date.
    """
    def latest(team: str):
        sub = team_long[(team_long["team"] == team) & (team_long["date"] < matchup_date)]
        if sub.empty:
            return {"win_rolling": 0.5, "pd_rolling": 0.0, "rest_days": 3}
        row = sub.iloc[-1]
        return {
            "win_rolling": row["win_rolling"],
            "pd_rolling": row["pd_rolling"],
            "rest_days": row["rest_days"],
        }

    home_stats = latest(home_team)
    away_stats = latest(away_team)
    out = pd.DataFrame(
        [
            {
                "date": matchup_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_win_rolling": home_stats["win_rolling"],
                "home_pd_rolling": home_stats["pd_rolling"],
                "home_rest_days": home_stats["rest_days"],
                "away_win_rolling": away_stats["win_rolling"],
                "away_pd_rolling": away_stats["pd_rolling"],
                "away_rest_days": away_stats["rest_days"],
            }
        ]
    )
    return out

