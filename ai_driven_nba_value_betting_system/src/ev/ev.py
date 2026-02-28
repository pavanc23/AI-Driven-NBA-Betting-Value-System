from __future__ import annotations

import pandas as pd


def add_market_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Add market average odds and deviation per matchup side."""
    df = df.copy()
    df["market_avg_odds"] = (
        df.groupby(["date", "team", "opponent"])["decimal_odds"].transform("mean")
    )
    df["deviation"] = (df["decimal_odds"] - df["market_avg_odds"]) / df["market_avg_odds"]
    return df


def add_ev(df: pd.DataFrame, prob_col: str = "p_hat", odds_col: str = "decimal_odds") -> pd.DataFrame:
    """Compute expected value given predicted probability and decimal odds."""
    df = df.copy()
    df["ev"] = (df[prob_col] * df[odds_col]) - 1.0
    return df

