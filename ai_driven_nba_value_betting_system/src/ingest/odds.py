from __future__ import annotations

"""Load sample odds for demonstration."""

from pathlib import Path
import pandas as pd


SAMPLE_ODDS_PATH = Path("data/sample/odds_sample.csv")


def load_sample_odds(path: Path = SAMPLE_ODDS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

