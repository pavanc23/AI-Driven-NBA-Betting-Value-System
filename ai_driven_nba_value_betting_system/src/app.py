from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ingest import balldontlie, odds
from features import build_features
from model import train as train_mod
from ev import ev as ev_mod


DATA_PROCESSED = Path("data/processed/games_features.parquet")
TEAM_LONG_PATH = Path("data/processed/team_long.parquet")
MODEL_PATH = Path("models/winprob_logreg.pkl")


def cmd_prepare(args: argparse.Namespace) -> None:
    games_df = balldontlie.load_sample_games()
    features_df, team_long = build_features.build_game_features(games_df, window=args.window)

    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(DATA_PROCESSED, index=False)
    team_long.to_parquet(TEAM_LONG_PATH, index=False)

    print(f"Prepared features -> {DATA_PROCESSED}")
    print(f"Saved team-long stats -> {TEAM_LONG_PATH}")


def cmd_train(args: argparse.Namespace) -> None:
    model, metrics = train_mod.train_model(DATA_PROCESSED, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")
    print("Validation metrics:", metrics)


def cmd_score(args: argparse.Namespace) -> None:
    model = train_mod.load_model(MODEL_PATH)
    odds_df = odds.load_sample_odds()
    team_long = pd.read_parquet(TEAM_LONG_PATH)

    preds = []
    for _, row in odds_df.iterrows():
        matchup_date = row["date"]
        team = row["team"]
        opponent = row["opponent"]
        home_away = row["home_away"]

        if home_away == "home":
            feat_row = build_features.features_for_matchup(team_long, matchup_date, team, opponent)
            proba_home = model.predict_proba(feat_row[train_mod.FEATURE_COLS])[0, 1]
            p_team = proba_home
        else:
            feat_row = build_features.features_for_matchup(team_long, matchup_date, opponent, team)
            proba_home = model.predict_proba(feat_row[train_mod.FEATURE_COLS])[0, 1]
            p_team = 1 - proba_home

        preds.append(p_team)

    odds_df["p_hat"] = preds
    scored = ev_mod.add_market_deviation(odds_df)
    scored = ev_mod.add_ev(scored, prob_col="p_hat", odds_col="decimal_odds")

    positive = scored[scored["ev"] > 0].copy()
    if positive.empty:
        print("No positive-EV bets found in sample odds.")
        return

    positive = positive.sort_values("ev", ascending=False)
    print("Top value bets (sample data):")
    print(
        positive[
            ["date", "team", "opponent", "home_away", "book", "decimal_odds", "p_hat", "ev", "deviation"]
        ]
        .head(args.top)
        .to_string(index=False, justify="left", float_format=lambda x: f"{x:0.3f}")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NBA value betting demo (Lakers-focused).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Build features from sample data.")
    p_prepare.add_argument("--window", type=int, default=5, help="Rolling window for form features.")
    p_prepare.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="Train win-probability model.")
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="Score sample odds and show EV bets.")
    p_score.add_argument("--top", type=int, default=5, help="Show top-N bets.")
    p_score.set_defaults(func=cmd_score)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

