# NBA Value Betting Demo (Lakers-focused)

Resume-ready, offline-friendly demo that:
- Builds team-level features from sample NBA games (Lakers + a few opponents)
- Trains a simple win-probability model
- Scores sample sportsbook odds (FanDuel, DraftKings, BetMGM, Fanatics, bet365) and flags positive-EV bets

Everything runs locally with bundled CSVs; swap in live data later if you want.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Build features from sample games
python src/app.py prepare

# 2) Train the model
python src/app.py train

# 3) Score sample odds and see positive-EV bets
python src/app.py score --top 5
```

## What’s included
- Sample games: `data/sample/games_sample.csv`
- Sample odds (Lakers vs Warriors mock matchup): `data/sample/odds_sample.csv`
- Feature engineering: rolling win %, point differential, rest days (`src/features/build_features.py`)
- Model: logistic regression win probability (`src/model/train.py`)
- EV + deviation: `src/ev/ev.py`
- CLI orchestration: `src/app.py`

## How it works
- Turn games into per-team rolling form (win pct, point diff, rest days).
- Train a home-win classifier on historical games.
- For a new matchup, predict win probability for each side, convert odds to EV:
  - `EV = (p_hat * decimal_odds) - 1`
- Flag bets where `EV > 0` and optionally use deviation from market average.

## Swapping in real data later
- Games: `src/ingest/balldontlie.py` has a helper to pull real NBA games (free, no key). Replace the sample loader with the fetcher and rerun `prepare`.
- Odds: Wire an odds API/scraper and output the same schema as `data/sample/odds_sample.csv` (date, team, opponent, home_away, book, decimal_odds). Then rerun `score`.

## Notes
- Python 3.11+ recommended (3.13 should work).
- The model is intentionally lightweight; feel free to upgrade to XGBoost/LightGBM and expand features (Elo, injuries, pace/efficiency). A few plots (EV vs deviation, feature importance) would polish the resume story further.

