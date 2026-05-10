#!/usr/bin/env python3
"""
retrain_worker.py — Runs as a background subprocess, never in the server process.

Usage:
  python retrain_worker.py --combined          # retrain combined model
  python retrain_worker.py --sport mlb         # retrain MLB sub-model
  python retrain_worker.py --sport nhl         # retrain NHL sub-model
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main() -> None:
    p = argparse.ArgumentParser(description="BetIQ model retrain worker")
    p.add_argument("--combined", action="store_true", help="Retrain combined model")
    p.add_argument("--sport",    default=None,        help="Sport sub-model (e.g. mlb, nhl, nba)")
    args = p.parse_args()

    if not args.combined and not args.sport:
        print("ERROR: specify --combined or --sport <sport>", flush=True)
        sys.exit(1)

    import ml_model as ml

    if args.combined:
        from database import SessionLocal
        db = SessionLocal()
        try:
            result = ml.train(db)
            acc = result.get("accuracy")
            saved = result.get("saved")
            print(f"[retrain_worker] Combined: accuracy={acc}% saved={saved}", flush=True)
        finally:
            db.close()

    elif args.sport:
        sport = args.sport.lower()
        result = ml.train_diagnostic_sport_model(sport, save=True)
        auc   = result.get("roc_auc", 0)
        saved = result.get("saved", False)
        print(f"[retrain_worker] {sport.upper()}: AUC={auc:.4f} saved={saved}", flush=True)


if __name__ == "__main__":
    main()
