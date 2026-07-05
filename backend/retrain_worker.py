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

# ── Training environment pin ─────────────────────────────────────────────────
# Models pickled with one numpy/sklearn version cannot be loaded by another.
# The server runs .venv with these exact versions.  If this worker runs with
# system Python (different versions), the saved .pkl files will be silently
# incompatible and sub-models will fall back to combined_v1 at inference time.
# This happened on 2026-06-09 and took 4 days to detect.
_REQUIRED_VERSIONS = {"numpy": "2.2.6", "scikit-learn": "1.7.2"}

def _verify_training_env():
    """Hard-fail if running with wrong numpy/sklearn."""
    import importlib.metadata as _meta
    errors = []
    for pkg, expected in _REQUIRED_VERSIONS.items():
        try:
            actual = _meta.version(pkg)
        except _meta.PackageNotFoundError:
            errors.append(f"{pkg} not installed")
            continue
        if actual != expected:
            errors.append(f"{pkg} {actual} (expected {expected})")
    if errors:
        print(f"FATAL: Training environment mismatch: {'; '.join(errors)}", flush=True)
        print(f"       Running: {sys.executable}", flush=True)
        print(f"       Expected: .venv/bin/python", flush=True)
        sys.exit(1)


def main() -> None:
    _verify_training_env()
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
        # Use recent-only training window for MLB (2025+ data);
        # other sports use full history (min_season=None).
        min_season = 2025 if sport == "mlb" else None
        result = ml.train_diagnostic_sport_model(sport, save=True, min_season=min_season)
        auc   = result.get("roc_auc", 0)
        saved = result.get("saved", False)
        print(f"[retrain_worker] {sport.upper()}: AUC={auc:.4f} saved={saved}", flush=True)


if __name__ == "__main__":
    main()
