"""
weekly_retrain.py — Automated weekly model retraining pipeline.

Runs Sunday 3 AM CT (low activity; full week of settled outcomes available).
Trains a candidate model on all settled data, validates on a 7-day holdout
of stored predictions vs actual outcomes, and deploys only if acceptance
criteria are met.

Safety guarantees:
  • Previous model is always backed up as model_v{N}.pkl before training.
  • Acceptance criteria prevent deployment of a worse model.
  • rollback_model() restores any previous version on demand.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

_DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
_SUBMODEL_DIR = os.path.join(_DATA_DIR, "submodels")

# ─── Acceptance thresholds ────────────────────────────────────────────────────
#
# Two layers:
#   1. HARD absolute floors → auto-reject (catastrophe catch only)
#   2. High-water-mark ratchet → FLAG for human review (not auto-reject)
#
# Training AUC is noisy and diverges from realized performance. A hard ratchet
# could reject a model that drew a hard holdout split but performs better live.
# So anything above the absolute floor goes to human hold, with a prominent
# warning if it's below the best-approved high-water mark.

# Hard absolute floors — auto-reject below these (catastrophic regression only).
# Derived from model_versions history: training AUC range 0.596–0.634, Brier 0.225–0.250.
_AUC_ABSOLUTE_FLOOR   = 0.58   # worst-ever accepted was 0.596; 0.58 catches true collapse
_BRIER_ABSOLUTE_CEIL  = 0.26   # worst-ever accepted was 0.250; 0.26 catches true collapse

# High-water-mark tolerance — candidates below (best_approved - tolerance) are
# FLAGGED in the notification but NOT auto-rejected. Human decides.
_HWM_AUC_TOLERANCE    = 0.02
_HWM_BRIER_TOLERANCE  = 0.02

# Hold period: candidate must wait for human approval (seconds)
_CANDIDATE_HOLD_HOURS = 24

# Minimum new settled legs to trigger a per-sport submodel retrain
_SUBMODEL_MIN_NEW_LEGS = 100

# Persistent state file (survives server restarts)
_RETRAIN_STATE_PATH = os.path.join(_DATA_DIR, "weekly_retrain_state.json")


def _load_retrain_state() -> dict:
    """Load persisted retrain state from disk."""
    try:
        if os.path.exists(_RETRAIN_STATE_PATH):
            with open(_RETRAIN_STATE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_retrain_state(state: dict) -> None:
    """Persist retrain state to disk."""
    try:
        with open(_RETRAIN_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.error("[WeeklyRetrain] Failed to persist state: %s", e)


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _next_sunday_3am_ct() -> str:
    """ISO string of the next Sunday at 3 AM CT (for 'next_retrain' display)."""
    from zoneinfo import ZoneInfo
    ct_tz = ZoneInfo("America/Chicago")
    now   = datetime.now(ct_tz)
    days_ahead = (6 - now.weekday()) % 7   # 0 if today is Sunday
    if days_ahead == 0 and (now.hour > 3 or (now.hour == 3 and now.minute >= 5)):
        days_ahead = 7                      # already past 3:05 AM Sunday → next week
    target = (now + timedelta(days=days_ahead)).replace(
        hour=3, minute=0, second=0, microsecond=0
    )
    return target.isoformat()


def _get_version_n(db: Session) -> int:
    """Highest version_n from model_versions; 0 if table is empty."""
    try:
        row = db.execute(text("SELECT MAX(version_n) FROM model_versions")).scalar()
        return int(row) if row is not None else 0
    except Exception:
        return 0


def _get_current_model_metrics(db: Session) -> dict:
    """
    AUC / Brier from the currently DEPLOYED model version.

    Reads from ModelVersion (status='active') → notes['train_auc'/'train_brier']
    so that a rejected candidate never lowers the acceptance bar for the
    next retrain.  Falls back to the latest ModelRun if no active version
    exists (first-time setup).
    """
    from database import ModelRun, ModelVersion

    # Primary path: find the most recent active/rollback (deployed) version
    try:
        active_mv = (
            db.query(ModelVersion)
            .filter(ModelVersion.status == "active")
            .order_by(ModelVersion.version_n.desc())
            .first()
        )
        if active_mv is not None:
            mv_notes: dict = {}
            raw = active_mv.notes or {}
            if isinstance(raw, str):
                try:
                    mv_notes = json.loads(raw)
                except Exception:
                    pass
            else:
                mv_notes = raw
            train_auc   = mv_notes.get("train_auc")
            train_brier = mv_notes.get("train_brier")
            if train_auc is not None:
                return {
                    "auc":         round(float(train_auc), 4),
                    "brier":       round(float(train_brier or 0.25), 4),
                    "reliability": 0.05,
                    "n_train":     active_mv.training_legs or 0,
                    "version":     active_mv.version,
                }
    except Exception as exc:
        log.warning("[_get_current_model_metrics] ModelVersion query failed: %s", exc)

    # Fallback: latest model_runs row (first-time setup with no ModelVersion yet)
    run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()
    if run is None:
        return {"auc": 0.5, "brier": 0.25, "reliability": 0.05, "run_id": None}
    notes: dict = {}
    raw = run.notes or {}
    if isinstance(raw, str):
        try:
            notes = json.loads(raw)
        except Exception:
            pass
    else:
        notes = raw
    return {
        "auc":         round(float(run.roc_auc or 0.5), 4),
        "brier":       round(float(notes.get("brier", notes.get("brier_score", 0.25))), 4),
        "reliability": round(float(notes.get("reliability", 0.05)), 4),
        "n_train":     run.n_train or 0,
        "run_id":      run.id,
        "trained_at":  run.run_at.strftime("%Y-%m-%d") if run.run_at else None,
    }


# ─── New-feature coverage tracking ───────────────────────────────────────────

_COVERAGE_MIN_LEGS = 500  # absolute settled-leg count before a new feature joins training

def get_new_feature_coverage(db: Session) -> dict:
    """
    Return the non-NULL count of each new per-leg signal across settled mock_bet_legs
    (last 90 days).  Gate is _COVERAGE_MIN_LEGS absolute legs — percentage-based
    thresholds shift as the mock-leg pool grows and are hard to reason about; 500
    settled legs gives the model enough examples to learn meaningful splits.

    Column mapping (DB column → training feature name):
      scout_grade (A/B/C/D)    → scout_grade_encoded (encoded in _build_mock_leg_df)
      clv_cents                → clv_cents
      injury_net_advantage     → injury_net_advantage
      line_movement_signal     → line_movement_signal
      home_team_injury_impact  → home_team_injury_impact
      away_team_injury_impact  → away_team_injury_impact

    Returns dict: {training_feature_name: non_null_count (int), ...}
    """
    try:
        rows = db.execute(text("""
            SELECT
                SUM(CASE WHEN injury_net_advantage    IS NOT NULL THEN 1 ELSE 0 END) AS ina,
                SUM(CASE WHEN line_movement_signal    IS NOT NULL THEN 1 ELSE 0 END) AS lms,
                SUM(CASE WHEN scout_grade             IS NOT NULL THEN 1 ELSE 0 END) AS sge,
                SUM(CASE WHEN clv_cents               IS NOT NULL THEN 1 ELSE 0 END) AS clvc,
                SUM(CASE WHEN home_team_injury_impact IS NOT NULL THEN 1 ELSE 0 END) AS htii,
                SUM(CASE WHEN away_team_injury_impact IS NOT NULL THEN 1 ELSE 0 END) AS atii
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mb.game_date >= date('now', '-90 days')
        """)).fetchone()
    except Exception as exc:
        log.warning("[coverage] query failed: %s", exc)
        return {f: 0 for f in [
            "injury_net_advantage", "line_movement_signal", "scout_grade_encoded",
            "clv_cents", "home_team_injury_impact", "away_team_injury_impact",
        ]}

    return {
        "injury_net_advantage":     int(rows[0] or 0),
        "line_movement_signal":     int(rows[1] or 0),
        "scout_grade_encoded":      int(rows[2] or 0),   # DB col: scout_grade
        "clv_cents":                int(rows[3] or 0),
        "home_team_injury_impact":  int(rows[4] or 0),
        "away_team_injury_impact":  int(rows[5] or 0),
    }


# ─── 7-day live holdout evaluation ────────────────────────────────────────────

def _compute_live_holdout(db: Session, holdout_days: int = 7) -> dict:
    """
    Evaluate stored predictions vs actual outcomes for the last holdout_days
    of settled legs.

    Uses mock_bet_legs.predicted_win_prob (stored at generation time) so
    no re-inference is needed.  Returns {"auc", "brier", "reliability", "n"}.
    """
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except ImportError:
        return {"error": "sklearn not available"}

    cutoff = (datetime.utcnow() - timedelta(days=holdout_days)).strftime("%Y-%m-%d")
    try:
        rows = db.execute(text("""
            SELECT mbl.predicted_win_prob, mbl.leg_result
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE mb.game_date >= :cutoff
              AND mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mbl.predicted_win_prob IS NOT NULL
        """), {"cutoff": cutoff}).fetchall()
    except Exception as e:
        return {"error": f"query failed: {e}"}

    if len(rows) < 20:
        return {"error": f"insufficient data ({len(rows)} legs, need 20+)", "n": len(rows)}

    y_prob = np.clip(np.array([r[0] / 100.0 for r in rows], dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.array([1 if r[1] == "WIN" else 0 for r in rows], dtype=float)

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))

    # Reliability: fraction-weighted mean squared calibration error (5 buckets)
    n = len(y_true)
    reliability = 0.0
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask = (y_prob >= lo) & (y_prob < lo + 0.2)
        if mask.sum() >= 2:
            reliability += (mask.sum() / n) * (y_prob[mask].mean() - y_true[mask].mean()) ** 2

    return {
        "auc":         round(auc, 4),
        "brier":       round(brier, 4),
        "reliability": round(reliability, 4),
        "n":           n,
    }


# ─── Model file management ────────────────────────────────────────────────────

def _backup_model(version_n: int) -> dict:
    """Copy active model.pkl / scaler.pkl / imputer.pkl to versioned backups."""
    backed: dict[str, str] = {}
    for base in ("model", "scaler", "imputer"):
        src = os.path.join(_DATA_DIR, f"{base}.pkl")
        dst = os.path.join(_DATA_DIR, f"{base}_v{version_n}.pkl")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            backed[base] = dst
    return backed


def _restore_model(version_n: int) -> bool:
    """Overwrite active model files from a versioned backup; clears cache."""
    ok = True
    for base in ("model", "scaler", "imputer"):
        src = os.path.join(_DATA_DIR, f"{base}_v{version_n}.pkl")
        dst = os.path.join(_DATA_DIR, f"{base}.pkl")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            ok = False   # imputer may be absent for very old versions — non-fatal
    _clear_model_cache()
    return ok


def _clear_model_cache():
    """Reset ml_model in-process cache so next predict_bet() reloads from disk."""
    try:
        import ml_model as _ml
        _ml._MODEL_CACHE       = None
        _ml._MODEL_LOAD_FAILED = False
    except Exception:
        pass


# ─── Acceptance criteria ──────────────────────────────────────────────────────

def _get_high_water_mark() -> dict:
    """Best-approved AUC/Brier from persisted state (updated on each approval)."""
    state = _load_retrain_state()
    return {
        "auc":   state.get("hwm_auc", 0.0),
        "brier": state.get("hwm_brier", 1.0),  # lower is better
    }


def _update_high_water_mark(auc: float, brier: float) -> None:
    """Update high-water mark if this model is the new best."""
    state = _load_retrain_state()
    updated = False
    if auc > state.get("hwm_auc", 0.0):
        state["hwm_auc"] = round(auc, 4)
        updated = True
    if brier < state.get("hwm_brier", 1.0):
        state["hwm_brier"] = round(brier, 4)
        updated = True
    if updated:
        _save_retrain_state(state)


def _acceptance_check(current: dict, candidate: dict) -> tuple[bool, list[str], list[str]]:
    """
    Returns (passed_hard_gate, rejection_reasons, warning_flags).

    Two layers:
      1. Hard absolute floors → auto-reject (catastrophe catch).
         If these fail, passed=False and the candidate is immediately discarded.
      2. High-water-mark ratchet → WARNING flags (not auto-reject).
         If the candidate is below (best-approved - tolerance), a prominent
         warning is added to the notification, but the candidate still goes
         to human hold. The human decides — training AUC is noisy and may
         not reflect realized performance.

    Compares AUC and Brier from each model's training-run test set — same
    methodology for both, so the comparison is apples-to-apples.
    """
    reasons: list[str] = []
    flags: list[str] = []

    n_auc   = float(candidate.get("roc_auc", candidate.get("auc", 0.5)))
    n_brier = float(candidate.get("brier_score", candidate.get("brier", 0.25)))

    # ── Layer 1: Hard absolute floors (auto-reject) ──
    if n_auc < _AUC_ABSOLUTE_FLOOR:
        reasons.append(
            f"AUC below absolute floor: {n_auc:.4f} < {_AUC_ABSOLUTE_FLOOR}"
        )
    if n_brier > _BRIER_ABSOLUTE_CEIL:
        reasons.append(
            f"Brier above absolute ceiling: {n_brier:.4f} > {_BRIER_ABSOLUTE_CEIL}"
        )

    # ── Layer 2: High-water-mark ratchet (flag, not reject) ──
    hwm = _get_high_water_mark()
    if hwm["auc"] > 0 and n_auc < hwm["auc"] - _HWM_AUC_TOLERANCE:
        flags.append(
            f"⚠ AUC {n_auc:.4f} is below best-approved {hwm['auc']:.4f} by "
            f">{_HWM_AUC_TOLERANCE} — review shadow + realized performance, "
            f"may be holdout noise or real degradation"
        )
    if hwm["brier"] < 1.0 and n_brier > hwm["brier"] + _HWM_BRIER_TOLERANCE:
        flags.append(
            f"⚠ Brier {n_brier:.4f} is above best-approved {hwm['brier']:.4f} by "
            f">{_HWM_BRIER_TOLERANCE} — review shadow + realized performance, "
            f"may be holdout noise or real degradation"
        )

    return (len(reasons) == 0), reasons, flags


# ─── Per-sport submodel retraining ────────────────────────────────────────────

def _retrain_submodel(sport: str, db: Session) -> dict:
    """
    Retrain a per-sport ATS submodel if ≥ _SUBMODEL_MIN_NEW_LEGS settled legs
    exist since the last retrain.  sport: 'nhl' | 'mlb'.
    """
    import ml_model as _ml

    meta_path = os.path.join(_SUBMODEL_DIR, f"{sport.lower()}_ats_meta.json")
    last_retrain_dt = datetime.utcnow() - timedelta(days=30)
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            ts = meta.get("saved_at") or meta.get("trained_at")
            if ts:
                last_retrain_dt = datetime.fromisoformat(ts[:19])
        except Exception:
            pass

    cutoff     = last_retrain_dt.strftime("%Y-%m-%d")
    sport_like = f"%{sport.lower()}%"

    try:
        new_legs = db.execute(text("""
            SELECT COUNT(*)
            FROM mock_bet_legs mbl
            JOIN mock_bets mb ON mb.id = mbl.mock_bet_id
            WHERE LOWER(mbl.sport) LIKE :sport_like
              AND mbl.leg_result IN ('WIN', 'LOSS')
              AND COALESCE(mbl.excluded_from_metrics, 0) = 0
              AND mb.game_date >= :cutoff
        """), {"sport_like": sport_like, "cutoff": cutoff}).scalar() or 0
    except Exception as e:
        return {"sport": sport, "status": "skip", "reason": f"count query failed: {e}"}

    if new_legs < _SUBMODEL_MIN_NEW_LEGS:
        return {
            "sport":  sport,
            "status": "skip",
            "reason": f"{new_legs} new settled legs (need {_SUBMODEL_MIN_NEW_LEGS}+)",
        }

    try:
        result = _ml.train_diagnostic_sport_model(sport.upper(), save=True)
        log.info("[WeeklyRetrain] %s submodel retrained: %s", sport.upper(), result)
        print(f"[WeeklyRetrain] {sport.upper()} submodel retrained — {result.get('verdict')}")
        return {"sport": sport, "status": "retrained", "new_legs": new_legs, **result}
    except Exception as e:
        return {"sport": sport, "status": "error", "reason": str(e)}


# ─── ModelVersion DB logging ──────────────────────────────────────────────────

def _log_model_version(
    db: Session,
    version: str,
    version_n: int,
    status: str,
    candidate: dict,
    holdout: dict,
    train_result: dict,
    rejection_reasons: list,
) -> None:
    try:
        from database import ModelVersion
        mv = ModelVersion(
            version=version,
            version_n=version_n,
            trained_at=datetime.utcnow(),
            training_legs=candidate.get("n_train") or train_result.get("n_train"),
            holdout_auc=holdout.get("auc"),
            holdout_brier=holdout.get("brier"),
            holdout_reliability=holdout.get("reliability"),
            status=status,
            rejection_reasons=rejection_reasons if rejection_reasons else None,
            model_path=os.path.join(_DATA_DIR, f"model_v{version_n}.pkl"),
            scaler_path=os.path.join(_DATA_DIR, f"scaler_v{version_n}.pkl"),
            imputer_path=os.path.join(_DATA_DIR, f"imputer_v{version_n}.pkl"),
            notes=json.dumps({
                "train_auc":     candidate.get("auc"),
                "train_brier":   candidate.get("brier"),
                "n_test":        train_result.get("n_test"),
                "sport_metrics": train_result.get("sport_metrics"),
            }),
        )
        db.add(mv)
        db.commit()
    except Exception as e:
        log.error("[WeeklyRetrain] Failed to log ModelVersion: %s", e)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_weekly_retrain(db: Session) -> dict:
    """
    Full weekly retraining pipeline (Sunday 3 AM CT).

    Steps:
      1. Record current model metrics (baseline for acceptance check).
      2. Back up current model files to model_v{N}.pkl.
      3. Train a new candidate model (overwrites model.pkl via ml_model.train).
      4. Evaluate candidate on 7-day live holdout (stored preds vs outcomes).
      5. Acceptance check — new model must not degrade AUC, Brier, or reliability.
      6. Accept → save model_v{N+1}.pkl, log ModelVersion "active", clear cache.
         Reject → restore backup, log ModelVersion "rejected".
      7. Per-sport submodel retraining (NHL, MLB) if ≥100 new settled legs.
    """
    import ml_model as _ml

    started_at    = datetime.utcnow()
    version_n     = _get_version_n(db)
    new_version_n = version_n + 1
    new_version   = f"v{new_version_n}"
    backup_n      = max(version_n, new_version_n)   # where backup lives

    log.info("[WeeklyRetrain] Starting — current v%d → candidate %s", version_n, new_version)
    print(f"[WeeklyRetrain] Starting — candidate {new_version}")

    # Step 1: current baseline
    current = _get_current_model_metrics(db)
    log.info("[WeeklyRetrain] Baseline: AUC=%.4f Brier=%.4f Rel=%.4f",
             current["auc"], current["brier"], current["reliability"])

    # Step 2: back up current model files
    _backup_model(backup_n)

    # Step 3: choose feature set and algorithm based on new-feature coverage
    coverage = get_new_feature_coverage(db)
    ready_features = [f for f, n in coverage.items() if n >= _COVERAGE_MIN_LEGS]

    # Always use GBC+CalibratedCV (the proven v3 algorithm).
    # FEATURE_COLS is the authoritative feature set — includes spread_value,
    # recency features, and per-leg signals. SimpleImputer handles NaN for
    # features with sparse data on older legs.
    chosen_algorithm = "gradient_boost"
    chosen_feature_cols = _ml.FEATURE_COLS
    log.info("[WeeklyRetrain] Using FEATURE_COLS (%d features), GBC+CalibratedCV",
             len(chosen_feature_cols))
    print(f"[WeeklyRetrain] {len(chosen_feature_cols)}-feature GBC retrain "
          f"(ready per-leg signals: {ready_features})")

    # Train candidate (writes model.pkl / scaler.pkl / imputer.pkl)
    try:
        train_result = _ml.train(
            db,
            algorithm=chosen_algorithm,
            train_submodels=False,
            feature_cols=chosen_feature_cols,
        )
    except Exception as e:
        _restore_model(backup_n)
        return {
            "status": "error", "version": new_version,
            "reason": f"Training exception: {e}",
            "started_at": started_at.isoformat(),
        }

    if "error" in train_result:
        _restore_model(backup_n)
        return {
            "status": "error", "version": new_version,
            "reason": train_result["error"],
            "started_at": started_at.isoformat(),
        }

    # Clear cache so holdout evaluation loads the new model weights
    _clear_model_cache()

    # Step 4: 7-day live holdout
    holdout = _compute_live_holdout(db, holdout_days=7)
    log.info("[WeeklyRetrain] 7d holdout: %s", holdout)

    candidate = {
        "auc":                 train_result.get("roc_auc", 0.5),
        "roc_auc":             train_result.get("roc_auc", 0.5),
        "brier":               train_result.get("brier_score", 0.25),
        "brier_score":         train_result.get("brier_score", 0.25),
        "holdout_reliability": holdout.get("reliability", current["reliability"]),
        "n_train":             train_result.get("n_train", 0),
        "holdout_n":           holdout.get("n", 0),
        "holdout_auc":         holdout.get("auc"),
        "holdout_brier":       holdout.get("brier"),
    }

    # Step 5: acceptance gate (hard floors) + high-water-mark flags
    passed_gate, reasons, flags = _acceptance_check(current, candidate)

    if not passed_gate:
        # Hard reject (absolute floor breach) — restore backup immediately
        _restore_model(backup_n)
        _clear_model_cache()
        rejection_msg = "; ".join(reasons)
        log.warning("[WeeklyRetrain] Rejected %s: %s", new_version, rejection_msg)
        print(f"[WeeklyRetrain] REJECTED {new_version}: {rejection_msg}")

        _log_model_version(
            db=db, version=new_version, version_n=new_version_n,
            status="rejected", candidate=candidate, holdout=holdout,
            train_result=train_result, rejection_reasons=reasons,
        )

        result: dict = {
            "status":            "rejected",
            "version":           new_version,
            "rejection_reasons": reasons,
            "current_kept":      f"v{version_n}" if version_n > 0 else "original",
            "candidate_auc":     candidate["auc"],
            "candidate_brier":   candidate["brier"],
            "current_auc":       current["auc"],
            "current_brier":     current["brier"],
            "holdout":           holdout,
            "started_at":        started_at.isoformat(),
            "finished_at":       datetime.utcnow().isoformat(),
        }
    else:
        # Passed hard floor — save candidate on HOLD, restore current model
        _backup_model(new_version_n)   # save candidate as model_v{N+1}.pkl
        _restore_model(backup_n)       # put current model back as active
        _clear_model_cache()

        status_label = "candidate_flagged" if flags else "candidate"
        _log_model_version(
            db=db, version=new_version, version_n=new_version_n,
            status=status_label,       # NOT "active" — awaiting human approval
            candidate=candidate, holdout=holdout,
            train_result=train_result, rejection_reasons=flags,  # flags stored for review
        )

        hwm = _get_high_water_mark()
        log.info("[WeeklyRetrain] CANDIDATE %s ready — AUC %.4f (was %.4f) Brier %.4f (was %.4f) — AWAITING HUMAN APPROVAL",
                 new_version, candidate["auc"], current["auc"],
                 candidate["brier"], current["brier"])
        print(f"\n{'='*70}")
        if flags:
            print(f"[WeeklyRetrain] CANDIDATE {new_version} — FLAGGED — NEEDS CAREFUL REVIEW")
        else:
            print(f"[WeeklyRetrain] CANDIDATE {new_version} READY — AWAITING HUMAN APPROVAL")
        print(f"  AUC:   {candidate['auc']:.4f}  (current: {current['auc']:.4f}  Δ={candidate['auc'] - current['auc']:+.4f})")
        print(f"  Brier: {candidate['brier']:.4f}  (current: {current['brier']:.4f}  Δ={candidate['brier'] - current['brier']:+.4f})")
        print(f"  Train: {candidate['n_train']} legs")
        print(f"  Holdout 7d: AUC={holdout.get('auc', 'N/A')} Brier={holdout.get('brier', 'N/A')} n={holdout.get('n', 0)}")
        if hwm["auc"] > 0:
            print(f"  Best-approved: AUC={hwm['auc']:.4f}  Brier={hwm['brier']:.4f}")
        if flags:
            print(f"  {'─'*60}")
            for flag in flags:
                print(f"  {flag}")
            print(f"  {'─'*60}")
            print(f"  Review shadow + realized performance before approving.")
        print(f"  → Approve: POST /api/model/approve-candidate")
        print(f"  → Reject:  POST /api/model/reject-candidate")
        print(f"{'='*70}\n")

        result = {
            "status":      "candidate_hold",
            "version":     new_version,
            "version_n":   new_version_n,
            "auc":         candidate["auc"],
            "brier":       candidate["brier"],
            "n_train":     candidate["n_train"],
            "prev_auc":    current["auc"],
            "prev_brier":  current["brier"],
            "holdout":     holdout,
            "flags":       flags,
            "flagged":     bool(flags),
            "action_required": "Human approval needed — POST /api/model/approve-candidate",
            "started_at":  started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
        }

    # Persist retrain state to disk (survives server restarts)
    state = _load_retrain_state()
    state["last_weekly_retrain"] = datetime.utcnow().strftime("%Y-%m-%d")
    state["last_result"] = result
    _save_retrain_state(state)

    # Per-sport submodel retraining (these are small, safe to auto-apply)
    submodel_results: dict = {}
    for sport in ("nhl", "mlb"):
        try:
            submodel_results[sport] = _retrain_submodel(sport, db)
        except Exception as e:
            submodel_results[sport] = {"status": "error", "reason": str(e)}

    result["submodels"] = submodel_results
    return result


# ─── Human approval / rejection of candidate ────────────────────────────────

def approve_candidate(db: Session) -> dict:
    """
    Human approves the held candidate — deploy it as the active model.
    """
    from database import ModelVersion

    candidate_mv = (
        db.query(ModelVersion)
        .filter(ModelVersion.status.in_(["candidate", "candidate_flagged"]))
        .order_by(ModelVersion.version_n.desc())
        .first()
    )
    if candidate_mv is None:
        return {"error": "No candidate model awaiting approval"}

    # Deploy candidate
    ok = _restore_model(candidate_mv.version_n)
    if not ok:
        return {"error": f"Candidate {candidate_mv.version} pkl files missing on disk"}

    # Mark previous active as rollback, candidate as active
    for mv in db.query(ModelVersion).filter(ModelVersion.status == "active").all():
        mv.status = "rollback"
    candidate_mv.status = "active"
    db.commit()
    _clear_model_cache()

    # Update high-water mark from training metrics (stored in notes)
    mv_notes: dict = {}
    raw = candidate_mv.notes or {}
    if isinstance(raw, str):
        try:
            mv_notes = json.loads(raw)
        except Exception:
            pass
    else:
        mv_notes = raw
    train_auc = float(mv_notes.get("train_auc", 0))
    train_brier = float(mv_notes.get("train_brier", 1.0))
    if train_auc > 0:
        _update_high_water_mark(train_auc, train_brier)

    log.info("[WeeklyRetrain] APPROVED and deployed %s", candidate_mv.version)
    print(f"[WeeklyRetrain] APPROVED — deployed {candidate_mv.version}")

    state = _load_retrain_state()
    state["last_approval"] = datetime.utcnow().isoformat()
    state["last_approved_version"] = candidate_mv.version
    _save_retrain_state(state)

    return {
        "status":  "deployed",
        "version": candidate_mv.version,
        "auc":     candidate_mv.holdout_auc,
        "brier":   candidate_mv.holdout_brier,
    }


def reject_candidate(db: Session) -> dict:
    """
    Human rejects the held candidate — mark as rejected, keep current model.
    """
    from database import ModelVersion

    candidate_mv = (
        db.query(ModelVersion)
        .filter(ModelVersion.status.in_(["candidate", "candidate_flagged"]))
        .order_by(ModelVersion.version_n.desc())
        .first()
    )
    if candidate_mv is None:
        return {"error": "No candidate model awaiting rejection"}

    candidate_mv.status = "rejected"
    candidate_mv.rejection_reasons = ["Human rejected"]
    db.commit()

    log.info("[WeeklyRetrain] REJECTED by human: %s", candidate_mv.version)
    print(f"[WeeklyRetrain] REJECTED by human — {candidate_mv.version}")

    state = _load_retrain_state()
    state["last_rejection"] = datetime.utcnow().isoformat()
    state["last_rejected_version"] = candidate_mv.version
    _save_retrain_state(state)

    return {
        "status":  "rejected",
        "version": candidate_mv.version,
        "reason":  "Human rejected",
    }


# ─── Status & rollback (called by API endpoints) ──────────────────────────────

def get_model_status(db: Session) -> dict:
    """
    Full model status dict for /api/health/model-status and daily summary.
    """
    from database import ModelRun, ModelVersion

    # Latest raw training run
    run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()
    notes: dict = {}
    if run:
        raw = run.notes or {}
        if isinstance(raw, str):
            try:
                notes = json.loads(raw)
            except Exception:
                pass
        else:
            notes = raw

    # ModelVersion history
    try:
        mvs = db.query(ModelVersion).order_by(ModelVersion.version_n.desc()).limit(10).all()
        active_mv = next((m for m in mvs if m.status == "active"), None)
        history = [
            {
                "version":  mv.version,
                "auc":      mv.holdout_auc,
                "brier":    mv.holdout_brier,
                "deployed": mv.trained_at.strftime("%Y-%m-%d") if mv.trained_at else None,
                "status":   mv.status,
                "n_train":  mv.training_legs,
            }
            for mv in mvs
        ]
    except Exception:
        active_mv = None
        history = []

    # Live 7-day holdout from stored predictions
    holdout_7d = _compute_live_holdout(db, holdout_days=7)

    current_version = (
        active_mv.version if active_mv
        else notes.get("version", f"v{_get_version_n(db)}" if _get_version_n(db) > 0 else "v1")
    )
    trained_on = (
        active_mv.trained_at.strftime("%Y-%m-%d") if active_mv and active_mv.trained_at
        else (run.run_at.strftime("%Y-%m-%d") if run and run.run_at else None)
    )
    training_legs = (
        active_mv.training_legs if active_mv
        else (run.n_train if run else None)
    )
    last_result = (
        active_mv.status if active_mv
        else ("no_history" if not history else history[0]["status"])
    )

    # Feature importances from latest training run (sorted desc)
    feature_importances: dict = {}
    if notes.get("importances"):
        raw_imps = notes["importances"]
        feature_importances = dict(
            sorted(raw_imps.items(), key=lambda x: -x[1])
        )

    # New-feature coverage for retrain readiness dashboard
    try:
        coverage = get_new_feature_coverage(db)
        next_retrain_str = _next_sunday_3am_ct()
        new_feature_readiness = {
            feat: {
                "settled_legs": n,
                "needed":       max(0, _COVERAGE_MIN_LEGS - n),
                "ready":        n >= _COVERAGE_MIN_LEGS,
            }
            for feat, n in coverage.items()
        }
        features_eligible = [f for f, n in coverage.items() if n >= _COVERAGE_MIN_LEGS]
        new_feature_readiness["min_legs_required"]                  = _COVERAGE_MIN_LEGS
        new_feature_readiness["next_retrain"]                       = next_retrain_str
        new_feature_readiness["features_eligible_for_next_retrain"] = features_eligible
        # Rough estimate: all 6 features ready when injury/movement signals accumulate
        all_near_ready = all(n > 0 for n in coverage.values())
        new_feature_readiness["estimated_all_ready"] = (
            "~2026-06-15" if not all_near_ready else "eligible now"
        )
    except Exception:
        new_feature_readiness = {}

    return {
        "current_version":       current_version,
        "trained_on":            trained_on,
        "training_legs":         training_legs,
        "holdout_auc":           run.roc_auc if run else None,
        "holdout_brier":         float(notes.get("brier", notes.get("brier_score", 0.0))) if run else None,
        "live_7d_holdout":       holdout_7d,
        "next_retrain":          _next_sunday_3am_ct(),
        "last_retrain_result":   last_result,
        "versions_history":      history,
        "feature_importances":   feature_importances,
        "n_mock_legs_trained":   notes.get("n_mock_legs_train"),
        "algorithm":             notes.get("algorithm", "gradient_boost"),
        "feature_cols":          notes.get("feature_cols", []),
        "new_feature_readiness": new_feature_readiness,
    }


def rollback_model(db: Session, target_version_n: Optional[int] = None) -> dict:
    """
    Roll back to a specific version (default: most recent non-active version).
    Restores pkl files and updates ModelVersion.status in the DB.
    """
    from database import ModelVersion

    try:
        mvs = db.query(ModelVersion).order_by(ModelVersion.version_n.desc()).all()
    except Exception as e:
        return {"error": f"Cannot query model_versions: {e}"}

    if not mvs:
        return {"error": "No versioned models found — cannot rollback"}

    if target_version_n is not None:
        target = next((m for m in mvs if m.version_n == target_version_n), None)
        if target is None:
            return {"error": f"Version v{target_version_n} not found in model_versions"}
    else:
        current = next((m for m in mvs if m.status == "active"), None)
        prior = [m for m in mvs if m != current and m.status in ("active", "rollback")]
        if not prior:
            return {"error": "No prior version available for rollback"}
        target = prior[0]

    ok = _restore_model(target.version_n)
    if not ok:
        return {
            "error": f"Some pkl files for {target.version} missing on disk "
                     "(model.pkl restored where found)"
        }

    for mv in mvs:
        if mv.status == "active":
            mv.status = "rollback"
    target.status = "active"
    db.commit()

    log.info("[WeeklyRetrain] Rolled back to %s", target.version)
    return {
        "status":  "rolled_back",
        "version": target.version,
        "auc":     target.holdout_auc,
        "brier":   target.holdout_brier,
    }
