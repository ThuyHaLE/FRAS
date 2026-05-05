# models/inference_runner.py

"""
Standalone inference — supports SurvivalTrainer and EnsembleTrainer checkpoints.

Both trainer types expose the same public API used here:
    trainer.predict_raw(df)      → raw hit probabilities (N, K)
    trainer.calibrate(raw, ...)  → calibrated probabilities + calibrators

Usage
-----
    from inference_runner import predict

    submission = predict(
        test_df         = test_df,
        checkpoint_path = "checkpoints/ensemble.pkl",
    )
    submission.to_csv("submission.csv", index=False)
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from features import TARGET_EVENT_ID


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _enforce_monotone(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.maximum.accumulate(probs, axis=1), 0.0, 1.0)


def _oof_distribution_check(
    raw_probs: np.ndarray,
    oof_raw: Optional[np.ndarray],
    horizons: List[int],
    tag: str = "",
) -> None:
    """Warn if test distribution has shifted significantly from OOF."""
    if oof_raw is None:
        return
    prefix = f"[{tag}] " if tag else ""
    for j, h in enumerate(horizons):
        ratio = raw_probs[:, j].mean() / (oof_raw[:, j].mean() + 1e-8)
        if ratio < 0.7 or ratio > 1.4:
            warnings.warn(
                f"{prefix}@{h}h: test/OOF ratio={ratio:.2f} — "
                "distribution shift detected, calibration may be off.",
                stacklevel=3,
            )


# ──────────────────────────────────────────────────────────────
# SurvivalTrainer predict path
# ──────────────────────────────────────────────────────────────

def _predict_from_survival_trainer(
    test_df: pd.DataFrame,
    trainer,                  # SurvivalTrainer
    verbose: bool,
) -> tuple[np.ndarray, List[int]]:
    """
    Uses only the public API of SurvivalTrainer:
        trainer.predict_raw(df)          → raw probs (already preprocessed)
        trainer.calibrate(raw, fit=False) → calibrated probs
    """
    if not trainer.is_fitted():
        raise RuntimeError(
            "SurvivalTrainer checkpoint is not fully fitted "
            "(missing model or calibrators)."
        )

    if verbose:
        print(
            f"[inference] SurvivalTrainer  strategy={trainer.strategy.name!r}  "
            f"features={len(trainer.feature_cols)}  rows={len(test_df)}"
        )

    raw = trainer.predict_raw(test_df, warn_extrapolation=True)
    _oof_distribution_check(raw, trainer.oof_raw, trainer.horizons, tag=trainer.strategy.name)

    cal, _ = trainer.calibrate(raw, fit=False)
    cal    = _enforce_monotone(cal)

    return cal, trainer.horizons


# ──────────────────────────────────────────────────────────────
# EnsembleTrainer predict path
# ──────────────────────────────────────────────────────────────

def _predict_from_ensemble_trainer(
    test_df: pd.DataFrame,
    ensemble,                 # EnsembleTrainer
    verbose: bool,
) -> tuple[np.ndarray, List[int]]:
    """
    Uses only the public API of EnsembleTrainer:
        sub_trainer.predict_raw(df)          → raw probs per sub-model
        ensemble._blend(raw_list)            → weighted blend
        ensemble._apply_ensemble_calibrator  → final calibrated probs
    """
    if not ensemble.is_fitted():
        raise RuntimeError(
            "EnsembleTrainer checkpoint is not fully fitted. "
            "All sub-trainers and ensemble calibrator must be fitted."
        )

    tags     = [tag for tag, _ in ensemble._trainers]
    horizons = ensemble.horizons

    if verbose:
        print(
            f"[inference] EnsembleTrainer  models={tags}  "
            f"weights={ensemble.ensemble_weights}  rows={len(test_df)}"
        )

    raw_list = []
    for tag, trainer in ensemble._trainers:
        if verbose:
            print(f"  [{tag}] predicting raw...")

        raw = trainer.predict_raw(test_df, warn_extrapolation=True)
        _oof_distribution_check(raw, trainer.oof_raw, horizons, tag=tag)
        raw_list.append(raw)

    raw_blended = ensemble._blend(raw_list)
    cal_probs   = ensemble._apply_ensemble_calibrator(raw_blended)
    cal_probs   = _enforce_monotone(cal_probs)

    return cal_probs, horizons


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def predict(
    test_df: pd.DataFrame,
    checkpoint_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a SurvivalTrainer or EnsembleTrainer checkpoint
    and produce a submission DataFrame.

    Parameters
    ----------
    test_df         : DataFrame with feature columns and ID column.
    checkpoint_path : Path to .pkl checkpoint saved by save_checkpoint().
    verbose         : Print progress and distribution summary.

    Returns
    -------
    DataFrame with columns [TARGET_EVENT_ID, prob_12h, prob_24h, prob_48h, prob_72h].
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required: pip install joblib")

    if verbose:
        print(f"[inference] Loading checkpoint ← {checkpoint_path}")

    ckpt      = joblib.load(checkpoint_path)
    ckpt_type = type(ckpt).__name__

    # ── Route by checkpoint type ──────────────────────────────
    if ckpt_type == "EnsembleTrainer":
        cal_probs, horizons = _predict_from_ensemble_trainer(test_df, ckpt, verbose)

    elif ckpt_type == "SurvivalTrainer":
        cal_probs, horizons = _predict_from_survival_trainer(test_df, ckpt, verbose)

    else:
        raise TypeError(
            f"Unsupported checkpoint type: {ckpt_type!r}. "
            "Expected 'SurvivalTrainer' or 'EnsembleTrainer'."
        )

    # ── Build output DataFrame ────────────────────────────────
    prob_cols = [f"prob_{h}h" for h in horizons]
    df_sub    = pd.DataFrame({TARGET_EVENT_ID: test_df[TARGET_EVENT_ID].values})
    for j, col in enumerate(prob_cols):
        df_sub[col] = np.round(cal_probs[:, j], 6)

    # ── Sanity assertions ─────────────────────────────────────
    assert (df_sub[prob_cols].values >= 0).all(),                             "Negative probabilities"
    assert (df_sub[prob_cols].values <= 1).all(),                             "Probabilities > 1"
    assert (df_sub[prob_cols].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all(), "Monotonicity violated"

    if verbose:
        print(f"\n[inference] Done — {len(df_sub)} rows")
        print(df_sub[prob_cols].describe().round(4).to_string())

    return df_sub