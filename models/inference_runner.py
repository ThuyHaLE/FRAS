# models/inference_runner.py

"""
Standalone inference — supports SurvivalTrainer and EnsembleTrainer checkpoints.

Usage
-----
    from inference_runner import predict

    submission = predict(
        test_df           = test_df,
        checkpoint_path   = "checkpoints/ensemble.pkl",
    )
    submission.to_csv("submission.csv", index=False)
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────
def _enforce_monotone(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.maximum.accumulate(probs, axis=1), 0.0, 1.0)

def _apply_temp(p: np.ndarray, T: float) -> np.ndarray:
    from scipy.special import expit, logit
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / T)

def _survival_to_hit_probs(
    survival_fns,
    horizons: List[int],
    warn_extrapolation: bool = True,
) -> np.ndarray:
    probs = np.zeros((len(survival_fns), len(horizons)))
    extrapolated: set = set()
    for i, sfn in enumerate(survival_fns):
        domain_max = sfn.domain[1]
        for j, t in enumerate(horizons):
            if t > domain_max:
                extrapolated.add(t)
            s_t = float(np.clip(sfn(min(t, domain_max)), 0.0, 1.0))
            probs[i, j] = 1.0 - s_t
    if warn_extrapolation and extrapolated:
        warnings.warn(
            f"Horizons {sorted(extrapolated)}h exceed survival function domain.",
            stacklevel=3,
        )
    return probs

def _preprocess(
    df: pd.DataFrame,
    feature_cols: List[str],
    clip_bounds: tuple,
    medians: pd.Series,
    scaler: Any,
) -> np.ndarray:
    X = df[feature_cols].copy()
    X = X.clip(lower=clip_bounds[0], upper=clip_bounds[1], axis=1)
    X = X.fillna(medians)
    return scaler.transform(X)

def _raw_probs_single(
    X: np.ndarray,
    model: Any,
    horizons: List[int],
) -> np.ndarray:
    """Raw hit probs from model (RSF or GradBoost)."""
    return _enforce_monotone(
        _survival_to_hit_probs(model.predict_survival_function(X), horizons)
    )

def _calibrate(raw_probs: np.ndarray, calibrators: List[float]) -> np.ndarray:
    cal = np.zeros_like(raw_probs)
    for j, T in enumerate(calibrators):
        cal[:, j] = _apply_temp(raw_probs[:, j], T)
    return cal

def _oof_distribution_check(
    raw_probs: np.ndarray,
    oof_raw: Optional[np.ndarray],
    horizons: List[int],
    tag: str = "",
) -> None:
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
# Checkpoint-specific predict paths
# ──────────────────────────────────────────────────────────────
def _predict_from_survival_trainer(
    test_df: pd.DataFrame,
    trainer: Any,
    verbose: bool,
) -> pd.DataFrame:
    """Path for SurvivalTrainer checkpoint."""
    feature_cols = trainer.feature_cols
    horizons     = trainer.horizons
    model_type   = trainer.model_type
    rsf_model    = getattr(trainer, "rsf_model",  None)
    gb_model     = getattr(trainer, "gb_model",   None)
    calibrators  = trainer.calibrators
    scaler       = trainer.scaler
    clip_bounds  = trainer._clip_bounds
    medians      = trainer._medians
    oof_raw      = getattr(trainer, "oof_raw", None)

    # Validate
    _missing = []
    if calibrators  is None: _missing.append("calibrators")
    if scaler       is None: _missing.append("scaler")
    if clip_bounds  is None: _missing.append("_clip_bounds")
    if medians      is None: _missing.append("_medians")
    if model_type == "rsf"      and rsf_model is None: _missing.append("rsf_model")
    if model_type == "gradboost" and gb_model  is None: _missing.append("gb_model")
    if _missing:
        raise RuntimeError(f"Checkpoint missing fitted state: {_missing}")

    if verbose:
        print(f"[inference] SurvivalTrainer  model_type={model_type!r}  "
              f"features={len(feature_cols)}  rows={len(test_df)}")

    X = _preprocess(test_df, feature_cols, clip_bounds, medians, scaler)
    model = rsf_model if model_type == "rsf" else gb_model
    raw   = _raw_probs_single(X, model, horizons)

    _oof_distribution_check(raw, oof_raw, horizons, tag=model_type.upper())

    cal_probs = _enforce_monotone(_calibrate(raw, calibrators))
    return cal_probs, horizons

def _predict_from_ensemble_trainer(
    test_df: pd.DataFrame,
    ensemble: Any,
    verbose: bool,
) -> Tuple[np.ndarray, List[int]]:
    """Path for EnsembleTrainer checkpoint."""
    horizons = ensemble.horizons
    w_rsf, w_gb = _normalized_weights(ensemble.ensemble_weights)

    results = {}
    for tag, sub_trainer in [("RSF", ensemble.rsf_trainer), ("GB", ensemble.gb_trainer)]:
        fc  = sub_trainer.feature_cols
        cb  = sub_trainer._clip_bounds
        med = sub_trainer._medians
        sc  = sub_trainer.scaler
        cal = sub_trainer.calibrators
        mdl = sub_trainer.rsf_model if sub_trainer.model_type == "rsf" else sub_trainer.gb_model
        oof = getattr(sub_trainer, "oof_raw", None)

        _missing = []
        if cal is None: _missing.append(f"{tag}.calibrators")
        if sc  is None: _missing.append(f"{tag}.scaler")
        if cb  is None: _missing.append(f"{tag}._clip_bounds")
        if med is None: _missing.append(f"{tag}._medians")
        if mdl is None: _missing.append(f"{tag}.model")
        if _missing:
            raise RuntimeError(f"Checkpoint missing fitted state: {_missing}")

        if verbose:
            print(f"[inference] {tag}  features={len(fc)}  rows={len(test_df)}")

        X   = _preprocess(test_df, fc, cb, med, sc)
        raw = _raw_probs_single(X, mdl, horizons)
        _oof_distribution_check(raw, oof, horizons, tag=tag)
        cal_probs = _enforce_monotone(_calibrate(raw, cal))
        results[tag] = cal_probs

    # Blend (ported từ EnsembleTrainer._blend)
    blended = _enforce_monotone(w_rsf * results["RSF"] + w_gb * results["GB"])
    return blended, horizons

def _normalized_weights(ensemble_weights: tuple) -> Tuple[float, float]:
    total = ensemble_weights[0] + ensemble_weights[1]
    return ensemble_weights[0] / total, ensemble_weights[1] / total

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
    and produce submission DataFrame.

    Parameters
    ----------
    test_df         : DataFrame include feature columns and ID column.
    checkpoint_path : Path to .pkl checkpoint.
    verbose         : Print progress and distribution summary.

    Returns
    -------
    DataFrame with columns [event_id, prob_12h, prob_24h, prob_48h, prob_72h].
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required: pip install joblib")

    if verbose:
        print(f"[inference] Loading checkpoint ← {checkpoint_path}")

    ckpt = joblib.load(checkpoint_path)

    # ── Detect checkpoint type và route ─────────────────────────
    ckpt_type = type(ckpt).__name__

    if ckpt_type == "EnsembleTrainer":
        id_col = getattr(ckpt.rsf_trainer, "TARGET_EVENT_ID", None)
        if verbose:
            print(f"[inference] Detected: EnsembleTrainer  "
                  f"weights={ckpt.ensemble_weights}  horizons={ckpt.horizons}")
        cal_probs, horizons = _predict_from_ensemble_trainer(test_df, ckpt, verbose)

    elif ckpt_type == "SurvivalTrainer":
        id_col = getattr(ckpt, "TARGET_EVENT_ID", None)
        if verbose:
            print(f"[inference] Detected: SurvivalTrainer")
        cal_probs, horizons = _predict_from_survival_trainer(test_df, ckpt, verbose)

    else:
        raise TypeError(
            f"Unsupported checkpoint type: {ckpt_type!r}. "
            "Expected 'SurvivalTrainer' or 'EnsembleTrainer'."
        )

    # ── Build output DataFrame ───────────────────────────────────
    from features import TARGET_EVENT_ID
    id_col = id_col or TARGET_EVENT_ID

    df_sub = pd.DataFrame({id_col: test_df[id_col].values})
    prob_cols = [f"prob_{h}h" for h in horizons]
    for j, col in enumerate(prob_cols):
        df_sub[col] = np.round(cal_probs[:, j], 6)

    # ── Sanity assertions ────────────────────────────────────────
    assert (df_sub[prob_cols].values >= 0).all(),                             "Negative probabilities"
    assert (df_sub[prob_cols].values <= 1).all(),                             "Probabilities > 1"
    assert (df_sub[prob_cols].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all(), "Monotonicity violated"

    if verbose:
        print(f"\n[inference] Done — {len(df_sub)} rows")
        print(df_sub[prob_cols].describe().round(4).to_string())

    return df_sub