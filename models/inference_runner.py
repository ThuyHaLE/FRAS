"""
Standalone inference — only need this file and the inference checkpoint.
No need for horizon_trainer.py, survival_trainer.py, base_trainer.py.

Usage:
    from inference_runner import predict

    submission = predict(
        test_df         = test_df,
        checkpoint_path = "checkpoints/inference.pkl",
    )
    submission.to_csv("submission.csv", index=False)
"""

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# Utility (copy from horizon_trainer.py and survival_trainer.py)
# ──────────────────────────────────────────
def _enforce_monotone(probs: np.ndarray) -> np.ndarray:
    """P_12h <= P_24h <= P_48h <= P_72h row-wise."""
    return np.clip(np.maximum.accumulate(probs, axis=1), 0.0, 1.0)

def _apply_temp(prob: np.ndarray, T: float) -> np.ndarray:
    """Temperature scaling calibration."""
    from scipy.special import expit, logit
    return expit(logit(np.clip(prob, 1e-6, 1 - 1e-6)) / T)

# ──────────────────────────────────────────
# HorizonTrainer inference
# ──────────────────────────────────────────
def _predict_horizon(payload: dict, test_df: pd.DataFrame) -> np.ndarray:
    horizons = payload["h_horizons"]
    raw      = {}

    for h in horizons:
        info    = payload["h_trained"][h]
        X       = test_df[info["features"]].values
        raw[h]  = info["model"].predict_proba(X)[:, 1]

    keys = sorted(raw)
    mat  = np.stack([raw[k] for k in keys], axis=1)
    return _enforce_monotone(mat)

# ──────────────────────────────────────────
# SurvivalTrainer inference
# ──────────────────────────────────────────
def _predict_survival(payload: dict, test_df: pd.DataFrame) -> np.ndarray:
    feature_cols = payload["s_feature_cols"]
    horizons     = payload["s_horizons"]

    # Preprocess
    X = test_df[feature_cols].copy()
    X = X.clip(lower=payload["s_clip_bounds"][0], upper=payload["s_clip_bounds"][1], axis=1)
    X = X.fillna(payload["s_medians"])
    X = payload["s_scaler"].transform(X)

    # Survival → hit probs
    sfns  = payload["s_rsf_model"].predict_survival_function(X)
    raw_s = np.zeros((len(test_df), len(horizons)))

    for i, sfn in enumerate(sfns):
        domain_max = sfn.domain[1]
        for j, t in enumerate(horizons):
            t_clamped   = min(t, domain_max)
            s_t         = float(np.clip(sfn(t_clamped), 0.0, 1.0))
            raw_s[i, j] = 1.0 - s_t

    raw_s = _enforce_monotone(raw_s)

    # Calibration
    cal_s = np.zeros_like(raw_s)
    for j, T in enumerate(payload["s_calibrators"]):
        cal_s[:, j] = _apply_temp(raw_s[:, j], T)

    return _enforce_monotone(cal_s)

# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────
def predict(
    test_df: pd.DataFrame,
    checkpoint_path: str,
    id_col: str = "event_id",
) -> pd.DataFrame:
    """
    Predict from an inference checkpoint.

    Parameters
    ----------
    test_df         : pd.DataFrame — test data
    checkpoint_path : str          — path to the .pkl file from save_inference_checkpoint()
    id_col          : str          — name of the ID column (default: 'event_id')

    Returns
    -------
    pd.DataFrame with columns: event_id, prob_h12, prob_h24, prob_h48, prob_h72

    Example
    -------
    >>> from inference_runner import predict
    >>> submission = predict(test_df, "models/checkpoints/inference.pkl")
    >>> submission.to_csv("submission.csv", index=False)
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib chưa được cài. Chạy: pip install joblib")

    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

    payload  = joblib.load(checkpoint_path)
    horizons = payload["horizons"]

    # Predict từ 2 model
    test_h = _predict_horizon(payload, test_df)
    test_s = _predict_survival(payload, test_df)

    # Blend theo best_weights
    test_ens = np.zeros_like(test_h)
    for j, w in enumerate(payload["best_weights"]):
        test_ens[:, j] = np.clip(
            w * test_h[:, j] + (1 - w) * test_s[:, j], 0.0, 1.0
        )
    test_ens = _enforce_monotone(test_ens)

    # Build submission
    submission = pd.DataFrame({id_col: test_df[id_col].values})
    for j, h in enumerate(horizons):
        submission[f"prob_h{h}"] = np.round(test_ens[:, j], 6)

    return submission