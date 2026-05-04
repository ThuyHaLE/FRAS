from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from typing import Optional, List, Tuple, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from features import TARGET_EVENT_ID, TARGET_EVENT, TARGET_TIME, HORIZONS, BRIER_W, CINDEX_HORIZON, RANDOM_STATE
from models import _STRATEGY_REGISTRY, RSFStrategy, ModelStrategy

# ──────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────
def enforce_monotone(probs: np.ndarray) -> np.ndarray:
    """Enforce P_12h <= P_24h <= P_48h <= P_72h row-wise."""
    return np.clip(np.maximum.accumulate(probs, axis=1), 0.0, 1.0)


def brier_matrix(
    event: np.ndarray,
    time: np.ndarray,
    probs: np.ndarray,
    horizons: Optional[List[int]] = None,
) -> np.ndarray:
    """Vectorized Brier score for all horizons. Returns shape (K,)."""
    if horizons is None:
        horizons = HORIZONS
    event = np.asarray(event, dtype=float)
    time  = np.asarray(time,  dtype=float)
    H     = np.asarray(horizons, dtype=float)[None, :]   # (1, K)
    E     = event[:, None]                               # (N, 1)
    T     = time[:, None]                                # (N, 1)

    hit_before = (E == 1) & (T <= H)
    safe_at_h  = (E == 0) & (T >= H)
    hit_after  = (E == 1) & (T >  H)
    mask       = hit_before | safe_at_h | hit_after

    label      = hit_before.astype(float)
    se         = (label - probs) ** 2
    se[~mask]  = np.nan
    return np.nanmean(se, axis=0)


def make_binary_outcome_matrix(
    event: np.ndarray,
    time: np.ndarray,
    horizons: Optional[List[int]] = None,
) -> np.ndarray:
    """Binary outcome matrix with nan for censored-before-horizon."""
    if horizons is None:
        horizons = HORIZONS
    event = np.asarray(event)
    time  = np.asarray(time)
    out   = np.full((len(event), len(horizons)), np.nan)
    for j, t in enumerate(horizons):
        hit_before              = (event == 1) & (time <= t)
        safe_at_h               = (event == 0) & (time >= t)
        hit_after               = (event == 1) & (time >  t)
        out[hit_before, j]      = 1.0
        out[safe_at_h,  j]      = 0.0
        out[hit_after,  j]      = 0.0
    return out


# ──────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────
class _FoldState:
    __slots__ = ("scaler", "clip_lo", "clip_hi", "medians")

    def __init__(self, scaler, clip_lo, clip_hi, medians):
        self.scaler  = scaler
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.medians = medians


# ──────────────────────────────────────────
# SurvivalTrainer
# ──────────────────────────────────────────
class SurvivalTrainer:
    """
    Right-censored survival trainer.

    Two ways to specify model:
        # 1. String (backward compatible)
        trainer = SurvivalTrainer(feature_cols=cols, model_type="rsf")

        # 2. Inject strategy directly (recommended)
        trainer = SurvivalTrainer(
            feature_cols=cols,
            strategy=RSFStrategy(n_trees=300, max_features="log2"),
        )

    Adding new models: implement ModelStrategy, pass into strategy=.

    Pipeline
    --------
    cross_validate(train_df)  → OOF predictions + CV score
    fit(train_df)             → retrain full data + fit calibrators
    predict_test(test_df)     → calibrated hit probabilities
    save_checkpoint(path)     → persist fitted trainer
    """

    HORIZONS       = HORIZONS
    BRIER_W        = BRIER_W
    CINDEX_HORIZON = CINDEX_HORIZON
    RANDOM_STATE   = RANDOM_STATE
    N_SPLITS       = 5

    _PROTECTED_ATTRS = frozenset({
        "_model", "calibrators", "oof_raw", "oof_cal",
        "cv_score", "scaler", "_clip_bounds", "_medians",
    })

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        model_type: str = "rsf",
        strategy: Optional[ModelStrategy] = None,
        n_splits: int = N_SPLITS,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
        auto_tune: bool = True,
    ):
        self.feature_cols = feature_cols or []
        self.horizons     = horizons or HORIZONS
        self.n_splits     = n_splits
        self.random_state = random_state
        self.verbose      = verbose
        self.auto_tune    = auto_tune

        # Resolve strategy: inject > registry
        if strategy is not None:
            self.strategy   = strategy
            self.model_type = strategy.name.lower()
        else:
            if model_type not in _STRATEGY_REGISTRY:
                raise ValueError(
                    f"model_type must be one of {set(_STRATEGY_REGISTRY)}, "
                    f"got {model_type!r}"
                )
            self.strategy   = _STRATEGY_REGISTRY[model_type]()
            self.model_type = model_type

        # Update n_features for RSFStrategy if needed
        if isinstance(self.strategy, RSFStrategy):
            self.strategy.n_features = len(self.feature_cols)

        # State
        self._model: Any                   = None
        self.scaler: Optional[StandardScaler] = None
        self._clip_bounds                  = None
        self._medians                      = None
        self.calibrators: Optional[List[float]] = None
        self.oof_raw: Optional[np.ndarray] = None
        self.oof_cal: Optional[np.ndarray] = None
        self.cv_score: Optional[float]     = None
        self._auto_params_n: Optional[int] = None
        self._cidx_col: int = (
            self.horizons.index(CINDEX_HORIZON)
            if CINDEX_HORIZON in self.horizons else -1
        )

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────
    def _auto_params(self, n: int) -> None:
        if not self.auto_tune:
            return
        self.strategy.auto_params(n)
        self._auto_params_n = n
        if self.verbose:
            print(f"[auto_params] n={n}  {self.strategy.summary_str()}")

    def _prob_col(self, h: int) -> str:
        return f"prob_{h}h"

    def _prob_cols(self) -> List[str]:
        return [self._prob_col(h) for h in self.horizons]

    def is_fitted(self) -> bool:
        return self._model is not None and self.calibrators is not None

    def summary(self) -> None:
        print(f"SurvivalTrainer  strategy={self.strategy.name!r}  auto_tune={self.auto_tune}")
        print(f"  features : {len(self.feature_cols)}  horizons : {self.horizons}")
        print(f"  fitted   : {self.is_fitted()}  cv_score : {self.cv_score}")
        print(f"  {self.strategy.summary_str()}")
        if self._auto_params_n:
            print(f"  auto_params tuned for n={self._auto_params_n}")

    # ──────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib not installed.")
        if not self.is_fitted():
            raise RuntimeError("Not fitted yet. Call cross_validate() then fit().")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path, compress=3)
        if self.verbose:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"[checkpoint] Saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        override: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> "SurvivalTrainer":
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib not installed.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        trainer = joblib.load(path)
        if not isinstance(trainer, cls):
            warnings.warn(f"Checkpoint contains {type(trainer).__name__}.", UserWarning)

        if override:
            blocked = set(override) & cls._PROTECTED_ATTRS
            if blocked:
                raise ValueError(f"Cannot override fitted state: {sorted(blocked)}")
            for k, v in override.items():
                setattr(trainer, k, v)

        if verbose:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"[checkpoint] Loaded ← {path}  ({size_mb:.1f} MB)")
            print(f"[checkpoint] strategy={trainer.strategy.name!r}")
            if trainer.cv_score is not None:
                print(f"[checkpoint] CV Score={trainer.cv_score:.4f}")
        return trainer

    # ──────────────────────────────────────────
    # Guards
    # ──────────────────────────────────────────
    def _check_cv_done(self) -> None:
        if self.oof_raw is None:
            raise RuntimeError(
                "cross_validate() has not been run.\n"
                "Order: cross_validate() → fit() → save_checkpoint()"
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError(
                "Not fitted. Call cross_validate() + fit(), or load checkpoint."
            )

    # ──────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────
    def _preprocess(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[np.ndarray, Any]:
        if not fit and self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        X = df[self.feature_cols].copy()
        if fit:
            self._clip_bounds = (X.quantile(0.01), X.quantile(0.99))
            self._medians     = X.median()
            self.scaler       = StandardScaler()

        X        = X.clip(lower=self._clip_bounds[0], upper=self._clip_bounds[1], axis=1)
        X        = X.fillna(self._medians)
        X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

        y = (
            Surv.from_dataframe(TARGET_EVENT, TARGET_TIME, df)
            if (TARGET_EVENT in df.columns and TARGET_TIME in df.columns)
            else None
        )
        return X_scaled, y

    def _preprocess_fold(
        self,
        df_trn: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> Tuple[np.ndarray, Any, np.ndarray, _FoldState]:
        X_trn   = df_trn[self.feature_cols].copy()
        clip_lo = X_trn.quantile(0.01)
        clip_hi = X_trn.quantile(0.99)
        medians = X_trn.median()
        X_trn   = X_trn.clip(lower=clip_lo, upper=clip_hi, axis=1).fillna(medians)
        scaler  = StandardScaler()
        X_trn_s = scaler.fit_transform(X_trn)

        X_val   = df_val[self.feature_cols].copy()
        X_val_s = scaler.transform(
            X_val.clip(lower=clip_lo, upper=clip_hi, axis=1).fillna(medians)
        )
        y_trn = Surv.from_dataframe(TARGET_EVENT, TARGET_TIME, df_trn)
        return X_trn_s, y_trn, X_val_s, _FoldState(scaler, clip_lo, clip_hi, medians)

    # ──────────────────────────────────────────
    # Survival → hit probabilities
    # ──────────────────────────────────────────
    def _survival_to_hit_probs(
        self,
        survival_fns,
        warn_extrapolation: bool = False,
    ) -> np.ndarray:
        probs        = np.zeros((len(survival_fns), len(self.horizons)))
        extrapolated: set = set()
        for i, sfn in enumerate(survival_fns):
            domain_max = sfn.domain[1]
            for j, t in enumerate(self.horizons):
                if t > domain_max:
                    extrapolated.add(t)
                s_t            = float(np.clip(sfn(min(t, domain_max)), 0.0, 1.0))
                probs[i, j]    = 1.0 - s_t
        if warn_extrapolation and extrapolated:
            warnings.warn(
                f"Horizons {sorted(extrapolated)}h exceed the domain of the survival function.",
                stacklevel=2,
            )
        return probs

    # ──────────────────────────────────────────
    # Core: build + fit + predict via strategy
    # ──────────────────────────────────────────
    def _fit_model(self, X: np.ndarray, y: Any, verbose: bool = True) -> Any:
        if self.verbose and verbose:
            print(self.strategy.summary_str())
        model = self.strategy.build(random_state=self.random_state)
        return self.strategy.fit(model, X, y)

    def _predict_raw(
        self,
        X: np.ndarray,
        model: Any = None,
        warn_extrapolation: bool = False,
    ) -> np.ndarray:
        m            = model if model is not None else self._model
        survival_fns = self.strategy.predict_survival(m, X)
        return enforce_monotone(
            self._survival_to_hit_probs(survival_fns, warn_extrapolation)
        )

    # ── Public API (used by EnsembleTrainer) ────────────────
    def predict_raw(
        self,
        df: pd.DataFrame,
        warn_extrapolation: bool = False,
    ) -> np.ndarray:
        """
        Public: preprocess df → predict raw survival probs.
        Called after fit(). EnsembleTrainer uses this method.
        """
        self._check_fitted()
        X, _ = self._preprocess(df, fit=False)
        return self._predict_raw(X, warn_extrapolation=warn_extrapolation)

    def calibrate(
        self,
        raw_probs: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        fit: bool = False,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Public: apply or fit temperature calibration.
        fit=True  → fit new calibrator (requires df)
        fit=False → use existing self.calibrators
        """
        if fit:
            if df is None:
                raise ValueError("df required when fit=True")
            binary = self._make_binary_outcomes(df)
            return self._calibrate_probs(
                raw_probs, binary, df_ref=df, fit=True, verbose=False
            )
        return self._calibrate_probs(
            raw_probs, None, fit=False, calibrators=self.calibrators
        )

    # ──────────────────────────────────────────
    # Fold-level fit + predict
    # ──────────────────────────────────────────
    def _predict_fold(
        self,
        X_trn: np.ndarray,
        y_trn: Any,
        X_val: np.ndarray,
        df_val: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        binary     = self._make_binary_outcomes(df_val)
        fold_model = self._fit_model(X_trn, y_trn, verbose=False)
        raw        = self._predict_raw(X_val, model=fold_model)
        del fold_model
        cal, cals  = self._calibrate_probs(raw, binary, df_ref=df_val, fit=True)
        return raw, enforce_monotone(cal), cals

    # ──────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────
    def _calibrate_probs(
        self,
        raw_probs: np.ndarray,
        true_binary: Optional[np.ndarray],
        df_ref: Optional[pd.DataFrame] = None,
        fit: bool = True,
        calibrators: Optional[List[float]] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[float]]:
        from scipy.optimize import minimize_scalar
        from scipy.special import expit, logit

        if fit and df_ref is None:
            raise ValueError("df_ref required when fit=True.")
        if not fit and calibrators is None:
            raise ValueError("calibrators required when fit=False.")

        cal_probs = np.zeros_like(raw_probs)

        def _apply_temp(p: np.ndarray, T: float) -> np.ndarray:
            return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / T)

        def _nll(T: float, p: np.ndarray, y: np.ndarray) -> float:
            p_cal = np.clip(_apply_temp(p, T), 1e-7, 1 - 1e-7)
            return -float(np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)))

        if fit:
            calibrators = []
            for j in range(raw_probs.shape[1]):
                t       = self.horizons[j]
                p_all   = raw_probs[:, j]
                y_all   = true_binary[:, j]
                mask    = ~np.isnan(y_all)
                p_inc, y_inc = p_all[mask], y_all[mask]
                n_pos   = int(y_inc.sum())
                n_neg   = int((y_inc == 0).sum())

                if n_neg < 10 or n_pos == 0:
                    calibrators.append(1.0)
                    continue

                res      = minimize_scalar(
                    _nll, args=(p_inc, y_inc), bounds=(0.5, 10.0), method="bounded"
                )
                T_fitted = float(res.x)
                base_rt  = float(y_inc.mean())
                ratio    = _apply_temp(p_all, T_fitted).mean() / (base_rt + 1e-8)
                best_T   = T_fitted if (0.5 <= ratio <= 2.0) else 1.0

                if verbose:
                    n_excl = int((~mask).sum())
                    tag    = f"T={T_fitted:.3f}"
                    if best_T == 1.0:
                        tag += f" rejected (ratio={ratio:.2f}) → T=1.000"
                    trial = _apply_temp(p_all, best_T)
                    print(
                        f"  @{t}h  n_pos={n_pos:3d} n_neg={n_neg:3d} "
                        f"excl={n_excl:3d}  base={base_rt:.3f}  {tag}  "
                        f"raw={p_all.mean():.3f}  cal={trial.mean():.3f}"
                    )
                calibrators.append(best_T)

        for j, T in enumerate(calibrators):
            cal_probs[:, j] = _apply_temp(raw_probs[:, j], T)

        return cal_probs, calibrators

    # ──────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────
    def _make_binary_outcomes(self, df: pd.DataFrame) -> np.ndarray:
        return make_binary_outcome_matrix(
            df[TARGET_EVENT].values,
            df[TARGET_TIME].values,
            self.horizons,
        )

    def _make_strat_key(self, df: pd.DataFrame) -> pd.Series:
        time_q = pd.qcut(df[TARGET_TIME], q=4, labels=False, duplicates="drop")
        return df[TARGET_EVENT].astype(str) + "_" + time_q.astype(str)

    def hybrid_score(self, cal_probs: np.ndarray, df: pd.DataFrame) -> float:
        h2c    = {h: i for i, h in enumerate(self.horizons)}
        bs_all = brier_matrix(
            df[TARGET_EVENT].values,
            df[TARGET_TIME].values,
            cal_probs,
            self.horizons,
        )
        brier_weighted = sum(
            w * float(bs_all[h2c[h]])
            for h, w in self.BRIER_W.items()
            if h in h2c
        )
        c_idx = concordance_index_censored(
            df[TARGET_EVENT].values.astype(bool),
            df[TARGET_TIME].values,
            cal_probs[:, h2c[self.CINDEX_HORIZON]],
        )[0]
        return 0.3 * c_idx + 0.7 * (1.0 - brier_weighted)

    def evaluate(
        self,
        raw_probs: np.ndarray,
        cal_probs: np.ndarray,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        events = df[TARGET_EVENT].values.astype(bool)
        times  = df[TARGET_TIME].values

        bs_raw_all = brier_matrix(events.astype(int), times, raw_probs, self.horizons)
        bs_cal_all = brier_matrix(events.astype(int), times, cal_probs, self.horizons)

        results: Dict[str, Any] = {}
        print(f"\n{'Horizon':>8} {'C-index':>10} {'Brier raw':>12} {'Brier cal':>12}")
        print("-" * 48)
        for j, t in enumerate(self.horizons):
            c_idx  = concordance_index_censored(events, times, cal_probs[:, j])[0]
            bs_raw = float(bs_raw_all[j])
            bs_cal = float(bs_cal_all[j])
            w_str  = f"  (×{self.BRIER_W[t]:.1f})" if t in self.BRIER_W else ""
            results[t] = dict(c_index=c_idx, brier_raw=bs_raw, brier_cal=bs_cal)
            print(f"{t:>6}h  {c_idx:>10.4f} {bs_raw:>12.4f} {bs_cal:>12.4f}{w_str}")

        hs = self.hybrid_score(cal_probs, df)
        results["hybrid_score"] = hs
        print("-" * 48)
        print(f"{'Hybrid Score':>30}  {hs:.4f}   = 0.3×C@{self.CINDEX_HORIZON}h + 0.7×(1−WBS)")
        return results

    # ──────────────────────────────────────────
    # Cross-validation
    # ──────────────────────────────────────────
    def cross_validate(
        self,
        df_train: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._auto_params(len(df_train))

        # Update n_features after auto_params (RSF needs to resolve max_features)
        if isinstance(self.strategy, RSFStrategy):
            self.strategy.n_features = len(self.feature_cols)

        skf       = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        oof_raw   = np.zeros((len(df_train), len(self.horizons)))
        oof_cal   = np.zeros((len(df_train), len(self.horizons)))
        fold_scores: List[float] = []
        strat_key = self._make_strat_key(df_train)

        tag = self.strategy.name
        for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, strat_key)):
            print(f"\n── Fold {fold+1}/{self.n_splits} [{tag}] ──")
            df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
            df_val = df_train.iloc[val_idx].reset_index(drop=True)

            X_trn, y_trn, X_val, _ = self._preprocess_fold(df_trn, df_val)
            raw_p, cal_p, _        = self._predict_fold(X_trn, y_trn, X_val, df_val)

            oof_raw[val_idx] = raw_p
            oof_cal[val_idx] = cal_p

            hs    = self.hybrid_score(cal_p, df_val)
            c_idx = concordance_index_censored(
                df_val[TARGET_EVENT].values.astype(bool),
                df_val[TARGET_TIME].values,
                cal_p[:, self._cidx_col],
            )[0]
            fold_scores.append(hs)
            print(f"   C-index@{self.CINDEX_HORIZON}h: {c_idx:.4f}  Hybrid Score: {hs:.4f}")

        mean_hs = float(np.mean(fold_scores))
        std_hs  = float(np.std(fold_scores))

        self.oof_raw  = oof_raw
        self.oof_cal  = oof_cal
        self.cv_score = mean_hs

        if self.verbose:
            scores_str      = "  ".join(f"{s:.4f}" for s in fold_scores)
            score_range     = max(fold_scores) - min(fold_scores)
            high_var_thresh = max(0.05, 150 / len(df_train))
            print(f"\n[{tag}] Fold scores          : [{scores_str}]")
            print(f"[{tag}] Mean CV Hybrid Score : {mean_hs:.4f} ± {std_hs:.4f}")
            if score_range > high_var_thresh:
                print(
                    f"[{tag}] WARNING: fold range={score_range:.4f} > {high_var_thresh:.3f}"
                    " — high variance"
                )
        return oof_raw, oof_cal

    # ──────────────────────────────────────────
    # Fit on full training set
    # ──────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame) -> None:
        self._check_cv_done()
        X_all, y_all = self._preprocess(df_train, fit=True)

        print("\n=== OOF evaluation ===")
        self.evaluate(self.oof_raw, self.oof_cal, df_train)

        tag = self.strategy.name
        print(f"\n=== Retraining [{tag}] on full training set ===")
        self._model = self._fit_model(X_all, y_all, verbose=True)

        print("\n=== Calibration diagnostics ===")
        binary_all = self._make_binary_outcomes(df_train)
        _, self.calibrators = self._calibrate_probs(
            self.oof_raw, binary_all, df_ref=df_train, fit=True, verbose=True
        )

        print("\n=== Calibration health check ===")
        cal_check, _ = self._calibrate_probs(
            self.oof_raw, None, fit=False, calibrators=self.calibrators
        )
        for j, h in enumerate(self.horizons):
            base = (
                (df_train[TARGET_EVENT] == 1) & (df_train[TARGET_TIME] <= h)
            ).mean()
            print(
                f"  @{h}h  raw={self.oof_raw[:, j].mean():.3f}  "
                f"cal={cal_check[:, j].mean():.3f}  base_rate={base:.3f}"
            )

    # ──────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────
    def predict_test(self, df_test: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        print(f"\n=== Predicting on test set [{self.strategy.name}] ===")

        X_test, _ = self._preprocess(df_test, fit=False)
        raw_test  = self._predict_raw(X_test, warn_extrapolation=True)

        if self.oof_raw is not None:
            for j, h in enumerate(self.horizons):
                ratio = raw_test[:, j].mean() / (self.oof_raw[:, j].mean() + 1e-8)
                if ratio < 0.7 or ratio > 1.4:
                    warnings.warn(
                        f"@{h}h: test/OOF ratio={ratio:.2f}. "
                        "Calibrators có thể bị lệch.",
                        stacklevel=2,
                    )

        cal_probs = enforce_monotone(
            self._calibrate_probs(
                raw_test, None, fit=False, calibrators=self.calibrators
            )[0]
        )

        df_sub = pd.DataFrame({TARGET_EVENT_ID: df_test[TARGET_EVENT_ID].values})
        for j, h in enumerate(self.horizons):
            df_sub[self._prob_col(h)] = np.round(cal_probs[:, j], 6)

        cols = self._prob_cols()
        assert (df_sub[cols].values >= 0).all(),  "Negative probabilities found"
        assert (df_sub[cols].values <= 1).all(),  "Probabilities > 1 found"
        assert (
            (df_sub[cols].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all()
        ), "Monotonicity violated"
        return df_sub

    def make_submission(
        self,
        df_test: pd.DataFrame,
        path: str = "submission.csv",
    ) -> pd.DataFrame:
        self._check_fitted()
        df_sub = self.predict_test(df_test)
        df_sub.to_csv(path, index=False)
        print(f"\nSubmission saved → {path}  ({len(df_sub)} rows)")
        print(df_sub.describe())
        return df_sub

    def save_oof(
        self,
        df_train: pd.DataFrame,
        path: str = "oof_predictions.csv",
    ) -> pd.DataFrame:
        if self.oof_cal is None:
            raise RuntimeError("Call cross_validate() before save_oof().")
        oof_df = pd.DataFrame({TARGET_EVENT_ID: df_train[TARGET_EVENT_ID].values})
        for j, h in enumerate(self.horizons):
            oof_df[self._prob_col(h)] = self.oof_cal[:, j]
        oof_df.to_csv(path, index=False)
        print(f"OOF predictions saved → {path}")
        return oof_df

    # ──────────────────────────────────────────
    # Permutation sanity check
    # ──────────────────────────────────────────
    def sanity_check_shuffle(
        self,
        train_df: pd.DataFrame,
        n_permutations: int = 10,
    ) -> Dict[str, Any]:
        if self._auto_params_n is None:
            self._auto_params(len(train_df))

        p            = train_df[TARGET_EVENT].mean()
        brier_chance = p * (1.0 - p)
        struct_floor = 0.3 * 0.5 + 0.7 * (1.0 - brier_chance)
        strat_key    = self._make_strat_key(train_df)

        if self.verbose:
            print("=== Permutation sanity check ===")
            print(f"Event rate       : {p:.3f}")
            print(f"Structural floor : {struct_floor:.4f}")
            print(f"Running {n_permutations} permutations × {self.n_splits}-fold CV...\n")

        rng             = np.random.default_rng(self.random_state)
        shuffled_scores: List[float] = []

        for i in range(n_permutations):
            df_shuf = train_df.copy()
            for col in self.feature_cols:
                df_shuf[col] = rng.permutation(df_shuf[col].values)

            skf        = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True,
                random_state=int(rng.integers(0, 9999))
            )
            fold_scores: List[float] = []
            for trn_idx, val_idx in skf.split(df_shuf, strat_key):
                df_trn = df_shuf.iloc[trn_idx].reset_index(drop=True)
                df_val = df_shuf.iloc[val_idx].reset_index(drop=True)
                X_trn, y_trn, X_val, _ = self._preprocess_fold(df_trn, df_val)
                _, cal_p, _ = self._predict_fold(X_trn, y_trn, X_val, df_val)
                fold_scores.append(self.hybrid_score(cal_p, df_val))

            hs_perm = float(np.mean(fold_scores))
            shuffled_scores.append(hs_perm)
            if self.verbose:
                print(f"  Permutation {i+1:2d}: Hybrid Score = {hs_perm:.4f}")

        shuf_mean          = float(np.mean(shuffled_scores))
        shuf_std           = float(np.std(shuffled_scores))
        adaptive_threshold = struct_floor + 3.0 * shuf_std
        real_score         = self.cv_score
        passed             = None

        if self.verbose:
            print(f"\nShuffled mean       : {shuf_mean:.4f} ± {shuf_std:.4f}")
            print(f"Adaptive threshold  : {adaptive_threshold:.4f}")

        if real_score is not None:
            passed = real_score > shuf_mean + 2.0 * shuf_std
            if self.verbose:
                print(f"Real CV score       : {real_score:.4f}")
                status = "PASS: signal is real." if passed else "WARNING: not significantly above shuffled."
                print(status)

        return {
            "shuffled_scores"   : shuffled_scores,
            "shuf_mean"         : shuf_mean,
            "shuf_std"          : shuf_std,
            "structural_floor"  : struct_floor,
            "adaptive_threshold": adaptive_threshold,
            "real_score"        : real_score,
            "passed"            : passed,
        }