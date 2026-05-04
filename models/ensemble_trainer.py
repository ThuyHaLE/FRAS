from __future__ import annotations

import warnings
import itertools
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize

from sksurv.metrics import concordance_index_censored
from features import TARGET_EVENT_ID, TARGET_EVENT, TARGET_TIME, HORIZONS, BRIER_W, CINDEX_HORIZON, RANDOM_STATE
from models import ModelStrategy, SurvivalTrainer, enforce_monotone, brier_matrix

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
@dataclass
class TrainerConfig:
    """
    Config for a sub-trainer in the ensemble.

    Example:
        TrainerConfig("RSF",    RSFStrategy(n_trees=300), rsf_cols)
        TrainerConfig("GB",     GradBoostStrategy(),      gb_cols),
        TrainerConfig("Coxnet", CoxnetStrategy(),         cox_cols),
        TrainerConfig("CGB",    CGBStrategy(),            cgb_cols),

        TrainerConfig("XGB",    XGBSurvivalStrategy(),    xgb_cols)  # custom
    """
    tag: str
    strategy: ModelStrategy
    feature_cols: List[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)

# ──────────────────────────────────────────
# Shared scoring util
# ──────────────────────────────────────────
def compute_hybrid_score(
    cal_probs: np.ndarray,
    df: pd.DataFrame,
    horizons: List[int],
    brier_w: Dict[int, float],
    cindex_horizon: int,
) -> float:
    h2c    = {h: i for i, h in enumerate(horizons)}
    bs_all = brier_matrix(
        df[TARGET_EVENT].values,
        df[TARGET_TIME].values,
        cal_probs,
        horizons,
    )
    brier_weighted = sum(
        w * float(bs_all[h2c[h]])
        for h, w in brier_w.items()
        if h in h2c
    )
    c_idx = concordance_index_censored(
        df[TARGET_EVENT].values.astype(bool),
        df[TARGET_TIME].values,
        cal_probs[:, h2c[cindex_horizon]],
    )[0]
    return 0.3 * c_idx + 0.7 * (1.0 - brier_weighted)

# ──────────────────────────────────────────
# EnsembleTrainer
# ──────────────────────────────────────────
class EnsembleTrainer:
    """
    Wraps N SurvivalTrainer instances with independent feature sets and strategies.

    Example:
        configs = [
            TrainerConfig("RSF",    RSFStrategy(n_trees=300), rsf_cols)
            TrainerConfig("GB",     GradBoostStrategy(),      gb_cols),
            TrainerConfig("Coxnet", CoxnetStrategy(),         cox_cols),
            TrainerConfig("CGB",    CGBStrategy(),            cgb_cols),

            TrainerConfig("XGB",    XGBSurvivalStrategy(),    xgb_cols), #custom
        ]
        ensemble = EnsembleTrainer(configs)
        ensemble.cross_validate(train_df)
        ensemble.optimize_weights(train_df, strategy="scipy")
        ensemble.fit(train_df)
        df_sub = ensemble.make_submission(test_df)
    """

    HORIZONS       = HORIZONS
    BRIER_W        = BRIER_W
    CINDEX_HORIZON = CINDEX_HORIZON

    def __init__(
        self,
        configs: List[TrainerConfig],
        ensemble_weights: Optional[Tuple[float, ...]] = None,
        horizons: Optional[List[int]] = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
    ):
        if len(configs) < 2:
            raise ValueError("Required at least 2 TrainerConfig.")

        self.configs      = configs
        self.horizons     = horizons or HORIZONS
        self.random_state = random_state
        self.verbose      = verbose

        shared = dict(
            horizons     = self.horizons,
            random_state = random_state,
            verbose      = verbose,
        )

        self._trainers: List[Tuple[str, SurvivalTrainer]] = [
            (
                cfg.tag,
                SurvivalTrainer(
                    feature_cols = cfg.feature_cols,
                    strategy     = cfg.strategy,
                    **shared,
                    **cfg.kwargs,
                ),
            )
            for cfg in configs
        ]

        n = len(self._trainers)
        self.ensemble_weights: Tuple[float, ...] = (
            ensemble_weights
            if ensemble_weights is not None
            else tuple(1.0 / n for _ in range(n))
        )
        if len(self.ensemble_weights) != n:
            raise ValueError(
                f"ensemble_weights length {len(self.ensemble_weights)} != n_models {n}"
            )

        # State
        self.calibrators: Optional[List[float]] = None
        self.oof_raw: Optional[np.ndarray]      = None
        self.oof_cal: Optional[np.ndarray]      = None
        self.cv_score: Optional[float]          = None

    # ──────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────
    @property
    def _n_models(self) -> int:
        return len(self._trainers)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────
    def _normalized_weights(
        self, weights: Optional[Tuple[float, ...]] = None
    ) -> Tuple[float, ...]:
        w     = weights or self.ensemble_weights
        total = sum(w)
        return tuple(wi / total for wi in w)

    def _blend(
        self,
        raw_list: List[np.ndarray],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> np.ndarray:
        norm_w = self._normalized_weights(weights)
        return enforce_monotone(
            sum(w * enforce_monotone(p) for w, p in zip(norm_w, raw_list))
        )

    def _prob_col(self, h: int) -> str:
        return f"prob_{h}h"

    def _prob_cols(self) -> List[str]:
        return [self._prob_col(h) for h in self.horizons]

    def _score(self, cal_probs: np.ndarray, df: pd.DataFrame) -> float:
        return compute_hybrid_score(
            cal_probs, df, self.horizons, self.BRIER_W, self.CINDEX_HORIZON
        )

    def _fit_ensemble_calibrator(
        self,
        raw_blended: np.ndarray,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[float]]:
        """Fit calibrator on blended OOF. Uses public API of trainers."""
        ref = self._trainers[0][1]
        cal, calibrators = ref.calibrate(raw_blended, df=df, fit=True)
        return enforce_monotone(cal), calibrators

    def _apply_ensemble_calibrator(self, raw_blended: np.ndarray) -> np.ndarray:
        """Apply fitted calibrator. Uses public API of trainers."""
        if self.calibrators is None:
            raise RuntimeError("Ensemble calibrator must be fit. Call fit() first.")
        ref = self._trainers[0][1]
        # Set calibrators to use calibrate (fit=False)
        original         = ref.calibrators
        ref.calibrators  = self.calibrators
        cal, _           = ref.calibrate(raw_blended, fit=False)
        ref.calibrators  = original
        return enforce_monotone(cal)

    # ──────────────────────────────────────────
    # Guards
    # ──────────────────────────────────────────
    def _check_cv_done(self, tags: Optional[List[str]] = None) -> None:
        check    = [(tag, t) for tag, t in self._trainers if tag in tags] if tags else self._trainers
        not_done = [tag for tag, t in check if t.oof_raw is None]
        if not_done:
            raise RuntimeError(
                f"cross_validate() not done for: {not_done}\n"
                "Order: cross_validate() → [optimize_weights()] → fit()"
            )

    def is_fitted(self) -> bool:
        return (
            all(t.is_fitted() for _, t in self._trainers)
            and self.calibrators is not None
        )

    # ──────────────────────────────────────────
    # Cross-validation
    # ──────────────────────────────────────────
    def cross_validate(self, df_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        for tag, trainer in self._trainers:
            print("=" * 60)
            print(f"  EnsembleTrainer: {tag} cross_validate")
            print("=" * 60)
            trainer.cross_validate(df_train)

        self.oof_raw = self._blend([t.oof_raw for _, t in self._trainers])

        if self.verbose:
            print("\n=== Fitting ensemble calibrator on blended OOF ===")
        self.oof_cal, self.calibrators = self._fit_ensemble_calibrator(
            self.oof_raw, df_train, verbose=self.verbose
        )
        self.cv_score = self._score(self.oof_cal, df_train)

        print(f"\n[Ensemble] Blended OOF Hybrid Score : {self.cv_score:.4f}")
        for (tag, trainer), w in zip(self._trainers, self._normalized_weights()):
            print(f"  {tag:<12} CV Score : {trainer.cv_score:.4f}  w={w:.3f}")

        return self.oof_raw, self.oof_cal

    # ──────────────────────────────────────────
    # Weight optimization
    # ──────────────────────────────────────────
    def optimize_weights(
        self,
        df_train: pd.DataFrame,
        strategy: str = "score",
        score_temperature: float = 20.0,
        n_restarts: int = 10,
        verbose: bool = True,
    ) -> Tuple[float, ...]:
        """
        strategy : "score" | "grid" | "scipy"

        "score"  — softmax over CV scores (zero overfit, recommended)
        "grid"   — grid search (only good for N=2,3)
        "scipy"  — minimize(-score) with Dirichlet restarts (good for N>=4)
        """
        self._check_cv_done()

        valid = ("score", "grid", "scipy")
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}")

        if strategy == "score":
            best_w = self._optimize_by_score(verbose, score_temperature)
        elif strategy == "grid":
            best_w = self._optimize_by_grid(df_train, verbose=verbose)
        else:
            best_w = self._optimize_by_scipy(df_train, n_restarts, verbose)

        self.ensemble_weights = best_w
        self.oof_raw = self._blend([t.oof_raw for _, t in self._trainers])
        self.oof_cal, self.calibrators = self._fit_ensemble_calibrator(
            self.oof_raw, df_train, verbose=False
        )
        self.cv_score = self._score(self.oof_cal, df_train)

        weights_str = "  ".join(
            f"{tag}={w:.3f}" for (tag, _), w in zip(self._trainers, best_w)
        )
        print(f"\n[optimize_weights/{strategy}]  {weights_str}  score={self.cv_score:.4f}")
        return self.ensemble_weights

    def _optimize_by_score(
        self, verbose: bool, temperature: float
    ) -> Tuple[float, ...]:
        scores = []
        for tag, trainer in self._trainers:
            if trainer.cv_score is None:
                warnings.warn(f"cv_score missing for {tag}. Using equal weights.", stacklevel=3)
                n = self._n_models
                return tuple(1.0 / n for _ in range(n))
            scores.append(trainer.cv_score)

        scores_arr = np.array(scores)
        exp_s      = np.exp((scores_arr - scores_arr.max()) * temperature)
        weights    = tuple(float(w) for w in exp_s / exp_s.sum())

        if verbose:
            print(f"\n=== optimize_weights [score, T={temperature}] ===")
            for (tag, _), s, w in zip(self._trainers, scores, weights):
                print(f"  {tag:<12} cv={s:.4f}  → w={w:.3f}")
        return weights

    def _optimize_by_scipy(
        self, df_train: pd.DataFrame, n_restarts: int, verbose: bool
    ) -> Tuple[float, ...]:
        raw_list = [t.oof_raw for _, t in self._trainers]
        n        = self._n_models

        def neg_score(w_raw: np.ndarray) -> float:
            w       = np.abs(w_raw) / np.abs(w_raw).sum()
            blended = self._blend(raw_list, tuple(w))
            cal, _  = self._fit_ensemble_calibrator(blended, df_train, verbose=False)
            return -self._score(cal, df_train)

        constraints = {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}
        bounds      = [(0.0, 1.0)] * n
        best_result = None

        if verbose:
            print(f"\n=== optimize_weights [scipy, restarts={n_restarts}] ===")

        rng = np.random.default_rng(self.random_state)
        for i in range(n_restarts):
            w0     = rng.dirichlet(np.ones(n))
            result = minimize(
                neg_score, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-7, "maxiter": 200},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                if verbose:
                    w_norm = np.abs(result.x) / np.abs(result.x).sum()
                    print(
                        f"  restart {i+1:>2}:  "
                        + "  ".join(f"{tag}={w:.3f}" for (tag, _), w in zip(self._trainers, w_norm))
                        + f"  score={-result.fun:.4f} ←"
                    )

        w_best = np.abs(best_result.x)
        w_best = w_best / w_best.sum()
        return tuple(float(w) for w in w_best)

    def _optimize_by_grid(
        self, df_train: pd.DataFrame, steps: int = 21, verbose: bool = True
    ) -> Tuple[float, ...]:
        raw_list = [t.oof_raw for _, t in self._trainers]
        n        = self._n_models

        if n > 3:
            warnings.warn(
                f"Grid search for {n} models is very slow. Use strategy='scipy'.",
                UserWarning, stacklevel=3,
            )

        best_score = -np.inf
        best_w     = tuple(1.0 / n for _ in range(n))
        candidates = np.linspace(0, 1, steps)

        for combo in itertools.product(*([candidates] * (n - 1))):
            last = 1.0 - sum(combo)
            if last < 0 or last > 1:
                continue
            w = combo + (last,)
            blended = self._blend(raw_list, w)
            cal, _  = self._fit_ensemble_calibrator(blended, df_train, verbose=False)
            score   = self._score(cal, df_train)

            if score > best_score:
                best_score = score
                best_w     = w
                if verbose:
                    w_str = "  ".join(
                        f"{tag}={wi:.2f}" for (tag, _), wi in zip(self._trainers, w)
                    )
                    print(f"  {w_str}  score={score:.4f} ←")

        return best_w

    # ──────────────────────────────────────────
    # Compare subsets
    # ──────────────────────────────────────────
    def compare_subsets(
        self,
        df_train: pd.DataFrame,
        subset_sizes: Optional[List[int]] = None,
        weight_strategy: str = "score",
        score_temperature: float = 20.0,
        top_k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare all subsets of models.

        Returns:
            {
                "results": [{"tags": (...), "weights": (...), "score": float}, ...],
                "best"   : {"tags": (...), "weights": (...), "score": float},
            }
        """
        self._check_cv_done()

        all_tags  = [tag for tag, _ in self._trainers]
        all_sizes = subset_sizes or list(range(2, self._n_models + 1))
        all_results: List[Dict[str, Any]] = []

        total_combos = sum(
            len(list(itertools.combinations(all_tags, k))) for k in all_sizes
        )
        print(f"\n{'='*60}")
        print(f"  compare_subsets: {total_combos} combinations, sizes={all_sizes}")
        print(f"{'='*60}")

        rng       = np.random.default_rng(self.random_state)
        combo_idx = 0

        for size in all_sizes:
            for combo_tags in itertools.combinations(all_tags, size):
                combo_idx += 1
                raw_list   = [
                    t.oof_raw for tag, t in self._trainers if tag in combo_tags
                ]
                scores_sub = [
                    t.cv_score for tag, t in self._trainers if tag in combo_tags
                ]

                if weight_strategy == "score":
                    scores_arr = np.array(scores_sub)
                    exp_s      = np.exp((scores_arr - scores_arr.max()) * score_temperature)
                    weights    = tuple(float(w) for w in exp_s / exp_s.sum())

                elif weight_strategy == "scipy":
                    n_sub = len(combo_tags)

                    def neg_score_sub(w_raw):
                        w = np.abs(w_raw) / np.abs(w_raw).sum()
                        blended = self._blend(raw_list, tuple(w))
                        cal, _  = self._fit_ensemble_calibrator(blended, df_train, verbose=False)
                        return -self._score(cal, df_train)

                    best_r = None
                    for _ in range(5):
                        w0 = rng.dirichlet(np.ones(n_sub))
                        r  = minimize(
                            neg_score_sub, w0, method="SLSQP",
                            bounds=[(0, 1)] * n_sub,
                            constraints={"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1},
                            options={"ftol": 1e-7, "maxiter": 100},
                        )
                        if best_r is None or r.fun < best_r.fun:
                            best_r = r
                    w_arr   = np.abs(best_r.x)
                    weights = tuple(float(w) for w in w_arr / w_arr.sum())

                else:
                    raise ValueError(f"weight_strategy không hợp lệ: {weight_strategy!r}")

                blended = self._blend(raw_list, weights)
                cal, _  = self._fit_ensemble_calibrator(blended, df_train, verbose=False)
                score   = self._score(cal, df_train)

                entry = {"tags": combo_tags, "weights": weights, "score": score}
                all_results.append(entry)

                if verbose:
                    tag_str = "+".join(combo_tags)
                    w_str   = "  ".join(f"{t}={w:.3f}" for t, w in zip(combo_tags, weights))
                    print(
                        f"  [{combo_idx:>3}/{total_combos}] {tag_str:<30}  "
                        f"{w_str}  score={score:.4f}"
                    )

        all_results.sort(key=lambda x: x["score"], reverse=True)

        print(f"\n{'─'*60}")
        print(f"  TOP {min(top_k, len(all_results))} combinations:")
        print(f"{'─'*60}")
        for i, entry in enumerate(all_results[:top_k]):
            tag_str = "+".join(entry["tags"])
            w_str   = "  ".join(
                f"{t}={w:.3f}" for t, w in zip(entry["tags"], entry["weights"])
            )
            print(f"  #{i+1}  {tag_str:<30}  {w_str}  score={entry['score']:.4f}")

        return {"results": all_results, "best": all_results[0]}

    def apply_best_subset(
        self,
        best: Dict[str, Any],
        df_train: pd.DataFrame,
    ) -> None:
        """Apply the best subset from compare_subsets()."""
        best_tags = set(best["tags"])
        dropped   = {tag for tag, _ in self._trainers} - best_tags
        if dropped:
            print(f"\n[apply_best_subset] Dropping models: {dropped}")
            self._trainers = [
                (tag, t) for tag, t in self._trainers if tag in best_tags
            ]

        self.ensemble_weights = best["weights"]
        self.oof_raw = self._blend([t.oof_raw for _, t in self._trainers])
        self.oof_cal, self.calibrators = self._fit_ensemble_calibrator(
            self.oof_raw, df_train, verbose=False
        )
        self.cv_score = self._score(self.oof_cal, df_train)

        print(
            f"[apply_best_subset] Active: {[t for t, _ in self._trainers]}\n"
            f"  weights: {self.ensemble_weights}\n"
            f"  score  : {self.cv_score:.4f}"
        )

    # ──────────────────────────────────────────
    # Fit on full training set
    # ──────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame) -> None:
        self._check_cv_done()

        for tag, trainer in self._trainers:
            print("\n" + "=" * 60)
            print(f"  EnsembleTrainer: {tag} fit")
            print("=" * 60)
            trainer.fit(df_train)

        print("\n=== EnsembleTrainer: Refitting ensemble calibrator ===")
        self.oof_raw = self._blend([t.oof_raw for _, t in self._trainers])
        self.oof_cal, self.calibrators = self._fit_ensemble_calibrator(
            self.oof_raw, df_train, verbose=self.verbose
        )
        self.cv_score = self._score(self.oof_cal, df_train)

        print("\n=== EnsembleTrainer: Blended OOF evaluation ===")
        ref_trainer = self._trainers[0][1]
        ref_trainer.evaluate(self.oof_raw, self.oof_cal, df_train)

    # ──────────────────────────────────────────
    # Predict + submit
    # ──────────────────────────────────────────
    def predict_test(self, df_test: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted():
            raise RuntimeError("Not fitted. Call cross_validate() + fit().")

        print("\n=== EnsembleTrainer: Predicting on test set ===")

        raw_list = []
        for tag, trainer in self._trainers:
            print(f"  [{tag}] predicting...")
            # Use public API — do not call _preprocess / _predict_raw directly
            raw = trainer.predict_raw(df_test, warn_extrapolation=True)

            if trainer.oof_raw is not None:
                for j, h in enumerate(self.horizons):
                    ratio = raw[:, j].mean() / (trainer.oof_raw[:, j].mean() + 1e-8)
                    if ratio < 0.7 or ratio > 1.4:
                        warnings.warn(
                            f"[{tag}] @{h}h: test/OOF ratio={ratio:.2f} — "
                            "distribution shift detected.",
                            stacklevel=2,
                        )
            raw_list.append(raw)

        raw_blended = self._blend(raw_list)
        cal_probs   = self._apply_ensemble_calibrator(raw_blended)

        prob_cols = self._prob_cols()
        df_out    = pd.DataFrame({TARGET_EVENT_ID: df_test[TARGET_EVENT_ID].values})
        for j, col in enumerate(prob_cols):
            df_out[col] = cal_probs[:, j]

        assert (df_out[prob_cols].values >= 0).all(),                             "Negative probabilities"
        assert (df_out[prob_cols].values <= 1).all(),                             "Probabilities > 1"
        assert (df_out[prob_cols].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all(), "Monotonicity violated"
        return df_out

    def make_submission(
        self,
        df_test: pd.DataFrame,
        path: str = "submission.csv",
    ) -> pd.DataFrame:
        df_sub    = self.predict_test(df_test)
        prob_cols = self._prob_cols()
        df_sub[prob_cols] = df_sub[prob_cols].round(6)
        df_sub.to_csv(path, index=False)
        print(f"\nSubmission saved → {path}  ({len(df_sub)} rows)")
        print(df_sub[prob_cols].describe().round(4).to_string())
        return df_sub