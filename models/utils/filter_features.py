#models/utils/filter_features.py

from __future__ import annotations

import copy
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from models.strategies import ModelStrategy
from models.survival_model import SurvivalTrainer, enforce_monotone
from features import TARGET_EVENT, TARGET_TIME

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
VALID_MODEL_TYPES = frozenset({"rsf", "gradboost", "coxnet", "cgb"})
_LINEAR_MODEL_TYPES = frozenset({"coxnet", "cgb"})

FeatureList = list[str]
TierMap     = dict[str, FeatureList]

# ──────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TierThresholds:
    """
    Importance thresholds - change in one place, apply everywhere.
    
    Note:
    Coxnet has importance scale ~10-30× larger than RSF/GradBoost (linear base learners more sensitive to permutation).
    Use COXNET_THRESHOLDS for Coxnet model.
    """
    star: float        = 0.005
    tier2: float       = 0.0005
    positive: float    = 0.0
    noise_ratio: float = 5.0   # relaxed from 3.0 for small dataset (~220 rows)

DEFAULT_THRESHOLDS = TierThresholds()
COXNET_THRESHOLDS = TierThresholds(star=0.05,  tier2=0.005,  positive=0.0, noise_ratio=5.0)

# ──────────────────────────────────────────────────────────────
# SurvivalModelWrapper — interface for PI
# ──────────────────────────────────────────────────────────────
class SurvivalModelWrapper(ABC):
    """
    Interface for PI wrapper.

    Each fold creates a deep copy of the wrapper → fit separately → compute PI.
    Implement this class to inject a custom model into PermutationImportance.
    """

    @abstractmethod
    def fit(self, df_trn: pd.DataFrame, df_val: pd.DataFrame) -> None:
        """Train model on fold, save internal state."""
        ...

    @abstractmethod
    def score(self, X: np.ndarray, df_val: pd.DataFrame) -> float:
        """Return scalar score (higher = better)."""
        ...

    @abstractmethod
    def get_X_val(self, df_val: pd.DataFrame) -> np.ndarray:
        """Preprocess df_val → feature matrix."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

# ──────────────────────────────────────────────────────────────
# Built-in wrappers — use SurvivalTrainer
# ──────────────────────────────────────────────────────────────
class _TrainerWrapper(SurvivalModelWrapper):
    """Base wrapper used for SurvivalTrainer + ModelStrategy. """

    def __init__(
        self,
        feature_cols: FeatureList,
        strategy: ModelStrategy,
        random_state: int = 0,
    ):
        self._feature_cols  = feature_cols
        self._strategy      = strategy
        self._random_state  = random_state
        self._trainer: SurvivalTrainer | None = None
        self._model: Any    = None

    def fit(self, df_trn: pd.DataFrame, df_val: pd.DataFrame) -> None:
        self._trainer = SurvivalTrainer(
            feature_cols = self._feature_cols,
            strategy     = copy.deepcopy(self._strategy),
            random_state = self._random_state,
            verbose      = False,
            auto_tune    = False,
        )
        X_trn, y_trn, _, _ = self._trainer._preprocess_fold(df_trn, df_val)
        self._model = self._trainer._fit_model(X_trn, y_trn, verbose=False)

    def get_X_val(self, df_val: pd.DataFrame) -> np.ndarray:
        _, _, X_val, _ = self._trainer._preprocess_fold(df_val, df_val)
        return X_val

    def score(self, X: np.ndarray, df_val: pd.DataFrame) -> float:
        raw      = self._trainer._predict_raw(X, model=self._model)
        surv_fns = self._trainer.strategy.predict_survival(self._model, X)
        raw      = enforce_monotone(self._trainer._survival_to_hit_probs(surv_fns))
        cal, _   = self._trainer._calibrate_probs(
            raw, None, fit=False, calibrators=[1.0] * len(self._trainer.horizons)
        )
        return self._trainer.hybrid_score(enforce_monotone(cal), df_val)

    @property
    def name(self) -> str:
        return self._strategy.name


def make_wrapper(
    model_type: str,
    feature_cols: FeatureList,
    random_state: int = 0,
) -> _TrainerWrapper:
    """
    Factory method to create wrapper from model_type string.
    Used internally for backward compatibility with filter_features(model_type=...).
    """
    from models.strategies.builtin import (
        RSFStrategy, GradBoostStrategy, CoxnetStrategy, CGBStrategy
    )
    registry = {
        "rsf"      : RSFStrategy,
        "gradboost": GradBoostStrategy,
        "coxnet"   : CoxnetStrategy,
        "cgb"      : CGBStrategy,
    }
    if model_type not in registry:
        raise ValueError(f"model_type phải là một trong {set(registry)}")
    strategy = registry[model_type](
        n_features=len(feature_cols)
    ) if model_type == "rsf" else registry[model_type]()
    return _TrainerWrapper(feature_cols, strategy, random_state)

# ──────────────────────────────────────────────────────────────
# PermutationImportance — main class
# ──────────────────────────────────────────────────────────────
class PermutationImportance:
    """
    Cross-validated permutation importance for any model wrapper.

    Example — built-in wrapper:
        from models.strategies.builtin import RSFStrategy
        wrapper = _TrainerWrapper(features, RSFStrategy())
        pi      = PermutationImportance(wrapper)
        summary = pi.run(df, features, enriched_features)

    Example — inject model custom:
        class MyWrapper(SurvivalModelWrapper):
            def fit(self, df_trn, df_val): ...
            def score(self, X, df_val): ...
            def get_X_val(self, df_val): ...

        pi = PermutationImportance(MyWrapper())
        summary = pi.run(df, features, enriched_features)
    """

    def __init__(
        self,
        wrapper: SurvivalModelWrapper,
        n_splits: int   = 3,
        n_repeats: int  = 30,
        random_state: int = 42,
        n_jobs: int     = -1,
    ):
        self.wrapper      = wrapper
        self.n_splits     = n_splits
        self.n_repeats    = n_repeats
        self.random_state = random_state
        self.n_jobs       = n_jobs

    # ── Public ────────────────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        all_features: FeatureList,
        enriched_features: FeatureList,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        skf       = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        time_q    = pd.qcut(df[TARGET_TIME], q=4, labels=False, duplicates="drop")
        strat_key = df[TARGET_EVENT].astype(str) + "_" + time_q.astype(str)

        fold_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._run_fold)(
                fold_idx = i,
                df_trn   = df.iloc[trn].reset_index(drop=True),
                df_val   = df.iloc[val].reset_index(drop=True),
                features = all_features,
            )
            for i, (trn, val) in enumerate(skf.split(df, strat_key))
        )

        summary = _aggregate_folds(fold_results)
        summary["type"] = summary["feature"].map(
            lambda f: "enriched" if f in set(enriched_features) else "base"
        )
        _print_importance_table(summary, self.wrapper.name)
        _print_type_summary(summary)

        if output_path:
            summary.to_csv(output_path, index=False)
            print(f"\nSaved → {output_path}")

        return summary

    # ── Internal ──────────────────────────────────────────────
    def _run_fold(
        self,
        fold_idx: int,
        df_trn: pd.DataFrame,
        df_val: pd.DataFrame,
        features: FeatureList,
    ) -> pd.DataFrame:
        # Deep copy to ensure each fold has its own state (thread-safe)
        wrapper = copy.deepcopy(self.wrapper)
        if hasattr(wrapper, "_random_state"):
            wrapper._random_state = fold_idx * 7

        wrapper.fit(df_trn, df_val)
        X_val    = wrapper.get_X_val(df_val)
        baseline = wrapper.score(X_val, df_val)

        print(f"  Fold {fold_idx+1} | {wrapper.name} baseline: {baseline:.4f}")

        rng  = np.random.default_rng(fold_idx * 77)
        rows = [
            {
                "feature": fname,
                **self._permute_feature(wrapper, X_val, df_val, col_idx, baseline, rng),
            }
            for col_idx, fname in enumerate(features)
        ]
        result         = pd.DataFrame(rows)
        result["fold"] = fold_idx + 1
        return result

    def _permute_feature(
        self,
        wrapper: SurvivalModelWrapper,
        X_val: np.ndarray,
        df_val: pd.DataFrame,
        feature_idx: int,
        baseline: float,
        rng: np.random.Generator,
    ) -> dict:
        drops = []
        for _ in range(self.n_repeats):
            Xp = X_val.copy()
            Xp[:, feature_idx] = rng.permutation(Xp[:, feature_idx])
            drops.append(baseline - wrapper.score(Xp, df_val))
        return {
            "importance_mean": float(np.mean(drops)),
            "importance_std":  float(np.std(drops)),
        }

# ──────────────────────────────────────────────────────────────
# Aggregation + display helpers
# ──────────────────────────────────────────────────────────────
def _aggregate_folds(all_fold_results: list[pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat(all_fold_results)
        .groupby("feature")["importance_mean"]
        .agg(importance="mean", fold_std="std")
        .sort_values("importance", ascending=False)
        .reset_index()
    )

def _importance_marker(importance: float, thresholds: TierThresholds) -> str:
    if importance >= thresholds.star:    return "★"
    if importance > thresholds.positive: return "+"
    return "✗"

def _print_importance_table(
    summary: pd.DataFrame,
    model_name: str,
    thresholds: TierThresholds = DEFAULT_THRESHOLDS,
) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  Permutation Importance [{model_name.upper()}]")
    print(sep)
    print(f"{'#':<4} {'Feature':<35} {'Importance':>11} {'FoldStd':>9}  {'Type':<10} Mark")
    print(sep)
    for rank, (_, row) in enumerate(summary.iterrows(), start=1):
        print(
            f"{rank:<4} {row['feature']:<35} {row['importance']:>11.5f} "
            f"{row['fold_std']:>9.5f}  {row['type']:<10} "
            f"{_importance_marker(row['importance'], thresholds)}"
        )
    print(sep)

def _print_type_summary(summary: pd.DataFrame) -> None:
    print("\n── Summary by type ──")
    for ftype in ("base", "enriched"):
        sub   = summary[summary["type"] == ftype]
        n_pos = (sub["importance"] > 0).sum()
        top3  = sub[sub["importance"] > 0]["feature"].head(3).tolist()
        print(f"  {ftype:<10}: {n_pos}/{len(sub)} positive | top-3: {top3}")

# ──────────────────────────────────────────────────────────────
# Feature classification
# ──────────────────────────────────────────────────────────────
def _classify_one(row: pd.Series, thresholds: TierThresholds) -> str:
    imp, std = row["importance"], row["fold_std"]
    noise    = std / (imp + 1e-9)
    if imp >= thresholds.star:   return "TIER1"
    if imp >= thresholds.tier2:  return "TIER2" if noise <= thresholds.noise_ratio else "TIER3"
    if imp > thresholds.positive: return "TIER3"
    return "DROP"

def classify_features(
    summary: pd.DataFrame,
    thresholds: TierThresholds = DEFAULT_THRESHOLDS,
    verbose: bool = True,
) -> dict[str, FeatureList | pd.DataFrame]:
    df         = summary.copy()
    df["tier"] = df.apply(_classify_one, axis=1, thresholds=thresholds)

    if verbose:
        for tier in ("TIER1", "TIER2", "TIER3", "DROP"):
            rows = df[df["tier"] == tier]
            print(f"\n{tier} ({len(rows)}):")
            for _, row in rows.iterrows():
                noise = row["fold_std"] / (row["importance"] + 1e-9)
                print(
                    f"  {row['feature']:<35} "
                    f"imp={row['importance']:>8.5f}  "
                    f"std={row['fold_std']:>8.5f}  "
                    f"ratio={noise:.1f}x"
                )

    tier_map: TierMap = {
        tier: df[df["tier"] == tier]["feature"].tolist()
        for tier in ("TIER1", "TIER2", "TIER3", "DROP")
    }
    survival_features = tier_map["TIER1"] + tier_map["TIER2"]

    if verbose:
        print(f"\nSurvival features (TIER1 + TIER2): {len(survival_features)}")
        print(survival_features)

    return {"summary": df, "survival_features": survival_features, **tier_map}

# ──────────────────────────────────────────────────────────────
# Disagreement warnings
# ──────────────────────────────────────────────────────────────
def _warn_disagreements(
    imp_a: pd.Series,
    imp_b: pd.Series,
    features_a: FeatureList,
    features_b: FeatureList,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    found = False
    for f in features_a:
        b_imp = imp_b.get(f, 0.0)
        if b_imp < 0:
            print(f"  ⚠  {f:<35} {label_a} keep | {label_b} negative (imp={b_imp:.5f})")
            found = True
    for f in features_b:
        a_imp = imp_a.get(f, 0.0)
        if a_imp < 0:
            print(f"  ⚠  {f:<35} {label_b} keep | {label_a} negative (imp={a_imp:.5f})")
            found = True
    if not found:
        print(f"  ✓  No serious disagreements ({label_a} vs {label_b}).")

# ──────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────
def filter_features(
    train_df: pd.DataFrame,
    base_features: list,
    enriched_features: list,
    model_type: str = "rsf",
    wrapper: SurvivalModelWrapper | None = None,
    n_repeats: int = 30,
    thresholds: TierThresholds = DEFAULT_THRESHOLDS,
    output_path: str | None = None,
) -> dict[str, FeatureList | pd.DataFrame]:
    """
    Single-model PI + feature classification.

    Two usage modes:
        # 1. String (backward compat)
        filter_features(train_df, base_f, enriched_f, model_type="rsf")

        # 2. Inject wrapper custom
        filter_features(train_df, base_f, enriched_f, wrapper=MyWrapper())
    """
    all_features = list(set(base_features + enriched_features))
    print(
        f"\nFeatures — Base: {len(base_features)} | "
        f"Enriched: {len(enriched_features)} | Total: {len(all_features)}"
    )

    if wrapper is None:
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}")
        if model_type in _LINEAR_MODEL_TYPES and thresholds == DEFAULT_THRESHOLDS:
            warnings.warn(
                f"{model_type.upper()} PI with DEFAULT_THRESHOLDS may select too many features.",
                UserWarning, stacklevel=2,
            )
        wrapper = make_wrapper(model_type, all_features)

    pi      = PermutationImportance(wrapper, n_repeats=n_repeats, n_jobs=1)
    summary = pi.run(train_df, all_features, enriched_features, output_path=output_path)
    return classify_features(summary, thresholds=thresholds)


def filter_features_independent(
    train_df: pd.DataFrame,
    base_features: list,
    enriched_features: list,
    n_repeats: int = 30,
    thresholds: TierThresholds = DEFAULT_THRESHOLDS,
    coxnet_thresholds: TierThresholds | None = None,
    rsf_output_path: str | None = "permutation_importance_rsf.csv",
    gb_output_path:  str | None = "permutation_importance_gb.csv",
    cox_output_path: str | None = "permutation_importance_coxnet.csv",
    cgb_output_path: str | None = "permutation_importance_cgb.csv",
    verbose: bool = True,
) -> dict[str, FeatureList]:
    """
    Run independent PI for RSF + GradBoost + Coxnet + CGB.
    Returns {"rsf": [...], "gradboost": [...], "coxnet": [...], "cgb": [...]}.
    """
    _cox_thr = coxnet_thresholds or COXNET_THRESHOLDS

    # (model_type, label, role, output_path, thresholds)
    _CONFIGS = [
        ("rsf",       "RSF",       "PRIMARY",       rsf_output_path, thresholds),
        ("gradboost", "GradBoost", "SUPPORT",        gb_output_path,  thresholds),
        ("coxnet",    "Coxnet",    "LINEAR",         cox_output_path, _cox_thr),
        ("cgb",       "CGB",       "COMPONENTWISE",  cgb_output_path, thresholds),
    ]

    results: dict[str, dict] = {}
    for step, (model_type, label, role, out_path, thr) in enumerate(_CONFIGS, 1):
        print(f"\n{'='*70}")
        print(f"  Step {step}/4: {label} Permutation Importance  [{role}]")
        if model_type in _LINEAR_MODEL_TYPES:
            print(f"  (thresholds: star={thr.star}, tier2={thr.tier2})")
        print("=" * 70)

        results[model_type] = filter_features(
            train_df, base_features, enriched_features,
            model_type  = model_type,
            n_repeats   = n_repeats,
            thresholds  = thr,
            output_path = out_path,
        )

    # Map name → short key
    named = {
        "rsf"   : results["rsf"],
        "gradboost": results["gradboost"],
        "coxnet": results["coxnet"],
        "cgb"   : results["cgb"],
    }

    if verbose:
        _print_venn_and_disagreements(named)

    return {k: v["survival_features"] for k, v in named.items()}


def _print_venn_and_disagreements(named: dict[str, dict]) -> None:
    rsf_f   = named["rsf"]["survival_features"]
    gb_f    = named["gradboost"]["survival_features"]
    cox_f   = named["coxnet"]["survival_features"]
    cgb_f   = named["cgb"]["survival_features"]
    rsf_imp = named["rsf"]["summary"].set_index("feature")["importance"]
    gb_imp  = named["gradboost"]["summary"].set_index("feature")["importance"]
    cox_imp = named["coxnet"]["summary"].set_index("feature")["importance"]
    cgb_imp = named["cgb"]["summary"].set_index("feature")["importance"]

    all_four = sorted(set(rsf_f) & set(gb_f) & set(cox_f) & set(cgb_f))
    print(f"\n{'='*70}")
    print("  Feature Selection Summary (Independent — 4 models)")
    print("=" * 70)
    print(f"  RSF features    : {len(rsf_f)}")
    print(f"  GB  features    : {len(gb_f)}")
    print(f"  Coxnet features : {len(cox_f)}")
    print(f"  CGB features    : {len(cgb_f)}")
    print(f"\n  All 4 agree     : {len(all_four):2d}  → {all_four}")

    pairs = [
        (rsf_imp, gb_imp,  rsf_f, gb_f,  "RSF",    "GradBoost"),
        (rsf_imp, cox_imp, rsf_f, cox_f, "RSF",    "Coxnet"),
        (rsf_imp, cgb_imp, rsf_f, cgb_f, "RSF",    "CGB"),
        (gb_imp,  cox_imp, gb_f,  cox_f, "GradBoost",     "Coxnet"),
        (gb_imp,  cgb_imp, gb_f,  cgb_f, "GradBoost",     "CGB"),
        (cox_imp, cgb_imp, cox_f, cgb_f, "Coxnet", "CGB"),
    ]
    for ia, ib, fa, fb, la, lb in pairs:
        print(f"\n── Disagreement check ({la} vs {lb}) ──")
        _warn_disagreements(ia, ib, fa, fb, label_a=la, label_b=lb)