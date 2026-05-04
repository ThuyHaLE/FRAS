from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Any
from .base import ModelStrategy

from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    ComponentwiseGradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis

# max_features auto-resolution thresholds cho RSF
_SMALL_FEATURE_THRESHOLD  = 10   # <= 10 → 1.0  (use all features)
_MIDDLE_FEATURE_THRESHOLD = 20   # <= 20 → 0.7

#-----------------------------------------------#
# RandomSurvivalForest                          #
#-----------------------------------------------#

@dataclass
class RSFStrategy(ModelStrategy):
    """
    Random Survival Forest.

    max_features=None → auto-resolve theo n_features:
        n <= 10  → 1.0
        n <= 20  → 0.7
        n >  20  → "sqrt"
    """
    n_features: int        = 10
    n_trees: int           = 250
    min_samples_split: int = 20
    min_samples_leaf: int  = 10
    max_features: Any      = None  # None → auto

    @property
    def name(self) -> str:
        return "RSF"

    def _resolve_max_features(self) -> Any:
        if self.max_features is not None:
            return self.max_features
        if self.n_features <= _SMALL_FEATURE_THRESHOLD:
            return 1.0
        if self.n_features <= _MIDDLE_FEATURE_THRESHOLD:
            return 0.7
        return "sqrt"

    def auto_params(self, n: int) -> None:
        self.min_samples_leaf  = max(math.floor(n * 0.05), 5)
        self.min_samples_split = max(math.floor(n * 0.10), 10)
        self.n_trees           = int(min(300, max(150, n * 1.2)))

    def build(self, random_state: int) -> RandomSurvivalForest:
        return RandomSurvivalForest(
            n_estimators      = self.n_trees,
            min_samples_split = self.min_samples_split,
            min_samples_leaf  = self.min_samples_leaf,
            max_features      = self._resolve_max_features(),
            n_jobs            = -1,
            random_state      = random_state,
        )

    def fit(self, model: RandomSurvivalForest, X: np.ndarray, y: Any) -> Any:
        return model.fit(X, y)

    def predict_survival(self, model: RandomSurvivalForest, X: np.ndarray) -> list:
        return model.predict_survival_function(X)

    def summary_str(self) -> str:
        mf = self._resolve_max_features()
        return (
            f"[RSF] trees={self.n_trees}  max_features={mf!r}  "
            f"leaf={self.min_samples_leaf}  split={self.min_samples_split}"
        )

#-----------------------------------------------#
# GradientBoostingSurvivalAnalysis              #
#-----------------------------------------------#

@dataclass
class GradBoostStrategy(ModelStrategy):
    """Gradient Boosting Survival Analysis."""
    n_trees: int             = 150
    lr: float                = 0.05
    max_depth: int           = 2
    min_samples_split: int   = 20
    min_samples_leaf: int    = 10
    subsample: float         = 0.75
    n_iter_no_change: int    = 20
    validation_fraction: float = 0.20

    @property
    def name(self) -> str:
        return "GradBoost"

    def auto_params(self, n: int) -> None:
        self.min_samples_leaf    = max(math.floor(n * 0.05), 5)
        self.min_samples_split   = max(math.floor(n * 0.10), 10)
        self.n_trees             = int(min(200, max(100, n * 0.8)))
        self.lr                  = 0.05 if n < 300 else 0.03
        self.max_depth           = 2    if n < 300 else 3
        self.subsample           = 0.75 if n < 300 else 0.8
        n_train_fold             = n * (1 - 1 / 5)
        self.validation_fraction = float(min(0.25, max(0.20, 30 / n_train_fold)))

    def build(self, random_state: int) -> GradientBoostingSurvivalAnalysis:
        return GradientBoostingSurvivalAnalysis(
            n_estimators        = self.n_trees,
            learning_rate       = self.lr,
            max_depth           = self.max_depth,
            min_samples_split   = self.min_samples_split,
            min_samples_leaf    = self.min_samples_leaf,
            subsample           = self.subsample,
            n_iter_no_change    = self.n_iter_no_change,
            validation_fraction = self.validation_fraction,
            tol                 = 1e-4,
            random_state        = random_state,
        )

    def fit(self, model: GradientBoostingSurvivalAnalysis, X: np.ndarray, y: Any) -> Any:
        return model.fit(X, y)

    def predict_survival(self, model: GradientBoostingSurvivalAnalysis, X: np.ndarray) -> list:
        return model.predict_survival_function(X)

    def summary_str(self) -> str:
        return (
            f"[GradBoost] trees={self.n_trees}  lr={self.lr}  "
            f"depth={self.max_depth}  sub={self.subsample}  "
            f"val_frac={self.validation_fraction:.2f}"
        )

#-----------------------------------------------#
# CoxnetSurvivalAnalysis                        #
#-----------------------------------------------#

@dataclass
class CoxnetStrategy(ModelStrategy):
    """Cox Elastic Net (L1 + L2 regularization)."""
    l1_ratio: float = 0.5
    n_alphas: int   = 100
    max_iter: int   = 100_000

    @property
    def name(self) -> str:
        return "Coxnet"

    def build(self, random_state: int) -> CoxnetSurvivalAnalysis:
        return CoxnetSurvivalAnalysis(
            l1_ratio           = self.l1_ratio,
            n_alphas           = self.n_alphas,
            max_iter           = self.max_iter,
            normalize          = False,       # StandardScaler has been applied in preprocess step
            fit_baseline_model = True,        # required for predict_survival_function
        )

    def fit(self, model: CoxnetSurvivalAnalysis, X: np.ndarray, y: Any) -> Any:
        return model.fit(X, y)

    def predict_survival(self, model: CoxnetSurvivalAnalysis, X: np.ndarray) -> list:
        return model.predict_survival_function(X)

    def summary_str(self) -> str:
        return (
            f"[Coxnet] l1_ratio={self.l1_ratio}  "
            f"n_alphas={self.n_alphas}  max_iter={self.max_iter}"
        )

#-----------------------------------------------#
# ComponentwiseGradientBoostingSurvivalAnalysis #
#-----------------------------------------------#

@dataclass
class CGBStrategy(ModelStrategy):
    """
    Componentwise Gradient Boosting — linear base learners.
    Only n_estimators, lr, and loss are valid parameters (no max_depth or subsample).
    """
    n_estimators: int = 100
    lr: float         = 0.1
    loss: str         = "coxph"   # "coxph" | "squared"

    @property
    def name(self) -> str:
        return "CGB"

    def auto_params(self, n: int) -> None:
        self.n_estimators = int(min(200, max(100, n * 0.6)))
        self.lr           = 0.1 if n < 300 else 0.05

    def build(self, random_state: int) -> ComponentwiseGradientBoostingSurvivalAnalysis:
        return ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators  = self.n_estimators,
            learning_rate = self.lr,
            loss          = self.loss,
            random_state  = random_state,
        )

    def fit(self, model: ComponentwiseGradientBoostingSurvivalAnalysis, X: np.ndarray, y: Any) -> Any:
        return model.fit(X, y)

    def predict_survival(self, model: ComponentwiseGradientBoostingSurvivalAnalysis, X: np.ndarray) -> list:
        return model.predict_survival_function(X)

    def summary_str(self) -> str:
        return (
            f"[CGB] n_estimators={self.n_estimators}  "
            f"lr={self.lr}  loss={self.loss!r}"
        )