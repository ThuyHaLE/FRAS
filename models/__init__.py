from models.strategies.base import ModelStrategy
from models.strategies.builtin import RSFStrategy, GradBoostStrategy, CoxnetStrategy, CGBStrategy

from models.survival_model import SurvivalTrainer, enforce_monotone, brier_matrix
from models.ensemble_trainer import EnsembleTrainer, TrainerConfig

from models.utils.filter_features import (
    PermutationImportance,
    SurvivalModelWrapper,
    filter_features,
    filter_features_independent,
    TierThresholds,
    DEFAULT_THRESHOLDS,
    COXNET_THRESHOLDS,
    CGB_THRESHOLDS,
)

# Registry: model_type string → strategy class (backward compat)
_STRATEGY_REGISTRY: dict[str, type[ModelStrategy]] = {
    "rsf"      : RSFStrategy,
    "gradboost": GradBoostStrategy,
    "coxnet"   : CoxnetStrategy,
    "cgb"      : CGBStrategy,
}

__all__ = [
    # Strategies
    "ModelStrategy",
    "RSFStrategy", "GradBoostStrategy", "CoxnetStrategy", "CGBStrategy",
    # Registry
    "_STRATEGY_REGISTRY",
    # Trainer
    "SurvivalTrainer",
    # Ensemble
    "EnsembleTrainer", "TrainerConfig",
    # Feature selection
    "PermutationImportance", "SurvivalModelWrapper",
    "filter_features", "filter_features_independent",
    "TierThresholds", "DEFAULT_THRESHOLDS", "COXNET_THRESHOLDS", "CGB_THRESHOLDS",
    # Utils
    "enforce_monotone", "brier_matrix",
]