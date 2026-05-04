# models/strategies/__init__.py

from .base import ModelStrategy
from .builtin import RSFStrategy, GradBoostStrategy, CoxnetStrategy, CGBStrategy

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
    "_STRATEGY_REGISTRY"
]