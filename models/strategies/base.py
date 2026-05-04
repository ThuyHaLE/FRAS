from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Any

class ModelStrategy(ABC):
    """
    Same contract for all survival models.
    
    This implement class to plug new models into SurvivalTrainer / 
    PermutationImportance without changing any other code.
    
    Minimal example:
        class MyStrategy(ModelStrategy):
            name = "MyModel"

            def build(self, random_state): return MyModel(...)
            def fit(self, model, X, y):    return model.fit(X, y)
            def predict_survival(self, model, X):
                return model.predict_survival_function(X)
    """

    @abstractmethod
    def build(self, random_state: int) -> Any:
        """Init and return unfitted model object."""
        ...

    @abstractmethod
    def fit(self, model: Any, X: np.ndarray, y: Any) -> Any:
        """Fit model, return fitted model."""
        ...

    @abstractmethod
    def predict_survival(self, model: Any, X: np.ndarray) -> list:
        """
        Return list of survival functions.
        Each element is a callable: f(t) → P(T > t).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name for logging."""
        ...

    def auto_params(self, n: int) -> None:
        """
        Scale hyperparams by dataset size n.
        Override if needed. Default: no-op.
        """
        pass

    def summary_str(self) -> str:
        """Summary string for logging. Override if needed."""
        return f"[{self.name}]"