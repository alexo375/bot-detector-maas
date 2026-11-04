from typing import Any

import numpy as np


class FeatureImportanceExtractor:
    """Единый экстрактор вектора важности признаков из моделей разных семейств."""

    @staticmethod
    def extract(model: Any) -> np.ndarray | None:
        """Извлечение вектора важности признаков из произвольной модели."""
        if isinstance(model, np.ndarray):
            return model
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_)
        if hasattr(model, "get_feature_importance"):
            return np.asarray(model.get_feature_importance())
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            return np.abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
        return None


__all__ = ["FeatureImportanceExtractor"]
