import optuna


class OptunaEarlyStoppingCallback:
    """Callback для ранней остановки Optuna при отсутствии улучшений в течение patience шагов."""

    def __init__(self, patience: int = 50):
        self.patience = patience
        self.best_value = float("inf")
        self.no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Останавливает исследование Optuna при достижении лимита шагов без улучшения."""
        if trial.value is not None and trial.value < self.best_value:
            self.best_value = trial.value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                study.stop()
