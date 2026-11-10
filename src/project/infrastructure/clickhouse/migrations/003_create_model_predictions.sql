-- Миграция 003: Создание таблицы предсказаний моделей (OOF кросс-валидации и тестовых скоров)

CREATE TABLE IF NOT EXISTS model_predictions (
    run_id LowCardinality(String),
    run_name LowCardinality(String),
    dataset_split LowCardinality(String),
    dialog_id String,
    participant_index UInt8,
    fold_idx UInt8,
    prob_bot Float32,
    prob_bot_calibrated Float32,
    is_bot Nullable(UInt8),
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
PARTITION BY dataset_split
ORDER BY (run_id, dialog_id, participant_index);
