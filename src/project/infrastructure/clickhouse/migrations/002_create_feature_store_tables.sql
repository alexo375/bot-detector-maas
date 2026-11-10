-- Миграция 002: Создание таблиц Feature Store слоя с версионированием через snapshot_id (Gold Layer)

CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id UInt32,
    snapshot_tag LowCardinality(String),
    features_count UInt16,
    feature_names Array(String),
    description String,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
ORDER BY snapshot_id;

CREATE TABLE IF NOT EXISTS dialogue_features (
    dialog_id String,
    participant_index UInt8,
    dataset_type LowCardinality(String),
    snapshot_id UInt32,
    snapshot_tag LowCardinality(String),
    features Map(String, Float32),
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
PARTITION BY snapshot_id
ORDER BY (snapshot_id, dataset_type, dialog_id, participant_index);
