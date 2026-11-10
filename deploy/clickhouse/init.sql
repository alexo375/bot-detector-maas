-- Инициализация аналитического хранилища ClickHouse DWH (SSOT синхронизирован с миграциями 001-005)
CREATE DATABASE IF NOT EXISTS bot_detection;

CREATE TABLE IF NOT EXISTS bot_detection.schema_migrations (
    version String,
    applied_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
ORDER BY version;

CREATE TABLE IF NOT EXISTS bot_detection.raw_dialogues (
    dataset_type LowCardinality(String),
    dialog_id String,
    participant_index UInt8,
    is_bot Nullable(UInt8),
    num_messages UInt16,
    duration_sec UInt32,
    messages_json String,
    original_order_idx UInt32 DEFAULT 0,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
ORDER BY (dataset_type, dialog_id, participant_index);

CREATE TABLE IF NOT EXISTS bot_detection.raw_dialogue_messages (
    dataset_type LowCardinality(String),
    dialog_id String,
    msg_idx UInt16,
    participant_index UInt8,
    text String,
    char_count UInt32,
    original_order_idx UInt32 DEFAULT 0,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (dataset_type, dialog_id, msg_idx);

CREATE TABLE IF NOT EXISTS bot_detection.feature_snapshots (
    snapshot_id UInt32,
    snapshot_tag LowCardinality(String),
    features_count UInt16,
    feature_names Array(String),
    description String,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
ORDER BY snapshot_id;

CREATE TABLE IF NOT EXISTS bot_detection.dialogue_features (
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

CREATE TABLE IF NOT EXISTS bot_detection.model_predictions (
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

CREATE TABLE IF NOT EXISTS bot_detection.chat_logs (
    dialog_id String,
    timestamp DateTime64(3, 'UTC') DEFAULT now64(3),
    num_messages UInt16,
    messages_json String,
    features_p0 Array(Float32),
    features_p1 Array(Float32),
    prob_p0 Float32,
    prob_p1 Float32,
    base_model_probs Map(String, Array(Float32)) DEFAULT map(),
    is_bot_p0 UInt8,
    is_bot_p1 UInt8,
    verdict String,
    confidence String,
    cached UInt8,
    latency_ms Float32,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, dialog_id)
SETTINGS index_granularity = 8192;

INSERT INTO bot_detection.schema_migrations (version) VALUES
    ('001_create_raw_tables.sql'),
    ('002_create_feature_store_tables.sql'),
    ('003_create_model_predictions.sql'),
    ('004_create_chat_logs.sql'),
    ('005_add_base_model_probs_to_chat_logs.sql');
