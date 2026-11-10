-- Миграция 001: Создание таблиц сырого слоя (Bronze Layer / ODS)

CREATE TABLE IF NOT EXISTS raw_dialogues (
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

CREATE TABLE IF NOT EXISTS raw_dialogue_messages (
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
