-- Таблица логирования и аудита продакшен-инференса chat_logs (SSOT синхронизирован с deploy/clickhouse/init.sql)
CREATE TABLE IF NOT EXISTS chat_logs (
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
