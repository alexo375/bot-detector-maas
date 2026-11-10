-- Добавление колонки вероятностей базовых моделей ансамбля для расчёта Concept Drift без повторного инференса
ALTER TABLE chat_logs
ADD COLUMN IF NOT EXISTS base_model_probs Map(String, Array(Float32)) DEFAULT map();
