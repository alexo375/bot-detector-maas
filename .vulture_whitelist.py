# Vulture whitelist для ложноположительных срабатываний (публичные контракты, FastAPI, Pydantic, Pytest, Locust, Protobuf и notebooks).

from typing import Any

_any: Any = None

# FastAPI роуты, middleware и обработчики исключений (регистрируются декораторами)
_any.domain_validation_handler
_any.model_artifact_handler
_any.infrastructure_handler
_any.project_error_handler
_any.sync_drift_metrics_middleware
_any.alertmanager_webhook
_any.liveness_check
_any.readiness_check
_any.predict_dialogues

# Поля Pydantic-схем и настроек (Alertmanager webhook, Health, SettingsConfigDict, validators)
_any.startsAt
_any.endsAt
_any.generatorURL
_any.fingerprint
_any.receiver
_any.groupLabels
_any.commonLabels
_any.commonAnnotations
_any.externalURL
_any.groupKey
_any.truncatedAlerts
_any.alert
_any.dispatched
_any.poll_url
_any.models_loaded
_any.rabbitmq_connected
_any.model_config
_any._resolve_url

# Torch CUDA/cuDNN детерминизм (атрибуты torch.backends.cudnn)
_any.deterministic
_any.benchmark

# Поля FeatureStore-схем (проецируются динамически через dataclasses.fields / SQL-схему)
_any.is_last_sender
_any.n_messages
_any.dialog_total_messages
_any.msg_share
_any.max_consecutive_msgs
_any.mean_words_per_msg
_any.max_words_per_msg
_any.mean_chars_per_msg
_any.max_chars_per_msg
_any.newline_count
_any.question_count
_any.partner_n_messages
_any.pure_bot_count
_any.total_words
_any.has_newline
_any.has_dot_end
_any.has_question
_any.has_emoji
_any.has_markdown
_any.has_code_syntax
_any.has_echo_marker
_any.has_spam_template
_any.mean_words_ratio
_any.n_messages_ratio
_any.jaccard_overlap
_any.partner_has_question
_any.is_first_sender
_any.pos_p0
_any.pos_p1
_any.first_msg_len_self
_any.last_msg_len_self
_any.std_chars_per_msg
_any.has_double_self
_any.has_double_partner
_any.diff_chars
_any.char_ratio
_any.diff_words
_any.diff_mean_len
_any.response_len_ratio_mean
_any.response_len_ratio_max
_any.turn_length_correlation
_any.is_even_dialogue

# Метаданные sklearn и dataclass-метрик MLflow (asdict)
_any.classes_
_any.fixed_vocabulary_
_any._idf_diag
_any.catboost_val_loss
_any.lightgbm_val_loss
_any.xgboost_val_loss
_any.meta_stacker_val_loss

# Публичные методы и поля, используемые исследовательскими ноутбуками (notebooks/) и оффлайн-экспортом
_any.load_predictions
_any.test_raw
_any.export_rubert_onnx
_any.collect_tokenizer_files

# Pytest-плагины, маркеры и тестовые FastAPI-роуты/моки
_any.pytest_plugins
_any.pytestmark
_any.prefect_isolated_server
_any.trigger_domain_error
_any.trigger_model_error
_any.trigger_infra_error
_any.trigger_project_error
_any.side_effect
_any.__aenter__
_any.__aexit__

# Locust load-тесты (вызываются движком Locust через метаклассы и @task)
_any.BotDetectorUser
_any.wait_time
_any.on_start
_any.predict_cache_miss
_any.predict_cache_hit
_any.async_pull
