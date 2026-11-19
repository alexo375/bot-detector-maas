# Bot Detection Service

Система бинарной классификации для определения ботов и людей в реальном времени по диалогам в чате. Использует двухуровневый ML-ансамбль, работающий с помощью NVIDIA Triton Inferece Server и асинхронных очередей

---

## 1. Быстрый старт

### Требования

* **Docker & Docker Compose** v2.20+
* **Python 3.13** & **`uv`**

### Настройка окружения и запуск инфраструктуры

```bash
cp .env.example .env
# При необходимости отредактируйте .env
docker compose up -d
```

---

## 2. Сервисы и порты

| Сервис | Порт | URL / Доступ | Описание |
| :--- | :--- | :--- | :--- |
| **API Gateway** | `8080` | [http://localhost:8080/docs](http://localhost:8080/docs) | REST API и Swagger UI |
| **Grafana** | `3000` | [http://localhost:3000](http://localhost:3000) | Дашборды (`admin` / `admin`) |
| **MLflow** | `5000` | [http://localhost:5000](http://localhost:5000) | Реестр моделей и эксперименты |
| **PostgreSQL** | `5432` | `127.0.0.1:5432` | Хранилище метаданных и реестра MLflow |
| **Prefect UI** | `4200` | [http://localhost:4200](http://localhost:4200) | Оркестратор регулярных задач |
| **Prometheus** | `9090` | [http://localhost:9090](http://localhost:9090) | Метрики и правила алертинга |
| **Alertmanager** | `9093` | [http://localhost:9093](http://localhost:9093) | Маршрутизация алертов |
| **Triton Server** | `8000` | HTTP `:8000`, gRPC `:8001` | Движок инференса |
| **RabbitMQ** | `15672` | [http://localhost:15672](http://localhost:15672) | Очереди сообщений (`guest` / `guest`) |
| **ClickHouse** | `8123` | HTTP `:8123`, TCP `:9000` | Аналитическое хранилище DWH (`bot_detection`) |
| **Redis** | `6379` | `localhost:6379` | Кэш инференса и статусы задач |

---

## 3. Проверка работоспособности и использование API

### Проверка здоровья

```bash
curl -s http://localhost:8080/health | jq .
```

### Предсказание для диалога (`POST /predict`)

```bash
curl -s -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "dialogues": [
      {
        "dialog_id": "test_001",
        "messages": [
          {"participant_index": 0, "text": "Здравствуйте! Где мой заказ?"},
          {"participant_index": 1, "text": "Добрый день! Заказ передан в доставку."}
        ]
      }
    ]
  }' | jq .
```

*(Возвращает `202 Accepted` с `task_id` при промахе кэша или `200 OK` из кэша Redis).*

---

## 4. Архитектура и поток данных

```
[ Client ] ──> [ API Gateway ]
                    │
            ┌───────┴───────┐
            ▼               ▼
      [ Redis Cache ]   [ RabbitMQ Queue ]
                            │
                            ▼
                    [ Inference Worker ]
                            │
                            ▼
                    [ Triton Server ]
                            ▼
                    [ ClickHouse DWH ] ──> [ Мониторинг дрифта ] ──> [ Переобучение ]
```

---

## 5. ML-модели, фичи и мониторинг дрифта

* **Ансамбль:**
  * Модели градиентного бустинга: CatBoost, LightGBM, XGBoost
  * LogReg мета-стекер
  * Опциональные постпроцессоры
* **Автоматическая оценка** `Champion` vs. `Challenger` перед деплоем новых версий моделей
* **Мониторинг:** отслеживание data / concept / prediction drift статистическими тестами с помощью Prefect
* **Трегинг эксперементов** с помощью MLflow
* Возможность в любой момент запустить **задачи дообучения** на новых данных или **отката моделей** к предыдущим версиям

---

## 6. Разработка и тестирование

```bash
uv run just check           # Полная проверка: статика, безопасность и тесты
uv run just check-fast      # Быстрая проверка: вся статика + модульные тесты
uv run just ci              # Локальная имитация GitLab CI (статика, безопасность, тесты, compose)
uv run just fix             # Автоформатирование и исправление ошибок линтера
uv run just verify-system   # Проверка доступности Triton, API Gateway и /metrics
uv run just load-smoke      # Нагрузочное smoke-тестирование Locust
uv run just retrain-fast    # Ускоренное переобучение в Prefect Worker
uv run just retrain-full    # Полноe переобучение в Prefect Worker
uv run just rollback        # Откат ансамбля моделей на предыдущую валидную версию
uv run poe test-unit        # Запуск модульных тестов
uv run poe test-integration # Запуск интеграционных тестов
```
