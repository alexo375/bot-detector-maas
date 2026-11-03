# Bot Detection Service

Система бинарной классификации для определения ботов и людей в реальном времени по диалогам в чате. Использует двухуровневый ML-ансамбль (градиентные бустинги, RuBERT-tiny2, мета-стекер и постпроцессоры HMM), работающий под управлением NVIDIA Triton и асинхронных очередей.

---

## 1. Быстрый старт

### Требования

* **Docker & Docker Compose** v2.20+
* **Python 3.13** & **`uv`**

### Запуск инфраструктуры

```bash
docker compose up -d
```

---

## 2. Сервисы и порты

| Сервис | Порт | URL / Доступ | Описание |
| :--- | :--- | :--- | :--- |
| **API Gateway** | `8080` | [http://localhost:8080/docs](http://localhost:8080/docs) | REST API и Swagger UI |
| **Grafana** | `3000` | [http://localhost:3000](http://localhost:3000) | Дашборды (`admin` / `admin`) |
| **MLflow** | `5000` | [http://localhost:5000](http://localhost:5000) | Реестр моделей и эксперименты |
| **Prefect UI** | `4200` | [http://localhost:4200](http://localhost:4200) | Оркестратор аудита дрифта и переобучения |
| **Prometheus** | `9090` | [http://localhost:9090](http://localhost:9090) | Метрики и правила алертинга |
| **Triton Server** | `8000` | HTTP `:8000`, gRPC `:8001` | Движок инференса (`bot_ensemble`) |
| **RabbitMQ** | `15672` | [http://localhost:15672](http://localhost:15672) | Очереди сообщений (`guest` / `guest`) |
| **ClickHouse** | `8123` | HTTP `:8123`, TCP `:9000` | Аналитическое хранилище DWH (`bot_detection`) |
| **Redis** | `6379` | `localhost:6379` | Кэш инференса и статусы задач |

---

## 3. Проверка работоспособности и использование API

### Проверка здоровья (Health Check)

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

*(Возвращает `202 Accepted` с `task_id` при промахе кэша или `200 OK` из кэша Redis быстрее чем за 1 мс при попадании).*

---

## 4. Архитектура и поток данных

```
[ Client ] ──> [ API Gateway (:8080) ]
                    │
            ┌───────┴───────┐
            ▼               ▼
      [ Redis Cache ]   [ RabbitMQ Queue ]
                            │
                            ▼
                    [ Inference Worker ]
                            │
                            ▼
                    [ Triton Server ] (Бустинги Уровня 1 + RuBERT + Стекер Уровня 2)
                            │
                            ▼
                    [ ClickHouse DWH ] ──> [ Мониторинг дрифта (Prefect) ] ──> [ Переобучение (MLflow) ]
```

---

## 5. ML-модели, фичи и мониторинг дрифта

* **Ансамбль:** Уровень 1 (CatBoost, LightGBM, XGBoost, эмбеддинги RuBERT-tiny2, символьный TF-IDF, признаки метаданных, типографики и интерактивности) -> Мета-стекер Уровня 2 -> Постпроцессоры HMM / парные постпроцессоры.
* **Привратник (Gatekeeper):** Автоматическая оценка Champion vs. Challenger перед деплоем новых версий моделей.
* **Дрифт и переобучение:** Непрерывный мониторинг дрифта данных (KS-тест / PSI), дрифта предсказаний и концептуального дрифта с помощью cron-задач Prefect, запускающих автоматическое переобучение в MLflow.
* **Внедрение / сброс дрифта (Drift Inject / Reset):**

  ```bash
  uv run just drift-inject
  uv run just drift-reset
  ```

---

## 6. Разработка и тестирование

Управляется через `just` (на базе `poe`):

```bash
uv run just check        # Полная проверка: линтер, типизация, безопасность и все тесты
uv run just check-fast   # Быстрая проверка: линтер + модульные тесты
uv run just fix          # Автоформатирование и исправление ошибок линтера
uv run just test-unit    # Запуск модульных тестов
uv run just test-integration # Запуск интеграционных тестов
uv run just load-smoke   # Нагрузочное тестирование Locust
```
