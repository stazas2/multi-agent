# Обзор кода и инфраструктуры Multi‑Agent (GCP)

_Дата: 2025‑10‑19_

## TL;DR — что поправить в первую очередь
1) **Секреты и артефакты в архиве**: в корне есть `.env`, в `agents/*/` лежат `function.zip`, а в `terraform/` — `.terraform/` и `terraform.tfstate`. В Git это коммитить нельзя. **Удали из репо/артефактов, добавь в .gitignore (у тебя уже есть правила — проверь, что на удалённом это не лежит)**. Ключи — **ротируй** и вынеси в **Secret Manager**.
2) **Вебхук без явной аутентификации**: `/webhook/agent-result` должен принимать **ID‑token** (service‑to‑service) или HMAC‑подпись. Сделай обработку **идемпотентной** по `subtask_id`.
3) **Pub/Sub надёжность**: добавь **DLQ**, ретраи с экспонентой, явные `ack_deadline_seconds`, опционально — **ordering key** по `task_id`.
4) **Согласованность состояния**: обновления `TaskContext/SubTask` — через **транзакции**/optimistic‑lock, чтобы не было двойной обработки при редоставке сообщений.
5) **Наблюдаемость**: унифицируй JSON‑логи, прокидывай **trace_id**, введи SLO (латентность/успех) + тревоги.
6) **CI/CD**: не хранить `function.zip`; собирать функции/образы в CI. Добавь `ruff/black/mypy`, security‑скан, Terraform plan/apply с approval.

---

## Что уже нормально ✅
- Разделение ролей: **Orchestrator (FastAPI/Cloud Run)** ↔ **Agents (Cloud Functions 2‑gen)** ↔ **Pub/Sub** ↔ **Firestore**.
- Общие типы/утилиты в `shared/`, SDK‑клиент в `sdk/`.
- Тестовый каркас (`test_system.py`) и локальные env‑настройки.

## Потенциальные риски/бреши 🔥
- Публикация `.env`/`tfstate`/артефактов.
- Плейн‑вебхук (спуфинг ответа агента возможен).
- Отсутствие DLQ → «ядовитые» сообщения теряются.
- Частичная идемпотентность → дубликаты сабтасков на редоставке.
- Скрейпер без чётких timeouts/backoff/robots — риск блокировок.

---

## Архитектура (как по коду)
- **Вход**: `/tasks` создаёт `TaskContext`, далее декомпозиция → очередь сабтасков в Pub/Sub.
- **Агенты**: `research` (сбор/скрейп), `analysis` (обработка), `code` (ген/ревью кода), `validator` (финал). Каждый шлёт POST на `/webhook/agent-result`.
- **Состояние**: Firestore: `TaskContext`, `SubTask`, результаты по агентам.

> Схема ок. Прокачай аутентификацию вебхука, идемпотентность, DLQ и наблюдаемость — будет прод‑уровень.

---

## Модели и SDK Gemini (важно)
Ты пишешь, что **сейчас всё работает на `google-generativeai`** — норм. Тогда:

### Если остаёшься на `google-generativeai`
- **Зафиксируй версии** в `requirements.txt` всех сервисов (у тебя `google-generativeai==0.3.1`).
- Централизуй конфиг модели в `shared/` и **проверь доступность модели при старте** (малый smoke‑тест), чтобы ловить ошибки типа преждевременной «ретирации» модели.
- Выставь **timeouts**, `max_output_tokens`, `top_p/temperature` per‑agent.
- Рекомендуемый маппинг по затратам/качеству:
  - `research`: по умолчанию **flash‑семейство** (быстро/дёшево).
  - `analysis`/`code`/`validator`: **pro‑семейство**, с **fallback** на flash.

**Скетч (google‑generativeai)**
```python
import google.generativeai as genai
from typing import Optional

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL)

def llm_call(prompt: str, system: Optional[str] = None):
    return model.generate_content(
        [
            {"role": "user", "parts": [prompt]}
        ],
        generation_config={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_output_tokens": 2048,
            "stop_sequences": ["</end>"]
        },
        safety_settings=None,
        tools=None,
        request_options={"timeout": 15}
    )
```

### Если захочешь перейти на новый `google-genai` (опционально)
- Унифицируешь клиента, добавляешь **ListModels** на старте и **fail‑fast**, если ID недоступен.
- Даёшь явные таймауты и лимиты токенов.

**Скетч (google‑genai)**
```python
from google import genai
from google.genai.types import GenerateContentConfig

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # тот же ключ
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def llm_call(prompt: str):
    cfg = GenerateContentConfig(
        temperature=0.6, top_p=0.95, max_output_tokens=2048,
        stop_sequences=["</end>"]
    )
    return client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
```

> Итог: раз всё работает — можно **оставаться** на `google-generativeai`. Просто добавь старто‑проверку модели и таймауты. Миграцию на `google-genai` держи «на полке» как улучшение на будущее.

---

## Вебхук: аутентификация + идемпотентность
**Аутентификация**
- Cloud Run: включи **require authentication**.
- Агенты (Cloud Functions) получают **ID‑token** на URL оркестратора и шлют `Authorization: Bearer <token>`.

**Скетч (клиент на агенте)**
```python
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

AUD = os.environ["ORCHESTRATOR_URL"].rstrip("/")

sess = requests.Session()
tok = id_token.fetch_id_token(grequests.Request(), AUD)
sess.headers["Authorization"] = f"Bearer {tok}"
sess.post(f"{AUD}/webhook/agent-result", json=payload, timeout=10)
```

**Идемпотентность**
- Ключ — `subtask_id`.
- Firestore‑транзакция: если `status==completed` — **no‑op** и 200 OK.
- Логи помечай `duplicate=true`.

**Скетч (транзакция Firestore)**
```python
@firestore.transactional
def mark_subtask_done(tx, ref, result):
    snap = ref.get(transaction=tx)
    if snap.exists and snap.to_dict().get("status") == "completed":
        return False
    tx.update(ref, {"status": "completed", "result": result,
                    "completed_at": firestore.SERVER_TIMESTAMP})
    return True
```

---

## Pub/Sub: надёжность
- **DLQ** (dead‑letter topic) + алерты на рост DLQ.
- **Retry policy**: экспонента, лимит доставок.
- **Ack deadline** под фактическое время работы агента (30–60s).
- **Attributes**: `agent_type` для роутинга; **ordering key** по `task_id` при необходимости.
- Большие payload → в **GCS**, в Pub/Sub — только ссылки.

**Terraform‑подсказка**
```hcl
resource "google_pubsub_subscription" "agent" {
  topic  = google_pubsub_topic.agent.id
  name   = "agent-sub"
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.agent_dlq.id
    max_delivery_attempts = 5
  }
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}
```

---

## Firestore: модель и консистентность
- Коллекции: `/tasks/{taskId}`, `/tasks/{taskId}/subtasks/{subtaskId}`, `/tasks/{taskId}/results/{agent}`.
- Все сквозные обновления через **транзакции**.
- TTL‑политики (7–30 дней) для снижения стоимости.
- Сохраняй `checksum` результата агента.

---

## Наблюдаемость и SLO
- **Логи**: JSON с полями `task_id`, `subtask_id`, `agent`, `phase`, `latency_ms`, `duplicate`, `error_type`.
- **Трейсинг**: `trace_id` из оркестратора → Pub/Sub attrs → агенты; экспорт в Cloud Trace.
- **Метрики**: `task_latency_ms` p50/p90/p99, `agent_fail_rate`, `pubsub_redeliveries`, `cost_per_task`.
- **Дашборды/алерты**: SLO, например 95% задач < 60s; 99% успех/24h.

---

## Контейнеры и рантайм
- Запуск под **не‑root** (`USER 65532`).
- Пин версий Python и пакетов; `constraints.txt` или `--require-hashes`.
- `health`/`readiness` endpoints.
- При необходимости — VPC‑коннектор и egress‑policy.

**Пример Dockerfile (orchestrator)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY orchestrator/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
COPY orchestrator/ /app/
COPY shared/ /app/shared/
ENV PYTHONUNBUFFERED=1 PORT=8080
USER appuser
CMD ["python", "main.py"]
```

---

## Security‑чеклист
- [ ] Секреты → **Secret Manager**, ключи ротировать.
- [ ] Раздельные **service accounts** с least‑privilege IAM.
- [ ] Включить **Audit Logs**, ограничить публичный вызов.
- [ ] Скрейпинг: `robots.txt`, таймауты, retries, ограничение размера ответа.
- [ ] Фильтры/нормализация контента до отправки в модель.

---

## Тестирование
- **Unit**: shared‑утилиты, Firestore‑manager (эмулятор).
- **Integration**: Pub/Sub + Firestore эмуляторы + Fake GenAI.
- **E2E**: локально поднимаешь оркестратор, гоняешь `test_system.py`; ассерт: все сабтаски завершены, validator OK, дубликатов нет, метрики/трейсы есть.
- **Нагрузка**: k6/Locust, пиковые 100 RPS на `/tasks`.

**Проверка идемпотентности (pytest)**
```python
def test_webhook_idempotent(client, db):
    payload = {"task_id": "t1", "subtask_id": "s1", "agent_type": "analysis", "result": {"ok": True}}
    r1 = client.post("/webhook/agent-result", json=payload)
    r2 = client.post("/webhook/agent-result", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    snap = db.get_subtask("s1")
    assert snap["status"] == "completed"
```

---

## Terraform / Инфра
- Подключи **dead_letter_policy** и **retry_policy** к подпискам.
- Cloud Run v2: `min/max instances`, `concurrency`, `timeout`, `ingress` только нужный.
- Secret Manager → env vars по версиям.
- Не хранить `function.zip` в репо; собирать в CI и заливать в GCS/Cloud Build.

---

## Research‑agent: гигиена скрейпа
- `User-Agent`, `robots.txt`.
- `requests` с retries/backoff; таймауты на connect/read.
- Ограничь объём, не исполняй непроверенный JS, делай простую кэш‑прослойку (короткий TTL).

**Сессия с ретраями**
```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429,500,502,503,504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))
```

---

## SDK (`sdk/multi_agent_client.py`) — доводка
- Таймауты/ретраи/брейкер, async‑варианты.
- Типизированные результаты (pydantic/dataclasses) и нормальные error‑классы.
- `stream=True`: SSE/пуллинг частичного прогресса.

---

## Экономия
- Роутить low‑value сабтаски на **flash**; pro — для сложных.
- Ограничить **макс токены/задачу**; бюджет на ключ.
- Дедуп подсказок по хешу контента.

---

## TODO — коротко
- [ ] Убрать `.env`, `function.zip`, `.terraform/`, `terraform.tfstate` из репо/артефактов; **ротировать** ключи; Secret Manager.
- [ ] Закрыть вебхук ID‑токеном; добавить идемпотентность + транзакции.
- [ ] Pub/Sub: DLQ, ack‑deadline, ретраи/алерты.
- [ ] Логи/трейсы/метрики + SLO.
- [ ] Docker hardening; настройки Cloud Run.
- [ ] CI/CD: линт/тайп/секьюрити; сборка функций/образов в пайплайне.

— end —

