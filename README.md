# RAG Assistant

RAG Assistant - это Python-проект для ответов на вопросы по локальным
документам. Проект загружает файлы из `rag_assistant/docs`, разбивает текст на
чанки, создает embeddings, сохраняет их в Qdrant, ищет релевантный контекст по
вопросу пользователя и генерирует ответ через Groq/OpenAI-compatible LLM API.

Проект подходит как учебный portfolio-проект для Junior AI Automation / AI
Engineer: в нем есть ingestion, chunking, vector database, semantic search,
CLI-команды, HTTP API и Docker Compose для инфраструктуры.

## Что Делает Проект

- Загружает локальные документы из `rag_assistant/docs`.
- Очищает текст от лишних пробелов и переносов.
- Разбивает документы на чанки с overlap.
- Создает embeddings через `sentence-transformers`.
- Сохраняет в Qdrant vector + payload: `text`, `source`, `chunk_id`.
- Ищет релевантные чанки по вопросу пользователя.
- Собирает контекст из найденных чанков.
- Генерирует ответ через LLM.
- Выводит ответ и источники через CLI.
- Предоставляет HTTP API на FastAPI.

## Архитектура RAG

```text
Документы
   |
   v
Загрузка и очистка текста
   |
   v
Chunking с overlap
   |
   v
Embedding model
   |
   v
Qdrant vector database
   |
   v
Вопрос пользователя -> embedding вопроса -> semantic search
   |
   v
Релевантные чанки + metadata
   |
   v
Prompt: вопрос + найденный контекст
   |
   v
LLM-ответ
```

## Основные Файлы

- `main.py` - CLI-команды `ingest`, `ask`, `reindex`.
- `api.py` - FastAPI HTTP API: `/health`, `/ask`, `/reindex`.
- `config.py` - настройки проекта из `.env`.
- `ingest.py` - загрузка документов, очистка, chunking, embeddings, upload в Qdrant.
- `retriever.py` - embedding вопроса и поиск в Qdrant.
- `llm.py` - сбор контекста, генерация ответа и извлечение источников.
- `docs/` - локальные документы для индексации.

## Стек Технологий

- Python
- FastAPI
- Uvicorn
- Qdrant
- Docker Compose
- Sentence Transformers
- OpenAI Python SDK
- Groq OpenAI-compatible API
- Pydantic Settings
- python-dotenv

## Быстрый Старт

```bash
docker compose up -d
pip install -r requirements.txt
python -m rag_assistant.main ingest
python -m rag_assistant.main ask "что такое FastAPI?"
```

## Запуск Qdrant

Qdrant запускается через `docker-compose.yml`:

```bash
docker compose up -d
```

Qdrant будет доступен по адресу:

```text
http://localhost:6333
```

Остановить Qdrant:

```bash
docker compose down
```

Данные сохраняются в Docker volume `qdrant_storage`, поэтому база не исчезает
после обычного перезапуска контейнера.

## Настройка `.env`

Создайте `.env` в корне проекта. Пример лежит в `rag_assistant/.env.example`:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COLLECTION_NAME=rag_assistant_docs
LLM_API_KEY=your_api_key_here
MODEL_NAME=openai/gpt-oss-20b
```

Важно:

- `LLM_API_KEY` - API key для Groq.
- Не коммитьте `.env` в GitHub.
- Если меняете `EMBEDDING_MODEL`, проверьте, что `vector_size` соответствует размерности embeddings.

## Индексация Документов

Положите документы в:

```text
rag_assistant/docs
```

Затем выполните:

```bash
python -m rag_assistant.main ingest
```

`ingest`:

1. Создает коллекцию в Qdrant, если ее нет.
2. Загружает документы из `rag_assistant/docs`.
3. Делит документы на чанки.
4. Создает embeddings только по `chunk["text"]`.
5. Сохраняет в Qdrant:

```python
{
    "text": "...",
    "source": "fast.md",
    "chunk_id": 3
}
```

`ingest` не удаляет коллекцию автоматически.

## Задать Вопрос Через CLI

```bash
python -m rag_assistant.main ask "что такое FastAPI?"
```

`ask`:

1. Проверяет, что коллекция существует.
2. Создает embedding вопроса.
3. Ищет похожие чанки в Qdrant.
4. Фильтрует чанки по `min_score`.
5. Отправляет контекст и вопрос в LLM.
6. Печатает ответ и источники.

Пример вывода:

```text
=== ANSWER ===

FastAPI - это современный высокопроизводительный веб-фреймворк для Python...

=== SOURCES ===
- fast.md, 2
- fast.md, 3
```

## Переиндексация

```bash
python -m rag_assistant.main reindex
```

`reindex`:

1. Удаляет существующую коллекцию, если она есть.
2. Создает коллекцию заново.
3. Повторно индексирует документы из `rag_assistant/docs`.

Используйте `reindex`, когда документы изменились или нужно полностью
пересобрать векторную базу.

## Запуск FastAPI

После установки зависимостей и запуска Qdrant:

```bash
uvicorn rag_assistant.api:app --reload
```

API будет доступен по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### `GET /health`

Проверяет, что API запущен:

```bash
curl http://127.0.0.1:8000/health
```

Пример ответа:

```json
{
  "status": "ok",
  "collection": "rag_assistant_docs"
}
```

### `POST /ask`

Задает вопрос по уже проиндексированной коллекции:

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"что такое FastAPI?\"}"
```

Пример ответа:

```json
{
  "answer": "FastAPI - это современный веб-фреймворк для Python..."
}
```

### `POST /reindex`

Переиндексирует документы через API:

```bash
curl -X POST http://127.0.0.1:8000/reindex
```

## Настройки По Умолчанию

В `config.py` заданы параметры:

```python
vector_size: int = 384
chunk_size: int = 100
chunk_overlap: int = 20
top_k: int = 5
min_score: float = 0.5
```

Их можно переопределить через `.env`.

## Почему Docker Compose

`docker-compose.yml` нужен, чтобы описать инфраструктуру проекта в файле, а не
запоминать длинные `docker run` команды.

Преимущества:

- Qdrant запускается одной командой.
- Порты и volume описаны явно.
- Данные сохраняются между перезапусками.
- Другой разработчик может поднять окружение без ручной настройки.
- В будущем легко добавить другие сервисы: API, Redis, PostgreSQL.


