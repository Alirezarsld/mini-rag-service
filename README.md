# Mini RAG Service

A small, self-contained **retrieval-augmented generation** API built with Django and
Django REST Framework. Upload documents, search them with hybrid retrieval, and ask questions
that are answered **only from your documents**, with citations.

What sets it apart: first-class **Persian** text handling, and a **benchmark that reports its
own weak spots** instead of only the flattering numbers (see *Retrieval quality*).

It runs fully offline with no API key (extractive answers). Add an Anthropic API key to get
LLM-written answers with the same retrieval and the same guards.

## How it works

```
POST /api/documents/            POST /api/ask/
        │                              │
   chunk_text()                  HybridRetriever.search()
 (sentence-aware,                       │
  overlapping)              ┌───────────┴───────────┐
        │                BM25 (lexical)        Dense vectors
   embed + store                  │             (cosine)
        │                         └─────┬──────────┘
        ▼                      Reciprocal Rank Fusion
   Document / Chunk                     │
   (SQLite or any DB)           MMR re-ranking (diversity)
                                        │
                              generate_answer()
                  LLM (Claude) ──or── extractive fallback
                                        │
                      output guard: valid [n] citations only,
                      length cap, "I don't know" when ungrounded
```

| Stage | What it does |
|---|---|
| Chunking | Paragraph → sentence → word splitting with a repeated tail (overlap) so facts that span a boundary stay findable |
| Normalisation | Unicode-aware tokenizer; unifies Arabic/Persian letter forms (ي→ی, ك→ک), digits and zero-width joiners, and drops stopwords (English + Persian) so "what does a…" cannot match every document |
| BM25 | Okapi BM25 over an inverted index, implemented from scratch |
| Typo rescue | Query terms that never occur in the corpus are mapped to the closest indexed term by character-trigram similarity |
| Dense vectors | Pluggable embedder: hashed n-gram features by default (offline, lexical only) or a multilingual sentence-transformer (semantic). Each chunk records which embedder made its vector, and mismatching chunks are ignored until `rag_reindex` |
| Fusion | Reciprocal Rank Fusion merges the two rankings without needing comparable scores |
| Re-ranking | Maximal Marginal Relevance removes near-duplicate passages |
| Generation | Prompt with numbered sources; the model is told to answer only from them, cite as `[n]`, and ignore instructions found inside sources |
| Claude API | Sources go in escaped `<source>` tags. A `refusal` stop reason is returned as "I don't know" (never replaced by extractive text), `max_tokens` marks the answer `truncated`, and a provider error falls back to an extractive answer with `llm_error` set and is never cached. Token `usage` is returned with every LLM answer |
| Guards | Sources are wrapped in escaped `<source>` tags so document text cannot forge delimiters, invalid citations are stripped, answers are length-capped, an ungrounded question returns *"I don't know…"*, and an LLM failure degrades to an extractive answer |
| Caching | Answers are cached per (question, corpus version); changing the corpus or the embedder makes old entries unreachable |
| Ingestion | Duplicate text (after normalisation) is rejected with `409` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DJANGO_DEBUG=1              # local development
python manage.py migrate
python manage.py rag_ingest examples/*.md
python manage.py runserver
```

```bash
# search
curl -s localhost:8000/api/search/ -H 'Content-Type: application/json' \
  -d '{"query": "how do workers get jobs from redis", "top_k": 3}'

# ask
curl -s localhost:8000/api/ask/ -H 'Content-Type: application/json' \
  -d '{"query": "What does VACUUM do?"}'

# add a document (JSON) or upload a file
curl -s localhost:8000/api/documents/ -H 'Content-Type: application/json' \
  -d '{"title": "Notes", "text": "Any text you want to be able to search."}'
curl -s localhost:8000/api/documents/upload/ -F file=@examples/celery.md
```

### Example output

Real answers from the retrieval core (default offline mode, no API key) over the bundled
documents. English, from `examples/`:

```
Q: What does VACUUM do?
A: VACUUM reclaims storage from dead tuples, and autovacuum normally handles it. [1]
   → citation: postgresql.md
```

Persian, from `benchmarks/corpus/persian-*.md`, with Arabic/Persian letter forms and ZWNJ
normalised:

```
Q: هش کردن گذرواژه چطور انجام می‌شود
A: رمز عبور هرگز نباید به‌صورت متن ساده ذخیره شود و باید با یک تابع هش امن نگهداری شود. [1]
   → citation: persian-security.md
```

In the API the same answer comes back as JSON from `POST /api/ask/` with `answer`,
`citations`, `sources`, `used_llm`, `usage` and `cached`.

Open <http://localhost:8000/> for a one-page UI that calls `/api/ask/` and shows the cited passages.

To use an LLM for answers: `export ANTHROPIC_API_KEY=...` (the model is set by `RAG_LLM_MODEL`).

### Semantic search (optional)

The default embedder only understands shared words. To match paraphrases and synonyms:

```bash
pip install -r requirements-semantic.txt     # sentence-transformers (downloads a model on first use)
export RAG_EMBEDDER=sentence-transformers    # or st:<any sentence-transformers model name>
python manage.py rag_reindex                 # re-embed what is already indexed
python benchmarks/run.py --embedder sentence-transformers
```

> The semantic embedder is not exercised by this repository's test suite (it needs a model
> download). Run the benchmark above to check the quality on your machine.

### Docker

```bash
cp .env.example .env     # set DJANGO_SECRET_KEY
docker compose up --build
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health/` | Liveness probe |
| GET / POST | `/api/documents/` | List / index a document (`title`, `text`, optional `source`) |
| POST | `/api/documents/upload/` | Index a UTF-8 `.md`/`.txt` file (`multipart/form-data`, field `file`, optional `title`, max 1 MB) |
| GET / DELETE | `/api/documents/<id>/` | Inspect / remove a document |
| POST | `/api/search/` | `{query, top_k}` → ranked chunks with BM25 and vector scores |
| POST | `/api/conversations/` | Start a conversation → `{id}` |
| POST | `/api/ask/` | `{query, top_k, conversation_id?}` → `answer`, `citations`, `sources`, `used_llm`, `llm_error`, `truncated`, `usage`, `cached` |

Set `RAG_API_KEY` to require an `X-API-Key` header on everything except `/api/health/`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DJANGO_SECRET_KEY` | *(required unless `DJANGO_DEBUG=1`)* | Django secret |
| `DJANGO_DEBUG` | `0` | Debug mode |
| `RAG_API_KEY` | empty | Enables API-key auth |
| `RAG_EMBEDDER` | `hashing` | `hashing`, `sentence-transformers` or `st:<model>` |
| `ANTHROPIC_API_KEY` | empty | Enables LLM answers |
| `RAG_LLM_MODEL` | `claude-sonnet-5-5` | Claude model ID used for answers |
| `RAG_LLM_MAX_TOKENS` | `600` | `max_tokens` for each answer |
| `RAG_LLM_TIMEOUT` | `30` | Request timeout in seconds (the SDK also retries 429/5xx/connection errors) |
| `RAG_CHUNK_CHARS` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Chunk size and overlap |
| `RAG_THROTTLE` | `120/min` | Per-client rate limit |

## Retrieval quality

Retrieval is measured, not guessed. `benchmarks/run.py` runs 56 labelled questions against a
12-document corpus and compares three modes. Questions are grouped by how hard they are:
*lexical* (shares words with the passage), *paraphrase* (almost no shared vocabulary),
*typo* and *persian*. Results with the default hashing embedder (`hit@3 / MRR`):

| mode | lexical | paraphrase | typo | persian | all |
|---|---|---|---|---|---|
| bm25 | 0.95 / 0.95 | 0.65 / 0.62 | 0.88 / 0.88 | 1.00 / 1.00 | 0.84 / 0.83 |
| vector | 0.86 / 0.86 | 0.30 / 0.28 | 0.12 / 0.12 | 1.00 / 1.00 | 0.57 / 0.56 |
| hybrid | 0.95 / 0.95 | 0.65 / 0.55 | 0.88 / 0.88 | 1.00 / 1.00 | 0.84 / 0.80 |

What this shows, honestly:

- With the default embedder, **hybrid is no better than BM25**: the hashed vectors are
  lexical too, so they add no new information. The *paraphrase* column is where a semantic
  embedder is expected to help, and `--embedder sentence-transformers` lets you check that.
- Dropping stopwords mattered: before it, hybrid was *worse* than BM25 because frequent words
  such as "what" and "does" pulled in unrelated documents.
- The corpus is small and synthetic, so treat the numbers as a regression check rather than a
  claim about production quality. Build a case file from your own documents for real numbers.

For your own data, `rag_eval` replays labelled questions through the live index; its
`--min-hit-rate` flag makes it a CI quality gate:

```bash
python manage.py rag_eval examples/eval_cases.json --k 5
# cases=6  hit_rate@5=1.000  mrr=1.000
```

## Project layout

```
rag/
  chunking.py     sentence-aware overlapping chunker      ┐
  embeddings.py   tokenizer + pluggable embedder          │
  bm25.py         Okapi BM25                              │ framework-independent core
  retrieval.py    hybrid search, RRF, MMR, typo rescue    │ (pure Python + numpy)
  generation.py   prompt, LLM call, output guards         │
  evaluation.py   hit-rate@k and MRR                      ┘
benchmarks/       corpus, labelled questions, run.py (bm25 vs vector vs hybrid)
  selectors.py    read queries + retriever cache (Django)
  services.py     writes and orchestration (Django)
  views.py        thin HTTP layer
```

Views only validate input and call `services` (writes) or `selectors` (reads); the retrieval
core knows nothing about Django, which keeps it unit-testable and reusable.

## Tests

```bash
python manage.py test            # core unit tests + API tests
python -m unittest rag.tests.test_core   # core only, no Django needed
python benchmarks/run.py                 # retrieval-quality comparison
```

## Limitations

- The default embedder is lexical and will not match synonyms (see the table above). Use the
  optional sentence-transformers embedder for semantic matching.
- Embeddings are stored as JSON and the index is rebuilt in memory when the corpus changes.
  That is fine for thousands of chunks; for larger corpora move vectors to pgvector (or another
  vector index) and BM25 to database full-text search.
- The retriever cache and the answer cache are per process (`LocMemCache`). With several
  workers, use a shared cache backend such as Redis.
- Documents are plain text; PDF/HTML extraction is out of scope.
- The stopword lists are short and the benchmark is small; both are starting points.

## License

MIT
