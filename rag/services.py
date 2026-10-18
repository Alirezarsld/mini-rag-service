"""Write-side operations and orchestration (ingestion, question answering)."""
import hashlib
from dataclasses import asdict

from django.conf import settings
from django.core.cache import cache
from django.db import IntegrityError, transaction

from . import selectors
from .chunking import chunk_text
from .embeddings import get_embedder, normalize
from .generation import generate_answer
from .models import Chunk, Conversation, Document


class DuplicateDocument(Exception):
    """The same text has already been indexed."""

    def __init__(self, document: Document):
        super().__init__(f"Duplicate of document {document.id}")
        self.document = document


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(normalize(text).split()).encode("utf-8")).hexdigest()


def ingest_document(title: str, text: str, source: str = "") -> Document:
    chunks = chunk_text(text, settings.RAG_CHUNK_CHARS, settings.RAG_CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("The document has no text to index.")

    digest = content_hash(text)
    existing = Document.objects.filter(content_hash=digest).first()
    if existing:
        raise DuplicateDocument(existing)

    embedder = get_embedder(settings.RAG_EMBEDDER)
    vectors = embedder.embed(chunks)
    try:
        with transaction.atomic():
            document = Document.objects.create(title=title, source=source, content_hash=digest)
            Chunk.objects.bulk_create(
                Chunk(document=document, index=i, text=t, embedding=vectors[i].tolist(),
                      embedder=embedder.name)
                for i, t in enumerate(chunks)
            )
    except IntegrityError:  # lost a race with a concurrent upload of the same text
        raise DuplicateDocument(Document.objects.get(content_hash=digest)) from None
    selectors.reset_cache()
    return document


def reindex() -> int:
    """Re-embed every chunk with the configured embedder; returns how many were updated."""
    embedder = get_embedder(settings.RAG_EMBEDDER)
    stale = list(Chunk.objects.exclude(embedder=embedder.name))
    for start in range(0, len(stale), 64):
        batch = stale[start:start + 64]
        vectors = embedder.embed([c.text for c in batch])
        for chunk, vector in zip(batch, vectors):
            chunk.embedding = vector.tolist()
            chunk.embedder = embedder.name
        Chunk.objects.bulk_update(batch, ["embedding", "embedder"])
    selectors.reset_cache()
    return len(stale)


def delete_document(document: Document) -> None:
    document.delete()
    selectors.reset_cache()


def search(query: str, top_k: int = 5):
    return selectors.get_retriever().search(query, top_k=top_k)


def _llm_client():
    """Anthropic client, or None when no key is configured (extractive mode)."""
    if not settings.ANTHROPIC_API_KEY:
        return None
    import anthropic

    # The SDK retries connection errors, 429 and 5xx with backoff (max_retries, default 2).
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=settings.RAG_LLM_TIMEOUT)


def create_conversation() -> Conversation:
    return Conversation.objects.create()


def answer_question(query: str, top_k: int = 5, conversation: Conversation | None = None) -> dict:
    """Answer a question; inside a conversation, also record the exchange."""
    result = _answer(query, top_k)
    if conversation is not None:
        conversation.messages.create(role="user", content=query)
        conversation.messages.create(role="assistant", content=result["answer"])
        result = {**result, "conversation_id": str(conversation.id)}
    return result


def _answer(query: str, top_k: int) -> dict:
    """Retrieve, generate, and cache the result per (question, corpus version)."""
    version = selectors.corpus_version()
    raw_key = f"{version}|{top_k}|{settings.RAG_LLM_MODEL}|{query.strip().lower()}"
    cache_key = "rag:answer:" + hashlib.sha256(raw_key.encode()).hexdigest()
    cached = cache.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    hits = search(query, top_k)
    answer = generate_answer(query, hits, client=_llm_client(), model=settings.RAG_LLM_MODEL,
                             max_tokens=settings.RAG_LLM_MAX_TOKENS)
    by_id = {h.chunk_id: h for h in hits}
    result = {
        "answer": answer.text,
        "used_llm": answer.used_llm,
        "llm_error": answer.llm_error,
        "truncated": answer.truncated,
        "usage": answer.usage,
        "citations": [asdict(by_id[i]) for i in answer.citations if i in by_id],
        "sources": [asdict(h) for h in hits],
        "cached": False,
    }
    # A provider error is transient: never cache the fallback answer it produced.
    if not answer.llm_error:
        cache.set(cache_key, result, settings.RAG_ANSWER_CACHE_SECONDS)
    return result
