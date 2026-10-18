"""Read-side queries. Nothing in here writes to the database."""
import numpy as np
from django.conf import settings
from django.db.models import Count, Max

from .embeddings import get_embedder
from .models import Chunk, Document
from .retrieval import ChunkRecord, HybridRetriever

_state: dict = {"version": None, "retriever": None}


def corpus_version() -> str:
    """Changes whenever chunks are added/removed or the embedder changes; keys the caches."""
    agg = Chunk.objects.aggregate(n=Count("id"), top=Max("id"))
    return f"{settings.RAG_EMBEDDER}:{agg['n']}-{agg['top'] or 0}"


def stale_chunk_count() -> int:
    """Chunks embedded with a different model than the configured one (need ``rag_reindex``)."""
    return Chunk.objects.exclude(embedder=get_embedder(settings.RAG_EMBEDDER).name).count()


def reset_cache() -> None:
    _state["version"] = None
    _state["retriever"] = None


def list_documents():
    return Document.objects.annotate(chunk_count=Count("chunks"))


def get_document(pk: int):
    return Document.objects.annotate(chunk_count=Count("chunks")).filter(pk=pk).first()


def get_retriever() -> HybridRetriever:
    """Return a retriever for the current corpus, rebuilding it only when it changed."""
    version = corpus_version()
    if _state["version"] != version or _state["retriever"] is None:
        embedder = get_embedder(settings.RAG_EMBEDDER)
        records = [
            ChunkRecord(
                id=c.id, document_id=c.document_id, document_title=c.document.title,
                index=c.index, text=c.text, embedding=np.asarray(c.embedding, dtype=np.float32),
            )
            for c in Chunk.objects.filter(embedder=embedder.name).select_related("document")
        ]
        _state["retriever"] = HybridRetriever(records, embedder)
        _state["version"] = version
    return _state["retriever"]
