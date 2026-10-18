"""Turn a follow-up question ("and how often?") into one that stands on its own.

Retrieval only sees the query text, so a pronoun-only follow-up finds nothing. With an LLM
client the question is rewritten from the recent history; without one (or if the call fails)
a cheap heuristic prepends the previous user question.
"""
import logging

from .embeddings import tokenize

logger = logging.getLogger(__name__)

HISTORY_MESSAGES = 6      # recent messages considered
MIN_CONTENT_TOKENS = 3    # fewer meaningful words than this = probably a follow-up
MAX_REWRITE_CHARS = 300

REWRITE_PROMPT = (
    "Rewrite the user's last question so it can be understood without the conversation. "
    "Keep its language. Answer with the rewritten question only; if it already stands alone, "
    "repeat it unchanged."
)


def _heuristic(query: str, history: list[tuple[str, str]]) -> str:
    if len(tokenize(query)) >= MIN_CONTENT_TOKENS:
        return query
    previous = next((text for role, text in reversed(history) if role == "user"), "")
    return f"{previous} {query}".strip() if previous else query


def _transcript(history: list[tuple[str, str]], query: str) -> str:
    lines = [f"{role}: {text[:300]}" for role, text in history[-HISTORY_MESSAGES:]]
    return "\n".join(lines + [f"user: {query}"])


def rewrite_query(query: str, history: list[tuple[str, str]], client=None, model: str = "") -> str:
    """``history`` is a list of ``(role, text)`` pairs, oldest first, not including ``query``."""
    if not history:
        return query
    if client is None:
        return _heuristic(query, history)
    try:
        response = client.messages.create(
            model=model, max_tokens=100, system=REWRITE_PROMPT,
            messages=[{"role": "user", "content": _transcript(history, query)}],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        lines = text.strip().splitlines()
        rewritten = lines[0].strip()[:MAX_REWRITE_CHARS] if lines else ""
        return rewritten or _heuristic(query, history)
    except Exception as exc:  # the rewrite is an optimisation; never fail the request over it
        logger.warning("Query rewrite failed (%s); using the heuristic", type(exc).__name__)
        return _heuristic(query, history)
