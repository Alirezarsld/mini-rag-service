"""Split documents into overlapping, sentence-aware chunks."""
import re

_SENTENCE_END = re.compile(r"(?<=[.!?؟۔])\s+")


def _split_words(text: str, limit: int) -> list[str]:
    """Split on word boundaries into pieces of at most ``limit`` characters."""
    pieces, current = [], ""
    for word in text.split(" "):
        while len(word) > limit:          # a single token longer than the limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:limit])
            word = word[limit:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current += " " + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, max_chars: int, overlap: int) -> list[str]:
    """Break text into units (paragraph -> sentence -> words).

    Oversized units are capped at ``max_chars - overlap`` so a repeated tail
    always fits in front of the next unit.
    """
    unit_limit = max(1, max_chars - overlap)
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            if len(sentence) <= max_chars:
                if sentence:
                    units.append(sentence)
            else:
                units.extend(_split_words(sentence, unit_limit))
    return units


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Greedily pack units into chunks of at most ``max_chars`` characters.

    The tail of each chunk (``overlap`` characters, cut at a word boundary) is
    repeated at the start of the next one so that facts spanning a boundary
    stay retrievable.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    chunks: list[str] = []
    current = ""
    for unit in _units(text or "", max_chars, overlap):
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= max_chars:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]
            candidate = f"{tail} {unit}".strip() if tail else unit
            current = candidate if len(candidate) <= max_chars else unit
    if current:
        chunks.append(current)
    return chunks
