"""Text normalisation, tokenisation and embedding backends.

Two embedders ship with the project:

* ``hashing`` (default): feature hashing of word unigrams/bigrams. No download and no
  API key, so everything runs offline. It captures *lexical* similarity only.
* ``sentence-transformers`` (optional): a real multilingual sentence-embedding model that
  matches paraphrases and synonyms. Install ``sentence-transformers`` and set
  ``RAG_EMBEDDER=sentence-transformers`` (or ``st:<model name>``).

An embedder is any object with ``name``, ``min_score`` and ``embed(texts) -> ndarray``
(L2-normalised rows). Register new ones in :func:`get_embedder`.
"""
import hashlib
import re
import unicodedata

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Arabic letter forms that people type instead of the Persian ones.
_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه",
    "ٱ": "ا", "إ": "ا", "أ": "ا",
    "‌": " ",      # zero-width non-joiner: "می‌کند" and "می کند" must match
    "‍": "",       # zero-width joiner
    "ـ": "",            # tatweel
})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS = re.compile("[ً-ٰٟ]")


def normalize(text: str) -> str:
    """Case-fold and unify look-alike characters so equivalent spellings match."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.translate(_CHAR_MAP).translate(_DIGITS)
    return _DIACRITICS.sub("", text)


# Very common function words carry no topic information; dropping them stops "what does a"
# from matching every document. Negations ("no", "not", "never") are deliberately kept.
STOPWORDS = frozenset("""
a an the and or but if then than so as of in on at to for from by with without about into over
is are was were be been being am do does did done have has had having can could should would
will shall may might must i me my we our you your he she it its they them their this that these
those there here what which who whom whose when where why how also just very more most some any
each other such only own same too
و در به از که این را با است برای آن یک هم تا بر یا هر می شود شد کرد باید چه چرا چگونه آیا کند
دارد نیز ها های تر ترین بود هست نیست بی اگر پس
""".split())


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Normalised unicode word tokens (works for Latin and Persian text)."""
    tokens = _TOKEN.findall(normalize(text))
    return tokens if keep_stopwords else [t for t in tokens if t not in STOPWORDS]


class HashingEmbedder:
    name = "hashing"
    min_score = 0.12   # cosine below this is treated as "no vector match"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        features = [(t, 1.0) for t in tokens]
        features += [(f"{a} {b}", 0.5) for a, b in zip(tokens, tokens[1:])]
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * weight
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


DEFAULT_ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder:
    """Multilingual semantic embeddings via the optional ``sentence-transformers`` package."""

    min_score = 0.30

    def __init__(self, model_name: str = DEFAULT_ST_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the optional package
            raise RuntimeError(
                "RAG_EMBEDDER=sentence-transformers needs: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


_CACHE: dict[str, object] = {}


def get_embedder(name: str = "hashing"):
    """Return a (cached) embedder. Loading a transformer model is expensive, so reuse it."""
    if name not in _CACHE:
        if name == "hashing":
            _CACHE[name] = HashingEmbedder()
        elif name == "sentence-transformers":
            _CACHE[name] = SentenceTransformerEmbedder()
        elif name.startswith("st:"):
            _CACHE[name] = SentenceTransformerEmbedder(name[3:])
        else:
            raise ValueError(f"Unknown embedder: {name!r}")
    return _CACHE[name]
