"""Settings. Every value that varies between environments comes from env vars."""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


DEBUG = env_bool("DJANGO_DEBUG", False)
TESTING = "test" in sys.argv
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not (DEBUG or TESTING):
        # Fail loudly instead of running production with a guessable key.
        # Local development: export DJANGO_DEBUG=1 (a throw-away key is used).
        from django.core.exceptions import ImproperlyConfigured
        raise ImproperlyConfigured("Set DJANGO_SECRET_KEY (or DJANGO_DEBUG=1 for local use).")
    SECRET_KEY = "insecure-development-key-do-not-use-in-production"

ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "rest_framework",
    "rag",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("RAG_DB_PATH", str(BASE_DIR / "db.sqlite3")),
    }
}

CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "UTC"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],  # access is controlled by the optional API key
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.AnonRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"anon": os.environ.get("RAG_THROTTLE", "120/min")},
    "UNAUTHENTICATED_USER": None,
}

# --- RAG configuration ------------------------------------------------------
RAG_API_KEY = os.environ.get("RAG_API_KEY", "")            # empty = open API (demo mode)
RAG_EMBEDDER = os.environ.get("RAG_EMBEDDER", "hashing")
RAG_CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", "800"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))
RAG_ANSWER_CACHE_SECONDS = int(os.environ.get("RAG_ANSWER_CACHE_SECONDS", "3600"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # empty = extractive answers only
RAG_LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-sonnet-5-5")
