import hmac

from django.conf import settings
from rest_framework.permissions import BasePermission


class HasApiKey(BasePermission):
    """Require ``X-API-Key`` when RAG_API_KEY is configured; open otherwise (demo mode)."""

    def has_permission(self, request, view) -> bool:
        expected = settings.RAG_API_KEY
        if not expected:
            return True
        provided = request.headers.get("X-API-Key", "")
        return hmac.compare_digest(provided.encode(), expected.encode())
