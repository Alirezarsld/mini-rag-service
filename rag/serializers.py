from rest_framework import serializers


class DocumentCreateSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=200)
    text = serializers.CharField(max_length=200_000)
    source = serializers.CharField(max_length=500, required=False, allow_blank=True, default="")


class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(max_length=1000)
    top_k = serializers.IntegerField(min_value=1, max_value=20, default=5)


class AskSerializer(QuerySerializer):
    # Optional: send the id from POST /api/conversations/ to keep the exchange in a session.
    conversation_id = serializers.UUIDField(required=False)


def document_to_dict(doc) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "source": doc.source,
        "chunks": getattr(doc, "chunk_count", None),
        "created_at": doc.created_at,
    }
