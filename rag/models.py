import uuid

from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=200)
    source = models.CharField(max_length=500, blank=True)
    # SHA-256 of the normalised text; lets ingestion reject duplicates. NULL for rows
    # created before the column existed.
    content_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return self.title


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    index = models.PositiveIntegerField()
    text = models.TextField()
    embedding = models.JSONField()  # list[float]; portable across SQLite/PostgreSQL
    # Name of the embedder that produced ``embedding``. Vectors from different models are
    # not comparable, so retrieval only uses chunks that match the configured embedder.
    embedder = models.CharField(max_length=120, default="hashing", db_index=True)

    class Meta:
        ordering = ["document_id", "index"]


class Conversation(models.Model):
    """A chat session; its messages give follow-up questions their context."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)


class Message(models.Model):
    ROLES = [("user", "user"), ("assistant", "assistant")]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]
