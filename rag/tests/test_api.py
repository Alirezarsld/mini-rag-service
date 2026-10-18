"""API tests (require Django; run with ``python manage.py test``)."""
from unittest import mock

from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import override_settings
from rest_framework.test import APITestCase

import json
import tempfile
from io import StringIO

from rag import selectors, services
from rag.generation import NO_ANSWER
from rag.models import Chunk, Conversation
from rag.tests.test_core import _FakeLLM

CELERY = "Celery is a distributed task queue. Workers consume jobs from a Redis broker."
PG = "PostgreSQL is a relational database. VACUUM reclaims storage from dead tuples."


class ApiTests(APITestCase):
    def setUp(self):
        cache.clear()
        selectors.reset_cache()

    def add(self, title, text):
        return self.client.post("/api/documents/", {"title": title, "text": text}, format="json")

    def test_home_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response["Content-Type"])
        self.assertIn(b"/api/ask/", response.content)

    def test_conversation_keeps_messages_in_order_and_cascades(self):
        conversation = Conversation.objects.create()
        conversation.messages.create(role="user", content="first")
        conversation.messages.create(role="assistant", content="second")
        self.assertEqual([m.content for m in conversation.messages.all()], ["first", "second"])
        conversation.delete()
        self.assertEqual(Conversation.objects.count(), 0)

    def test_ask_inside_a_conversation_records_the_exchange(self):
        self.add("Celery", CELERY)
        conversation_id = self.client.post("/api/conversations/").json()["id"]
        body = {"query": "redis broker workers", "conversation_id": conversation_id}
        response = self.client.post("/api/ask/", body, format="json")
        self.assertEqual(response.json()["conversation_id"], conversation_id)
        roles = list(Conversation.objects.get(pk=conversation_id).messages.values_list("role", flat=True))
        self.assertEqual(roles, ["user", "assistant"])

    def test_ask_without_a_conversation_stores_nothing(self):
        self.add("Celery", CELERY)
        response = self.client.post("/api/ask/", {"query": "redis broker workers"}, format="json")
        self.assertNotIn("conversation_id", response.json())
        self.assertEqual(Conversation.objects.count(), 0)

    def test_unknown_conversation_is_404(self):
        body = {"query": "anything", "conversation_id": "7c1f3a52-0000-4000-8000-000000000000"}
        self.assertEqual(self.client.post("/api/ask/", body, format="json").status_code, 404)
        self.assertEqual(self.client.post("/api/ask/", {"query": "x", "conversation_id": "nope"},
                                          format="json").status_code, 400)

    def test_health(self):
        self.assertEqual(self.client.get("/api/health/").json(), {"status": "ok"})

    def test_ingest_list_get_delete(self):
        created = self.add("Celery", CELERY)
        self.assertEqual(created.status_code, 201)
        doc_id = created.json()["id"]
        self.assertGreaterEqual(created.json()["chunks"], 1)

        self.assertEqual(len(self.client.get("/api/documents/").json()), 1)
        self.assertEqual(self.client.get(f"/api/documents/{doc_id}/").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/documents/{doc_id}/").status_code, 204)
        self.assertEqual(self.client.get(f"/api/documents/{doc_id}/").status_code, 404)
        self.assertEqual(self.client.get("/api/documents/").json(), [])

    def test_validation_errors(self):
        self.assertEqual(self.add("", CELERY).status_code, 400)
        self.assertEqual(self.add("Empty", "   \n  ").status_code, 400)
        self.assertEqual(self.client.post("/api/search/", {"query": "x", "top_k": 99},
                                          format="json").status_code, 400)

    def test_search_returns_relevant_document_first(self):
        self.add("Celery", CELERY)
        self.add("PostgreSQL", PG)
        response = self.client.post("/api/search/", {"query": "redis broker workers"}, format="json")
        results = response.json()["results"]
        self.assertEqual(results[0]["document_title"], "Celery")

    def test_deleted_document_is_no_longer_searchable(self):
        doc_id = self.add("Celery", CELERY).json()["id"]
        self.client.delete(f"/api/documents/{doc_id}/")
        response = self.client.post("/api/search/", {"query": "redis broker"}, format="json")
        self.assertEqual(response.json()["results"], [])

    def test_ask_extractive_mode_cites_sources(self):
        self.add("Celery", CELERY)
        body = self.client.post("/api/ask/", {"query": "where do workers get jobs"},
                                format="json").json()
        self.assertFalse(body["used_llm"])
        self.assertIn("[1]", body["answer"])
        self.assertEqual(body["citations"][0]["document_title"], "Celery")

    def test_ask_unknown_topic_says_idk(self):
        self.add("Celery", CELERY)
        body = self.client.post("/api/ask/", {"query": "quantum chromodynamics"},
                                format="json").json()
        self.assertEqual(body["answer"], NO_ANSWER)
        self.assertEqual(body["citations"], [])

    def test_ask_uses_llm_when_configured_and_caches(self):
        self.add("Celery", CELERY)
        fake = _FakeLLM("Workers read jobs from the Redis broker [1].")
        with mock.patch("rag.services._llm_client", return_value=fake):
            first = self.client.post("/api/ask/", {"query": "who reads jobs"}, format="json").json()
            second = self.client.post("/api/ask/", {"query": "who reads jobs"}, format="json").json()
        self.assertTrue(first["used_llm"])
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])

    def test_provider_errors_are_reported_and_not_cached(self):
        self.add("Celery", CELERY)
        broken = _FakeLLM(error=RuntimeError("overloaded"))
        with mock.patch("rag.services._llm_client", return_value=broken):
            first = self.client.post("/api/ask/", {"query": "who reads jobs"}, format="json").json()
            second = self.client.post("/api/ask/", {"query": "who reads jobs"}, format="json").json()
        self.assertEqual(first["llm_error"], "RuntimeError")
        self.assertFalse(first["used_llm"])
        self.assertNotEqual(first["answer"], NO_ANSWER)   # extractive fallback still answers
        self.assertFalse(second["cached"])                # a transient failure is never cached

    def test_ask_reports_usage(self):
        self.add("Celery", CELERY)
        with mock.patch("rag.services._llm_client", return_value=_FakeLLM("Redis [1].")):
            body = self.client.post("/api/ask/", {"query": "who reads jobs"}, format="json").json()
        self.assertEqual(body["usage"]["output_tokens"], 15)
        self.assertFalse(body["truncated"])

    def test_cache_is_invalidated_when_corpus_changes(self):
        self.add("Celery", CELERY)
        q = {"query": "vacuum dead tuples"}
        before = self.client.post("/api/ask/", q, format="json").json()
        self.assertEqual(before["answer"], NO_ANSWER)
        self.add("PostgreSQL", PG)
        after = self.client.post("/api/ask/", q, format="json").json()
        self.assertNotEqual(after["answer"], NO_ANSWER)

    def upload(self, name, data, **extra):
        return self.client.post("/api/documents/upload/",
                                {"file": SimpleUploadedFile(name, data), **extra}, format="multipart")

    def test_upload_indexes_a_text_file(self):
        response = self.upload("celery-notes.md", CELERY.encode())
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["title"], "celery-notes")
        self.assertEqual(response.json()["source"], "celery-notes.md")
        found = self.client.post("/api/search/", {"query": "redis broker"}, format="json").json()
        self.assertEqual(found["results"][0]["document_title"], "celery-notes")

    def test_upload_rejects_bad_input(self):
        self.assertEqual(self.client.post("/api/documents/upload/", {}, format="multipart").status_code, 400)
        self.assertEqual(self.upload("x.txt", b"\xff\xfe\x00bad").status_code, 400)
        self.assertEqual(self.upload("empty.txt", b"  \n").status_code, 400)
        self.assertEqual(self.upload("big.txt", b"a " * 600_000).status_code, 413)

    def test_upload_detects_duplicates(self):
        self.upload("a.md", CELERY.encode())
        self.assertEqual(self.upload("b.md", CELERY.encode()).status_code, 409)

    def test_duplicate_text_is_rejected_with_409(self):
        first = self.add("Celery", CELERY).json()
        again = self.add("Another title", "  " + CELERY.upper() + "  ")
        self.assertEqual(again.status_code, 409)
        self.assertEqual(again.json()["document_id"], first["id"])
        self.assertEqual(len(self.client.get("/api/documents/").json()), 1)

    def test_chunks_from_another_embedder_are_ignored_until_reindexed(self):
        self.add("Celery", CELERY)
        Chunk.objects.update(embedder="some-other-model")
        selectors.reset_cache()
        self.assertEqual(selectors.stale_chunk_count(), 1)
        body = {"query": "redis broker workers"}
        self.assertEqual(self.client.post("/api/search/", body, format="json").json()["results"], [])

        self.assertEqual(services.reindex(), 1)
        self.assertEqual(selectors.stale_chunk_count(), 0)
        results = self.client.post("/api/search/", body, format="json").json()["results"]
        self.assertEqual(results[0]["document_title"], "Celery")

    def test_reindex_and_eval_commands(self):
        self.add("Celery", CELERY)
        self.add("PostgreSQL", PG)
        out = StringIO()
        call_command("rag_reindex", stdout=out)
        self.assertIn("re-embedded: 0", out.getvalue())

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"question": "redis broker workers", "document": "Celery"},
                       {"question": "vacuum dead tuples", "document": "PostgreSQL"}], f)
        out = StringIO()
        call_command("rag_eval", f.name, "--json", stdout=out)
        report = json.loads(out.getvalue())
        self.assertEqual(report["hit_rate"], 1.0)

    @override_settings(RAG_API_KEY="s3cret")
    def test_api_key_is_enforced_when_configured(self):
        self.assertEqual(self.client.get("/api/documents/").status_code, 403)
        self.assertEqual(self.client.get("/api/documents/", HTTP_X_API_KEY="wrong").status_code, 403)
        self.assertEqual(self.client.get("/api/documents/", HTTP_X_API_KEY="s3cret").status_code, 200)
        self.assertEqual(self.client.get("/api/health/").status_code, 200)
