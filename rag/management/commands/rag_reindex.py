from django.core.management.base import BaseCommand

from rag import selectors, services


class Command(BaseCommand):
    help = "Re-embed chunks that were indexed with a different embedder than RAG_EMBEDDER."

    def handle(self, *args, **options):
        before = selectors.stale_chunk_count()
        updated = services.reindex()
        self.stdout.write(f"stale before: {before}; re-embedded: {updated}")
