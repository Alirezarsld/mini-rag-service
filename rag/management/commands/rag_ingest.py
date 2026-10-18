from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rag import services


class Command(BaseCommand):
    help = "Index one or more UTF-8 text/markdown files (title = file name)."

    def add_arguments(self, parser):
        parser.add_argument("paths", nargs="+", type=Path)

    def handle(self, *args, paths, **options):
        for path in paths:
            if not path.is_file():
                raise CommandError(f"Not a file: {path}")
            text = path.read_text(encoding="utf-8")
            document = services.ingest_document(title=path.stem, text=text, source=str(path))
            self.stdout.write(f"Indexed {path.name} -> document {document.id}")
