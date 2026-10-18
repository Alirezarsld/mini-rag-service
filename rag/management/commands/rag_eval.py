import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rag import selectors
from rag.evaluation import evaluate


class Command(BaseCommand):
    help = (
        "Replay labelled questions through retrieval and report hit-rate@k and MRR. "
        "Cases file: JSON list of {\"question\": ..., \"document\": <title>}."
    )

    def add_arguments(self, parser):
        parser.add_argument("cases", type=Path)
        parser.add_argument("--k", type=int, default=5)
        parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
        parser.add_argument("--min-hit-rate", type=float, default=0.0,
                            help="exit with an error below this hit-rate (for CI)")

    def handle(self, *args, cases, k, as_json=False, min_hit_rate=0.0, **options):
        try:
            raw = json.loads(cases.read_text(encoding="utf-8"))
            pairs = [(c["question"], c["document"]) for c in raw]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CommandError(f"Invalid cases file: {exc}") from exc

        report = evaluate(selectors.get_retriever().search, pairs, k=k)
        if as_json:
            self.stdout.write(json.dumps(report.as_dict()))
        else:
            self.stdout.write(f"cases={report.n}  hit_rate@{k}={report.hit_rate:.3f}  mrr={report.mrr:.3f}")
        if report.hit_rate < min_hit_rate:
            raise CommandError(f"hit_rate {report.hit_rate:.3f} is below {min_hit_rate}")
