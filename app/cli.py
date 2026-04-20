"""CLI companion — same core services as the API, easier for local testing.

    poetry run python -m app.cli chat "find me regulatory affairs experts in pharma"
    poetry run python -m app.cli ingest --force
"""
from __future__ import annotations

import argparse
import json
import sys

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.ingest import build_index
from app.routes.chat import get_search_service, invalidate_search_service


def _json_default(value):
    # Fall back for any non-serializable leaf (datetimes if they sneak in, etc.)
    return str(value)


def cmd_chat(args) -> int:
    service = get_search_service()
    resp = service.search(args.query)
    print(json.dumps(resp.model_dump(), indent=2, default=_json_default))
    return 0


def cmd_ingest(args) -> int:
    s = get_settings()
    store = ChromaStore(persist_path=s.chroma_persist_path)
    if store.count() > 0 and not args.force:
        print(
            "Index already populated; pass --force to rebuild.",
            file=sys.stderr,
        )
        return 1
    result = build_index(
        dsn=s.database_url,
        api_key=s.openrouter_api_key,
        embedding_model=s.embedding_model,
        store=store,
        limit=s.ingest_limit,
    )
    invalidate_search_service()
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="expert-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Submit a NL query and print the ChatResponse as JSON")
    p_chat.add_argument("query")
    p_chat.set_defaults(func=cmd_chat)

    p_ing = sub.add_parser("ingest", help="Build the vector index from Postgres")
    p_ing.add_argument("--force", action="store_true",
                       help="Rebuild even if the index is already populated")
    p_ing.set_defaults(func=cmd_ingest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
