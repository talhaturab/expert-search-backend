from fastapi import APIRouter

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.ingest import build_index
from app.models import IngestRequest, IngestResponse
from app.routes.chat import invalidate_search_service


router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    s = get_settings()
    store = ChromaStore(persist_path=s.chroma_persist_path)
    if store.count() > 0 and not req.force:
        return IngestResponse(
            candidates_indexed=0,
            documents_written=0,
            duration_seconds=0.0,
        )
    result = build_index(
        dsn=s.database_url,
        api_key=s.openrouter_api_key,
        embedding_model=s.embedding_model,
        store=store,
        limit=s.ingest_limit,
    )
    # Chroma was rebuilt — any cached SearchService now holds stale vectors.
    invalidate_search_service()
    return IngestResponse(**result)
