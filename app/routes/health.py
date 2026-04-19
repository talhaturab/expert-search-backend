from fastapi import APIRouter

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.models import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    checks: dict[str, bool] = {
        "openrouter_api_key_set": bool(s.openrouter_api_key),
        "database_url_set":       bool(s.database_url),
        "chroma_populated":       False,
    }
    try:
        store = ChromaStore(persist_path=s.chroma_persist_path)
        checks["chroma_populated"] = store.count() > 0
    except Exception:
        pass
    status = "ok" if all(checks.values()) else "degraded"
    return HealthResponse(status=status, checks=checks)
