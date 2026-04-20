from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes import chat as chat_routes
from app.routes import experts as experts_routes
from app.routes import health as health_routes
from app.routes import ingest as ingest_routes


STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Expert Search", version="0.1.0")
    app.include_router(health_routes.router)
    app.include_router(ingest_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(experts_routes.router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def _index():  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
