from fastapi import FastAPI

from app.routes import health as health_routes
from app.routes import ingest as ingest_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Expert Search", version="0.1.0")
    app.include_router(health_routes.router)
    app.include_router(ingest_routes.router)
    return app


app = create_app()
