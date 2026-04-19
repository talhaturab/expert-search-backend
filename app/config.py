from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    openrouter_api_key: str
    chroma_persist_path: str = "./data/chroma"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    embedding_model: str = "openai/text-embedding-3-small"

    hyde_enabled: bool = True
    rag_top_k: int = 50
    deterministic_top_k: int = 5
    final_top_k: int = 5

    # Cap for offline indexing (None = all 10k candidates).
    # Starts at 200 for fast dev cycles; raise to None when ready for full run.
    ingest_limit: int | None = 200


def get_settings() -> Settings:
    return Settings()  # re-read env each call; fine for this scale
