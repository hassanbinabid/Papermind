"""
config.py — Centralized, typed application settings.

Replaces scattered os.getenv() calls with a single cached Settings
object, injected via FastAPI's Depends(). Values are read from the
environment (or a .env file) once, validated by pydantic, and reused
everywhere — instead of re-reading env vars ad hoc inside each route.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -- App metadata --
    app_name: str = "PaperMind RAG API"
    app_version: str = "2.1.0"

    # -- Security --
    api_key: str = "change-me"  # required X-API-Key header on write endpoints
    frontend_origin: str = "http://localhost:3000"  # tighten before going public — no "*"

    # -- Rate limiting --
    rate_limit_default: str = "60/minute"
    rate_limit_ingest: str = "5/minute"

    # -- External services (existing env vars, now typed) --
    groq_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached singleton settings instance. lru_cache means the environment
    is parsed once per process and the same Settings object is reused
    across every Depends(get_settings) call — cheap, and consistent.
    """
    return Settings()
