"""Typed application settings.

Values come from (highest priority first): process environment variables, then the
project's `.env` file. Secrets are held as `SecretStr`, so printing or logging the
settings object never reveals them.

Set `OMNIFLOW_ENV_FILE` to load a different env file (tests point it at a file that
does not exist, so a developer's real `.env` never leaks into the test run).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Values people commonly leave in place when copying `.env.example`.
_PLACEHOLDER_KEYS = {"", "your-api-key", "your_api_key", "changeme", "xxx", "..."}


def _env_file() -> Path:
    return Path(os.environ.get("OMNIFLOW_ENV_FILE", PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `KEY=` in .env means "use the default"
        extra="ignore",
        populate_by_name=True,
    )

    # --- Gemini ---
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = Field(default="gemini-3.8-flash", validation_alias="GEMINI_MODEL")
    gemini_fallback_model: str | None = Field(
        default="gemini-3.5-flash-lite", validation_alias="GEMINI_FALLBACK_MODEL"
    )
    gemini_thinking_level: Literal["low", "medium", "high"] | None = Field(
        default=None, validation_alias="GEMINI_THINKING_LEVEL"
    )
    gemini_temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, validation_alias="GEMINI_TEMPERATURE"
    )
    gemini_max_output_tokens: int = Field(
        default=8192, ge=256, le=65536, validation_alias="GEMINI_MAX_OUTPUT_TOKENS"
    )
    gemini_timeout_seconds: float = Field(
        default=60.0, gt=0, le=600, validation_alias="GEMINI_TIMEOUT_SECONDS"
    )
    gemini_max_retries: int = Field(default=2, ge=0, le=5, validation_alias="GEMINI_MAX_RETRIES")

    # --- App limits ---
    max_history_messages: int = Field(
        default=40, ge=2, le=500, validation_alias="OMNIFLOW_MAX_HISTORY_MESSAGES"
    )
    max_input_chars: int = Field(
        default=8000, ge=100, le=200_000, validation_alias="OMNIFLOW_MAX_INPUT_CHARS"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", validation_alias="OMNIFLOW_LOG_LEVEL"
    )

    # --- Agent limits ---
    max_tool_calls: int = Field(
        default=3, ge=1, le=10, validation_alias="OMNIFLOW_MAX_TOOL_CALLS"
    )
    max_agent_steps: int = Field(
        default=8, ge=2, le=50, validation_alias="OMNIFLOW_MAX_AGENT_STEPS"
    )

    # --- Web search (DuckDuckGo via the keyless `ddgs` library) ---
    web_search_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_WEB_SEARCH_ENABLED")
    web_search_timeout_seconds: float = Field(
        default=10.0, gt=0, le=60, validation_alias="OMNIFLOW_WEB_SEARCH_TIMEOUT_SECONDS"
    )
    web_search_max_results: int = Field(
        default=5, ge=1, le=10, validation_alias="OMNIFLOW_WEB_SEARCH_MAX_RESULTS"
    )

    # --- Tavily research search (official API; the user's own free-plan key) ---
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    tavily_timeout_seconds: float = Field(
        default=15.0, gt=0, le=60, validation_alias="OMNIFLOW_TAVILY_TIMEOUT_SECONDS"
    )
    tavily_max_results: int = Field(
        default=5, ge=1, le=10, validation_alias="OMNIFLOW_TAVILY_MAX_RESULTS"
    )

    # --- Weather (Open-Meteo free API; keyless for non-commercial use) ---
    weather_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_WEATHER_ENABLED")

    # --- Currency (Frankfurter public API; keyless, reference rates) ---
    currency_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_CURRENCY_ENABLED")

    # --- News and academic research (keyless providers) ---
    news_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_NEWS_ENABLED")
    academic_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_ACADEMIC_ENABLED")
    news_max_results: int = Field(
        default=8, ge=1, le=15, validation_alias="OMNIFLOW_NEWS_MAX_RESULTS"
    )
    academic_max_results: int = Field(
        default=8, ge=1, le=15, validation_alias="OMNIFLOW_ACADEMIC_MAX_RESULTS"
    )
    research_timeout_seconds: float = Field(
        default=12, gt=0, le=60, validation_alias="OMNIFLOW_RESEARCH_TIMEOUT_SECONDS"
    )

    # --- Persistent conversations (local SQLite, gitignored data/ folder) ---
    history_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_HISTORY_ENABLED")
    conversations_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "omniflow_conversations.db",
        validation_alias="OMNIFLOW_CONVERSATIONS_DB",
    )
    checkpoints_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "omniflow_checkpoints.db",
        validation_alias="OMNIFLOW_CHECKPOINTS_DB",
    )
    history_list_limit: int = Field(
        default=50, ge=5, le=500, validation_alias="OMNIFLOW_HISTORY_LIST_LIMIT"
    )

    # --- Long-term memory (local SQLite, separate from chat history) ---
    memory_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_MEMORY_ENABLED")
    memory_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "omniflow_memory.db",
        validation_alias="OMNIFLOW_MEMORY_DB",
    )
    memory_max_items: int = Field(  # memories added to one prompt
        default=8, ge=1, le=30, validation_alias="OMNIFLOW_MEMORY_MAX_ITEMS"
    )
    memory_max_chars: int = Field(  # total memory text added to one prompt
        default=1500, ge=200, le=8000, validation_alias="OMNIFLOW_MEMORY_MAX_CHARS"
    )

    # --- Image generation (credentials are created by the user; never in code). The provider
    # is chosen only here; no other setting switches it to the mock. ---
    image_generation_provider: Literal["disabled", "mock", "cloudflare"] = Field(
        default="disabled", validation_alias="OMNIFLOW_IMAGE_PROVIDER")
    cloudflare_account_id: str | None = Field(default=None,
                                              validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: SecretStr | None = Field(default=None,
                                                   validation_alias="CLOUDFLARE_API_TOKEN")
    generated_images_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "generated_images",
        validation_alias="OMNIFLOW_GENERATED_IMAGES_DIR")
    max_tool_rounds: int = Field(  # chained tool rounds per question (orchestration)
        default=3, ge=1, le=5, validation_alias="OMNIFLOW_MAX_TOOL_ROUNDS")

    # --- Planner (projects and tasks; local SQLite, separate from chat and memory) ---
    planner_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_PLANNER_ENABLED")
    planner_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "omniflow_planner.db",
        validation_alias="OMNIFLOW_PLANNER_DB",
    )

    # --- Coding assistant (repositories are untrusted input) ---
    code_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_CODE_ENABLED")
    code_workspace_dir: Path = Field(  # extracted uploads and patch backups (gitignored)
        default=PROJECT_ROOT / "data" / "code_workspaces",
        validation_alias="OMNIFLOW_CODE_WORKSPACE_DIR",
    )
    code_max_file_kb: int = Field(default=256, ge=8, le=2048,
                                  validation_alias="OMNIFLOW_CODE_MAX_FILE_KB")
    code_max_files: int = Field(default=1500, ge=10, le=20_000,
                                validation_alias="OMNIFLOW_CODE_MAX_FILES")
    code_max_total_mb: float = Field(  # total text indexed per project
        default=12, gt=0, le=200, validation_alias="OMNIFLOW_CODE_MAX_TOTAL_MB")
    code_zip_max_mb: float = Field(default=20, gt=0, le=200,
                                   validation_alias="OMNIFLOW_CODE_ZIP_MAX_MB")
    code_zip_max_uncompressed_mb: float = Field(
        default=150, gt=0, le=2000, validation_alias="OMNIFLOW_CODE_ZIP_MAX_UNCOMPRESSED_MB")
    code_zip_max_members: int = Field(default=10_000, ge=10, le=100_000,
                                      validation_alias="OMNIFLOW_CODE_ZIP_MAX_MEMBERS")
    code_read_max_lines: int = Field(default=200, ge=20, le=1000,
                                     validation_alias="OMNIFLOW_CODE_READ_MAX_LINES")
    code_workspace_ttl_minutes: int = Field(default=180, ge=5, le=24 * 60,
                                            validation_alias="OMNIFLOW_CODE_WORKSPACE_TTL_MINUTES")
    code_check_timeout_seconds: float = Field(default=120, gt=0, le=900,
                                              validation_alias="OMNIFLOW_CODE_CHECK_TIMEOUT_SECONDS")
    code_check_max_output_kb: int = Field(default=64, ge=4, le=1024,
                                          validation_alias="OMNIFLOW_CODE_CHECK_MAX_OUTPUT_KB")
    # None = automatic: allowed on a private local install, off on a shared deployment.
    code_local_folders_enabled: bool | None = Field(
        default=None, validation_alias="OMNIFLOW_CODE_LOCAL_FOLDERS_ENABLED")
    code_execution_enabled: bool | None = Field(
        default=None, validation_alias="OMNIFLOW_CODE_EXECUTION_ENABLED")
    code_patch_apply_enabled: bool | None = Field(
        default=None, validation_alias="OMNIFLOW_CODE_PATCH_APPLY_ENABLED")
    # Local folders whose tests may run (executing tests runs that project's code).
    # This application's own folder is always trusted; separate entries with ";".
    code_trusted_roots: str = Field(default="", validation_alias="OMNIFLOW_CODE_TRUSTED_ROOTS")

    # --- Who owns saved data ---
    # single_user: one owner (a private local install). per_session: each browser session is
    # its own owner, so visitors of a shared deployment never see each other's data.
    # auto: per_session on Streamlit Community Cloud, single_user everywhere else.
    access_mode: Literal["auto", "single_user", "per_session"] = Field(
        default="auto", validation_alias="OMNIFLOW_ACCESS_MODE"
    )
    session_data_ttl_hours: float = Field(  # per-session data older than this is purged
        default=24, gt=0, le=24 * 30, validation_alias="OMNIFLOW_SESSION_DATA_TTL_HOURS"
    )

    # --- Image understanding (Gemini vision via the existing key) ---
    images_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_IMAGES_ENABLED")
    image_max_file_mb: float = Field(
        default=10, gt=0, le=20, validation_alias="OMNIFLOW_IMAGE_MAX_FILE_MB"
    )
    image_max_pixels: int = Field(  # decoded width x height; guards decompression bombs
        default=40_000_000, ge=1_000_000, le=100_000_000,
        validation_alias="OMNIFLOW_IMAGE_MAX_PIXELS",
    )
    image_max_side: int = Field(  # longer side after downscaling, before sending to Gemini
        default=1536, ge=512, le=3072, validation_alias="OMNIFLOW_IMAGE_MAX_SIDE"
    )
    image_max_per_conversation: int = Field(
        default=6, ge=1, le=20, validation_alias="OMNIFLOW_IMAGE_MAX_PER_CONVERSATION"
    )
    image_max_per_call: int = Field(
        default=3, ge=1, le=6, validation_alias="OMNIFLOW_IMAGE_MAX_PER_CALL"
    )

    # --- Documents (PDF RAG): local embeddings only ---
    embedding_model: Literal[
        "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
    ] = Field(default="BAAI/bge-small-en-v1.5", validation_alias="OMNIFLOW_EMBEDDING_MODEL")
    model_cache_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "models", validation_alias="OMNIFLOW_MODEL_CACHE_DIR"
    )
    rag_max_files: int = Field(default=10, ge=1, le=50, validation_alias="OMNIFLOW_RAG_MAX_FILES")
    rag_max_file_mb: float = Field(
        default=20, gt=0, le=25, validation_alias="OMNIFLOW_RAG_MAX_FILE_MB"
    )
    rag_max_pages: int = Field(
        default=300, ge=1, le=2000, validation_alias="OMNIFLOW_RAG_MAX_PAGES"
    )
    rag_max_chars_per_file: int = Field(
        default=1_500_000, ge=1000, le=10_000_000,
        validation_alias="OMNIFLOW_RAG_MAX_CHARS_PER_FILE",
    )
    rag_max_chunks: int = Field(
        default=2000, ge=10, le=20_000, validation_alias="OMNIFLOW_RAG_MAX_CHUNKS"
    )
    rag_chunk_size: int = Field(
        default=1000, ge=200, le=4000, validation_alias="OMNIFLOW_RAG_CHUNK_SIZE"
    )
    rag_chunk_overlap: int = Field(
        default=150, ge=0, le=1000, validation_alias="OMNIFLOW_RAG_CHUNK_OVERLAP"
    )
    rag_top_k: int = Field(default=5, ge=1, le=10, validation_alias="OMNIFLOW_RAG_TOP_K")
    # Calibrated for bge-small-en-v1.5 on 2026-10-10: relevant passages scored 0.71-0.85,
    # unrelated queries 0.38-0.47 (bge scores cluster high; see its model card).
    rag_min_score: float = Field(
        default=0.55, ge=-1.0, le=1.0, validation_alias="OMNIFLOW_RAG_MIN_SCORE"
    )
    rag_extract_timeout_seconds: float = Field(
        default=120, gt=0, le=1800, validation_alias="OMNIFLOW_RAG_EXTRACT_TIMEOUT_SECONDS"
    )
    rag_session_ttl_minutes: int = Field(
        default=120, ge=5, le=1440, validation_alias="OMNIFLOW_RAG_SESSION_TTL_MINUTES"
    )
    rag_max_sessions: int = Field(
        default=20, ge=1, le=500, validation_alias="OMNIFLOW_RAG_MAX_SESSIONS"
    )

    # --- Optional local OCR (Tesseract) for image-only pages ---
    ocr_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_OCR_ENABLED")
    tesseract_cmd: str | None = Field(default=None, validation_alias="OMNIFLOW_TESSERACT_CMD")
    ocr_max_pages: int = Field(default=20, ge=0, le=500, validation_alias="OMNIFLOW_OCR_MAX_PAGES")
    ocr_page_timeout_seconds: float = Field(
        default=30, gt=0, le=300, validation_alias="OMNIFLOW_OCR_PAGE_TIMEOUT_SECONDS"
    )

    # --- Data analysis (CSV / XLSX), all computed locally with pandas ---
    data_max_files: int = Field(default=5, ge=1, le=20, validation_alias="OMNIFLOW_DATA_MAX_FILES")
    data_max_file_mb: float = Field(
        default=20, gt=0, le=25, validation_alias="OMNIFLOW_DATA_MAX_FILE_MB"
    )
    data_max_rows: int = Field(
        default=200_000, ge=10, le=2_000_000, validation_alias="OMNIFLOW_DATA_MAX_ROWS"
    )
    data_max_columns: int = Field(
        default=200, ge=1, le=2000, validation_alias="OMNIFLOW_DATA_MAX_COLUMNS"
    )
    data_max_memory_mb: float = Field(
        default=300, gt=0, le=4000, validation_alias="OMNIFLOW_DATA_MAX_MEMORY_MB"
    )
    data_max_excel_sheets: int = Field(
        default=5, ge=1, le=50, validation_alias="OMNIFLOW_DATA_MAX_EXCEL_SHEETS"
    )
    data_result_rows: int = Field(
        default=50, ge=5, le=500, validation_alias="OMNIFLOW_DATA_RESULT_ROWS"
    )
    data_chart_points: int = Field(
        default=500, ge=10, le=5000, validation_alias="OMNIFLOW_DATA_CHART_POINTS"
    )
    data_timeout_seconds: float = Field(
        default=30, gt=0, le=300, validation_alias="OMNIFLOW_DATA_TIMEOUT_SECONDS"
    )
    data_forecast_max_horizon: int = Field(
        default=24, ge=1, le=120, validation_alias="OMNIFLOW_DATA_FORECAST_MAX_HORIZON"
    )
    data_max_series_periods: int = Field(
        default=5000, ge=50, le=50_000, validation_alias="OMNIFLOW_DATA_MAX_SERIES_PERIODS"
    )

    # --- SQLite SQL agent (read-only) ---
    sql_sample_enabled: bool = Field(default=True, validation_alias="OMNIFLOW_SQL_SAMPLE_ENABLED")
    sql_sample_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sample" / "omniflow_sample.db",
        validation_alias="OMNIFLOW_SQL_SAMPLE_PATH",
    )
    # Your own database: used only when you set BOTH the path and the table allowlist.
    sql_database_path: Path | None = Field(
        default=None, validation_alias="OMNIFLOW_SQL_DATABASE_PATH"
    )
    sql_database_name: str = Field(
        default="my_database", validation_alias="OMNIFLOW_SQL_DATABASE_NAME"
    )
    sql_allowed_tables: str = Field(default="", validation_alias="OMNIFLOW_SQL_ALLOWED_TABLES")
    sql_blocked_columns: str = Field(default="", validation_alias="OMNIFLOW_SQL_BLOCKED_COLUMNS")
    sql_max_rows: int = Field(default=200, ge=1, le=5000, validation_alias="OMNIFLOW_SQL_MAX_ROWS")
    sql_timeout_seconds: float = Field(
        default=5, gt=0, le=60, validation_alias="OMNIFLOW_SQL_TIMEOUT_SECONDS"
    )
    sql_max_query_chars: int = Field(
        default=4000, ge=100, le=20_000, validation_alias="OMNIFLOW_SQL_MAX_QUERY_CHARS"
    )
    # Per-session "Import into SQL" (row/column limits reuse OMNIFLOW_DATA_MAX_ROWS/COLUMNS).
    sql_import_max_tables: int = Field(
        default=5, ge=1, le=20, validation_alias="OMNIFLOW_SQL_IMPORT_MAX_TABLES"
    )

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> Settings:
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("OMNIFLOW_RAG_CHUNK_OVERLAP must be smaller than the chunk size")
        return self

    def _local_feature(self, value: bool | None) -> bool:
        if value is not None:
            return value
        return self.effective_access_mode == "single_user" and not self.hosted_on_community_cloud

    @property
    def code_local_folders_allowed(self) -> bool:
        return self._local_feature(self.code_local_folders_enabled)

    @property
    def code_execution_allowed(self) -> bool:
        return self._local_feature(self.code_execution_enabled)

    @property
    def code_patch_apply_allowed(self) -> bool:
        return self._local_feature(self.code_patch_apply_enabled)

    @property
    def code_trusted_root_paths(self) -> list[Path]:
        roots = [PROJECT_ROOT]
        for part in self.code_trusted_roots.split(";"):
            if part.strip():
                roots.append(Path(part.strip()).expanduser())
        return [r.resolve() for r in roots]

    @field_validator("conversations_db_path", "checkpoints_db_path", "memory_db_path",
                     "sql_sample_path", "code_workspace_dir", "planner_db_path",
                     "generated_images_dir",
                     mode="after")
    @classmethod
    def _relative_to_project(cls, value: Path) -> Path:
        """A relative path in .env means 'inside the project', whatever the working dir."""
        value = value.expanduser()
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def hosted_on_community_cloud(self) -> bool:
        """Streamlit Community Cloud runs apps from a checkout under /mount/src."""
        return PROJECT_ROOT.as_posix().startswith("/mount/src/")

    @property
    def effective_access_mode(self) -> str:
        if self.access_mode != "auto":
            return self.access_mode
        return "per_session" if self.hosted_on_community_cloud else "single_user"

    @field_validator("gemini_api_key", "tavily_api_key", mode="after")
    @classmethod
    def _drop_placeholder_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        cleaned = value.get_secret_value().strip()
        if cleaned.lower() in _PLACEHOLDER_KEYS:
            return None
        return SecretStr(cleaned)

    @field_validator("gemini_fallback_model", mode="after")
    @classmethod
    def _blank_fallback_disables(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()

    @property
    def gemini_configured(self) -> bool:
        """True when a non-placeholder key is present. This does NOT mean it is valid."""
        return self.gemini_api_key is not None


def load_settings() -> Settings:
    """Build settings fresh from the environment and env file."""
    return Settings(_env_file=_env_file())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Restart the app after editing `.env`."""
    return load_settings()
