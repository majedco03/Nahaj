from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./nahaj.db")
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    api_key: str = os.getenv("NAHAJ_API_KEY", "dev-key")
    supervisor_factory: str = os.getenv("SUPERVISOR_FACTORY", "")
    progress_check_interval_minutes: int = int(os.getenv("PROGRESS_CHECK_INTERVAL_MINUTES", "15"))
    progress_scheduler_enabled: bool = _as_bool(os.getenv("PROGRESS_SCHEDULER_ENABLED"), True)
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    timezone: str = os.getenv("NAHAJ_TIMEZONE", "Asia/Riyadh")


settings = Settings()
