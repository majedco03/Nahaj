from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from threading import Lock


_write_lock = Lock()
_project_root = Path(__file__).resolve().parent.parent


def log_path() -> Path:
    return Path(os.getenv("NAHAJ_LOG_FILE", str(_project_root / "nahaj.log")))


def log(message: object) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    path = log_path()
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{line}\n")
    except OSError as exc:
        print(f"[{timestamp}] Could not save log to {path}: {exc}", flush=True)
