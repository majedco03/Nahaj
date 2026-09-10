from __future__ import annotations

import re
from pathlib import Path

from backend.logger import log, log_path


def test_default_log_file_is_in_project_root(monkeypatch):
    monkeypatch.delenv("NAHAJ_LOG_FILE", raising=False)
    assert log_path() == Path(__file__).resolve().parents[2] / "nahaj.log"


def test_log_prints_and_saves_utc_timestamp_and_message(capsys, monkeypatch, tmp_path):
    path = tmp_path / "nahaj.log"
    monkeypatch.setenv("NAHAJ_LOG_FILE", str(path))
    log("hello")

    output = capsys.readouterr().out.strip()
    assert re.fullmatch(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00\] hello", output)
    assert path.read_text(encoding="utf-8").strip() == output
