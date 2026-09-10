from __future__ import annotations

import re

from backend.logger import log


def test_log_prints_and_saves_utc_timestamp_and_message(capsys, monkeypatch, tmp_path):
    path = tmp_path / "nahaj.log"
    monkeypatch.setenv("NAHAJ_LOG_FILE", str(path))
    log("hello")

    output = capsys.readouterr().out.strip()
    assert re.fullmatch(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00\] hello", output)
    assert path.read_text(encoding="utf-8").strip() == output
