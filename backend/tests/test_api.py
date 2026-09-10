from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

_TEST_ROOT = Path(tempfile.gettempdir()) / f"nahaj-tests-{uuid4().hex}"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_ROOT}.db"
os.environ["UPLOAD_DIR"] = str(_TEST_ROOT / "uploads")
os.environ["NAHAJ_LOG_FILE"] = str(_TEST_ROOT / "logs" / "nahaj.log")
os.environ["NAHAJ_API_KEY"] = "test-key"
os.environ["PROGRESS_SCHEDULER_ENABLED"] = "false"
os.environ.pop("SUPERVISOR_FACTORY", None)

from fastapi.testclient import TestClient

from backend import main
from backend.database import SessionLocal
from backend.main import app
from backend.models import Quiz, QuizOption, QuizQuestion
from backend.supervisor_contract import SupervisorEvent


AUTH = {"Authorization": "Bearer test-key"}


def _client():
    return TestClient(app)


def _course(client: TestClient) -> str:
    client.put(
        "/api/v1/semester",
        headers=AUTH,
        json={"name": "Fall", "start_date": "2026-09-01", "end_date": "2026-12-31", "timezone": "Asia/Riyadh"},
    )
    code = f"CS{uuid4().hex[:6].upper()}"
    response = client.post("/api/v1/courses", headers=AUTH, json={"code": code, "name": "Algorithms", "color": "#123456"})
    assert response.status_code == 201
    return response.json()["id"]


def test_requests_are_saved_with_trace_details():
    log_file = Path(os.environ["NAHAJ_LOG_FILE"])
    log_file.unlink(missing_ok=True)

    with _client() as client:
        response = client.get("/health")

    saved = log_file.read_text(encoding="utf-8")
    assert response.headers["X-Request-ID"]
    assert "Request started id=" in saved
    assert "method=GET path=/health" in saved
    assert "Request completed id=" in saved
    assert "status=200 duration_ms=" in saved


def test_health_auth_and_course_validation():
    with _client() as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/courses").status_code == 401
        course_id = _course(client)
        assert course_id
        invalid = client.post("/api/v1/courses", headers=AUTH, json={"code": "CS102", "name": "Bad", "color": "red"})
        assert invalid.status_code == 422


def test_upload_rejects_unsafe_and_duplicate_files():
    with _client() as client:
        course_id = _course(client)
        unsafe = client.post(
            f"/api/v1/courses/{course_id}/documents",
            headers=AUTH,
            data={"collection": "slides"},
            files={"file": ("../notes.pdf", b"%PDF-1.7", "application/pdf")},
        )
        assert unsafe.status_code == 400
        uploaded = client.post(
            f"/api/v1/courses/{course_id}/documents",
            headers=AUTH,
            data={"collection": "slides"},
            files={"file": ("notes.pdf", b"%PDF-1.7", "application/pdf")},
        )
        assert uploaded.status_code == 202
        duplicate = client.post(
            f"/api/v1/courses/{course_id}/documents",
            headers=AUTH,
            data={"collection": "slides"},
            files={"file": ("copy.pdf", b"%PDF-1.7", "application/pdf")},
        )
        assert duplicate.status_code == 409


def test_plan_is_the_calendar_projection():
    with _client() as client:
        course_id = _course(client)
        start = datetime.now(timezone.utc).replace(microsecond=0)
        created = client.post(
            "/api/v1/tasks",
            headers=AUTH,
            json={"course_id": course_id, "title": "Review", "start_at": start.isoformat(), "end_at": (start + timedelta(hours=1)).isoformat()},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        plan = client.get("/api/v1/plan", headers=AUTH).json()
        assert [task["id"] for task in plan["tasks"]] == [task_id]
        assert plan["summary"]["done"] == 0
        assert client.patch(f"/api/v1/tasks/{task_id}", headers=AUTH, json={"status": "done"}).status_code == 200
        assert client.get("/api/v1/plan", headers=AUTH).json()["summary"]["done"] == 1


def test_quiz_hides_correctness_until_completion_and_grades_ids():
    with _client() as client:
        course_id = _course(client)
        with SessionLocal() as db:
            quiz = Quiz(course_id=course_id, title="Practice", source_collections=["slides"])
            db.add(quiz)
            db.flush()
            question = QuizQuestion(quiz_id=quiz.id, prompt="Two plus two?", topic="arithmetic", explanation="Basic addition.", correct_option_id="correct", position=0)
            db.add(question)
            db.flush()
            for position, option_id in enumerate(["wrong-a", "correct", "wrong-b", "wrong-c"]):
                db.add(QuizOption(question_id=question.id, option_id=option_id, text=option_id, position=position))
            db.commit()
            quiz_id, question_id = quiz.id, question.id

        public = client.get(f"/api/v1/quizzes/{quiz_id}", headers=AUTH)
        assert public.status_code == 200
        assert "correct_option_id" not in public.text
        attempt = client.post(f"/api/v1/quizzes/{quiz_id}/attempts", headers=AUTH).json()
        assert "correct_option_id" not in str(attempt)
        attempt_id = attempt["id"]
        assert client.post(f"/api/v1/attempts/{attempt_id}/answers", headers=AUTH, json={"question_id": question_id, "selected_option_id": "correct"}).status_code == 200
        completed = client.post(f"/api/v1/attempts/{attempt_id}/complete", headers=AUTH)
        assert completed.status_code == 200
        assert completed.json()["score_percent"] == 100
        assert completed.json()["questions"][0]["correct_option_id"] == "correct"


def test_notifications_dedupe_and_openai_contract():
    with _client() as client:
        with SessionLocal() as db:
            state = {"text": [], "citations": [], "created_links": []}
            event = SupervisorEvent("notification_create", {"title": "Check in", "message": "Review today", "dedupe_key": "today"})
            main._apply_event(db, event, state)
            main._apply_event(db, event, state)
        assert len(client.get("/api/v1/notifications?unread=true", headers=AUTH).json()) == 1
        bad_model = client.post("/v1/chat/completions", headers=AUTH, json={"model": "other", "messages": [{"role": "user", "content": "Hi"}]})
        assert bad_model.status_code == 404
        unavailable = client.post("/v1/chat/completions", headers=AUTH, json={"model": "nahaj-supervisor", "messages": [{"role": "user", "content": "Hi"}]})
        assert unavailable.status_code == 503
        stream = client.post("/v1/chat/completions", headers=AUTH, json={"model": "nahaj-supervisor", "stream": True, "messages": [{"role": "user", "content": "Hi"}]})
        assert stream.status_code == 200
        assert "[DONE]" in stream.text


def test_chat_validates_the_complete_supervisor_output(monkeypatch):
    class UnsafeGateway:
        async def handle(self, _request):
            yield SupervisorEvent("text_delta", {"text": "OPENROUTER_API_KEY=very-secret-value"})

    monkeypatch.setattr(main, "get_supervisor", lambda: UnsafeGateway())
    with _client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "nahaj-supervisor", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "I couldn't provide that response because it did not pass the safety check."
    )
    assert "very-secret-value" not in response.text


def test_chat_uses_forwarded_openwebui_chat_and_task_ids(monkeypatch):
    captured = []

    class CapturingGateway:
        async def handle(self, request):
            captured.append(request)
            yield SupervisorEvent("text_delta", {"text": "ok"})

    monkeypatch.setattr(main, "get_supervisor", lambda: CapturingGateway())
    headers = {
        **AUTH,
        "X-OpenWebUI-Chat-Id": "chat-123",
        "X-OpenWebUI-Task": "title_generation",
    }
    with _client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "nahaj-supervisor", "messages": [{"role": "user", "content": "Make a title"}]},
        )

    assert response.status_code == 200
    assert captured[0].context["thread_id"] == "chat-123:task:title_generation"
    assert captured[0].context["client_task"] == "title_generation"
