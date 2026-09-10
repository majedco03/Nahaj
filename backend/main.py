from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, Query, UploadFile, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload
from starlette.requests import Request

from .app_config import settings
from .database import get_db, init_db
from .guardrails import safe_final_response
from .logger import log, log_path
from .models import (
    Course,
    Document,
    Notification,
    Quiz,
    QuizAnswer,
    QuizAttempt,
    QuizOption,
    QuizQuestion,
    Semester,
    StudyTask,
    now_utc,
)
from .rag_ingestion import CourseIngestor
from .schemas import (
    AnswerInput,
    AttemptOut,
    AttemptStartOut,
    CourseCreate,
    CourseOut,
    CourseUpdate,
    DocumentOut,
    NotificationOut,
    PlanOut,
    ProgressOut,
    QuizDetailOut,
    QuizListOut,
    QuizOptionOut,
    QuizQuestionOut,
    ReviewQuestionOut,
    SemesterInput,
    SemesterOut,
    TaskCreate,
    TaskOut,
    TaskUpdate,
)
from .supervisor_contract import (
    SupervisorEvent,
    SupervisorRequest,
    SupervisorUnavailable,
    get_supervisor,
)

ALLOWED_COLLECTIONS = {"slides", "past_exam"}
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
}


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _error(status_code: int, code: str, message: str, details: dict[str, Any] | None = None):
    raise APIError(status_code, code, message, details)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log(f"Starting Nahaj API version={app.version} log_file={log_path()}")
    init_db()
    log("Database initialized")
    worker = None
    if settings.progress_scheduler_enabled and settings.progress_check_interval_minutes > 0:
        worker = asyncio.create_task(_progress_worker())
        log(f"Progress scheduler started interval_minutes={settings.progress_check_interval_minutes}")
    try:
        yield
    finally:
        if worker:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        log("Nahaj API stopped")


app = FastAPI(title="Nahaj API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def enforce_root_auth(request: Request, call_next):
    """Keep every API surface private except the liveness endpoint."""

    request_id = uuid4().hex[:12]
    request.state.request_id = request_id
    started_at = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    log(f"Request started id={request_id} method={request.method} path={request.url.path} client={client}")
    try:
        if request.url.path == "/health":
            response = await call_next(request)
        elif not settings.api_key:
            response = JSONResponse(
                status_code=503,
                content={"error": {"code": "api_key_not_configured", "message": "The Nahaj API key is not configured.", "details": {}}},
            )
        else:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not secrets.compare_digest(token, settings.api_key):
                response = JSONResponse(
                    status_code=401,
                    content={"error": {"code": "unauthorized", "message": "A valid Nahaj bearer token is required.", "details": {}}},
                )
            else:
                response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log(f"Request failed id={request_id} duration_ms={duration_ms} error={exc}")
        raise

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    log(f"Request completed id={request_id} status={response.status_code} duration_ms={duration_ms}")
    return response


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    log(
        f"API error id={getattr(request.state, 'request_id', 'unknown')} "
        f"status={exc.status_code} code={exc.code} message={exc.message}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    log(
        f"Validation error id={getattr(request.state, 'request_id', 'unknown')} "
        f"path={request.url.path} error_count={len(exc.errors())}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request contains invalid fields.",
                "details": {"errors": exc.errors()},
            }
        },
    )


def require_api_key(authorization: str = Header(default="")) -> None:
    if not settings.api_key:
        _error(503, "api_key_not_configured", "The Nahaj API key is not configured.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, settings.api_key):
        _error(401, "unauthorized", "A valid Nahaj bearer token is required.")


def _active_semester(db: Session) -> Semester | None:
    return db.scalar(select(Semester).where(Semester.active.is_(True)).order_by(Semester.created_at.desc()))


def _get_course(db: Session, course_id: str) -> Course:
    course = db.get(Course, course_id)
    if not course:
        _error(404, "course_not_found", "Course was not found.", {"course_id": course_id})
    return course


def _get_document(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if not document:
        _error(404, "document_not_found", "Document was not found.", {"document_id": document_id})
    return document


def _get_task(db: Session, task_id: str) -> StudyTask:
    task = db.get(StudyTask, task_id)
    if not task:
        _error(404, "task_not_found", "Study task was not found.", {"task_id": task_id})
    return task


def _get_quiz(db: Session, quiz_id: str) -> Quiz:
    quiz = db.scalar(
        select(Quiz)
        .options(joinedload(Quiz.questions).joinedload(QuizQuestion.options))
        .where(Quiz.id == quiz_id)
    )
    if not quiz:
        _error(404, "quiz_not_found", "Quiz was not found.", {"quiz_id": quiz_id})
    return quiz


def _get_attempt(db: Session, attempt_id: str) -> QuizAttempt:
    attempt = db.scalar(
        select(QuizAttempt)
        .options(
            joinedload(QuizAttempt.quiz).joinedload(Quiz.questions).joinedload(QuizQuestion.options),
            joinedload(QuizAttempt.answers),
        )
        .where(QuizAttempt.id == attempt_id)
    )
    if not attempt:
        _error(404, "attempt_not_found", "Quiz attempt was not found.", {"attempt_id": attempt_id})
    return attempt


def _iso_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            _error(422, "invalid_datetime", f"{field_name} must be an ISO-8601 datetime.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _task_out(task: StudyTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "course_id": task.course_id,
        "title": task.title,
        "notes": task.notes,
        "start_at": task.start_at,
        "end_at": task.end_at,
        "status": task.status,
        "origin": task.origin,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _question_public(question: QuizQuestion, include_explanation: bool = False) -> dict[str, Any]:
    return {
        "id": question.id,
        "prompt": question.prompt,
        "topic": question.topic,
        "explanation": question.explanation if include_explanation else None,
        "options": [
            {"option_id": option.option_id, "text": option.text}
            for option in sorted(question.options, key=lambda item: item.position)
        ],
    }


def _quiz_list_item(db: Session, quiz: Quiz) -> dict[str, Any]:
    completed = db.scalars(
        select(QuizAttempt).where(QuizAttempt.quiz_id == quiz.id, QuizAttempt.status == "completed")
    ).all()
    scores = [attempt.score_percent for attempt in completed if attempt.score_percent is not None]
    return {
        "id": quiz.id,
        "course_id": quiz.course_id,
        "title": quiz.title,
        "source_collections": quiz.source_collections or [],
        "question_count": len(quiz.questions),
        "created_at": quiz.created_at,
        "latest_score": scores[-1] if scores else None,
        "best_score": max(scores) if scores else None,
    }


def _attempt_out(attempt: QuizAttempt) -> dict[str, Any]:
    answer_by_question = {answer.question_id: answer for answer in attempt.answers}
    if attempt.status == "completed":
        questions: list[dict[str, Any]] = []
        for question in attempt.quiz.questions:
            answer = answer_by_question.get(question.id)
            public = _question_public(question, include_explanation=True)
            public.update(
                {
                    "correct_option_id": question.correct_option_id,
                    "selected_option_id": answer.selected_option_id if answer else None,
                    "is_correct": bool(answer and answer.is_correct),
                }
            )
            questions.append(public)
    else:
        questions = [_question_public(question) for question in attempt.quiz.questions]
    return {
        "id": attempt.id,
        "quiz_id": attempt.quiz_id,
        "status": attempt.status,
        "score_percent": attempt.score_percent,
        "weak_topics": attempt.weak_topics or [],
        "started_at": attempt.started_at,
        "completed_at": attempt.completed_at,
        "questions": questions,
    }


def _validate_upload(filename: str, media_type: str | None, content: bytes) -> tuple[str, str]:
    if (
        not filename
        or Path(filename).name != filename
        or "\x00" in filename
        or "\\" in filename
        or any(ord(character) < 32 for character in filename)
    ):
        _error(400, "unsafe_filename", "The uploaded filename is not safe.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        _error(415, "unsupported_file_type", "Only PDF, PPTX, DOCX, PNG, and JPEG files are supported.")
    if len(content) > settings.max_upload_bytes:
        _error(413, "file_too_large", "The file exceeds the 50 MiB upload limit.")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        _error(415, "invalid_file_signature", "The uploaded file is not a valid PDF.")
    if suffix in {".pptx", ".docx"} and not content.startswith(b"PK"):
        _error(415, "invalid_file_signature", "The uploaded Office file is not a valid ZIP package.")
    if suffix == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        _error(415, "invalid_file_signature", "The uploaded file is not a valid PNG.")
    if suffix in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
        _error(415, "invalid_file_signature", "The uploaded file is not a valid JPEG.")
    detected = mimetypes.guess_type(filename)[0] or media_type or "application/octet-stream"
    if detected not in ALLOWED_MIME_TYPES and media_type not in ALLOWED_MIME_TYPES:
        _error(415, "unsupported_media_type", "The uploaded media type is not supported.")
    return suffix, detected if detected in ALLOWED_MIME_TYPES else str(media_type)


def _stored_path(document_id: str, course_id: str, collection: str, suffix: str) -> Path:
    base = settings.upload_dir.resolve()
    path = (base / course_id / collection / document_id / f"original{suffix}").resolve()
    if base not in path.parents:
        _error(500, "unsafe_storage_path", "Could not determine a safe upload path.")
    return path


def _source_payload(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": {"id": citation.get("document_id"), "name": citation.get("label", "Course material"), "type": "file"},
            "document": [citation.get("excerpt", "")],
            "metadata": [
                {
                    "file_id": citation.get("document_id"),
                    "name": citation.get("label", "Course material"),
                    "source": citation.get("label", "Course material"),
                    "page": citation.get("page"),
                    "slide": citation.get("slide"),
                }
            ],
        }
        for citation in citations
    ]


def _citation_footer(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    lines = ["\n\nSources:"]
    for index, citation in enumerate(citations, start=1):
        label = citation.get("label", "Course material")
        page = citation.get("page") or citation.get("slide")
        suffix = f", p. {page}" if page else ""
        document_id = citation.get("document_id")
        link = f"/api/nahaj/api/v1/documents/{document_id}/content" if document_id else "#"
        lines.append(f"[{index}] [{label}{suffix}]({link})")
    return "\n".join(lines)


def _coerce_event(event: Any) -> SupervisorEvent:
    """Normalize the separately-owned gateway's tagged event representation."""

    if isinstance(event, SupervisorEvent):
        return event
    if isinstance(event, dict):
        return SupervisorEvent(
            kind=str(event.get("kind", event.get("type", ""))),
            data=dict(
                event.get("data")
                or {key: value for key, value in event.items() if key not in {"kind", "type"}}
            ),
        )
    return SupervisorEvent(
        kind=str(getattr(event, "kind", getattr(event, "type", ""))),
        data=dict(getattr(event, "data", {}) or {}),
    )


def _apply_event(db: Session, event: SupervisorEvent, state: dict[str, Any]) -> None:
    data = event.data or {}
    if event.kind == "text_delta":
        state["text"].append(str(data.get("text", "")))
        return
    if event.kind == "citation":
        state["citations"].append(data)
        return
    if event.kind == "document_status":
        document_id = data.get("document_id") or state.get("document_id")
        if document_id:
            document = db.get(Document, document_id)
            if document:
                status = str(data.get("status", document.status))
                if status not in {"ready", "failed"}:
                    _error(422, "invalid_document_status", "Document status must be ready or failed.")
                document.status = status
                document.error = data.get("error")
                document.updated_at = now_utc()
                db.commit()
        return
    if event.kind == "task_upsert":
        task_id = data.get("id")
        task = db.get(StudyTask, task_id) if task_id else None
        if task is None:
            task = StudyTask(id=task_id or str(uuid4()))
            db.add(task)
        if data.get("course_id"):
            _get_course(db, data["course_id"])
        task.course_id = data.get("course_id", task.course_id)
        task.title = str(data.get("title", task.title or "Study task"))
        task.notes = data.get("notes", task.notes)
        task.start_at = _iso_datetime(data.get("start_at", task.start_at or now_utc()), "start_at")
        task.end_at = _iso_datetime(data.get("end_at", task.end_at or now_utc()), "end_at")
        if task.end_at <= task.start_at:
            _error(422, "invalid_task_range", "A task end must be after its start.")
        status = data.get("status", task.status or "todo")
        if status not in {"todo", "done", "skipped"}:
            _error(422, "invalid_task_status", "Task status must be todo, done, or skipped.")
        task.status = status
        task.origin = "supervisor"
        task.updated_at = now_utc()
        db.commit()
        state["plan_changed"] = True
        return
    if event.kind == "task_delete":
        task = db.get(StudyTask, data.get("id"))
        if task:
            db.delete(task)
            db.commit()
            state["plan_changed"] = True
        return
    if event.kind == "quiz_create":
        questions = data.get("questions") or []
        if not questions:
            _error(422, "invalid_quiz", "A quiz must contain at least one question.")
        quiz = Quiz(
            id=str(uuid4()),
            course_id=data.get("course_id"),
            title=str(data.get("title", "Practice quiz")),
            source_collections=list(data.get("source_collections") or []),
        )
        if any(collection not in ALLOWED_COLLECTIONS for collection in quiz.source_collections):
            _error(422, "invalid_collection", "Quiz source collections must be slides or past_exam.")
        if quiz.course_id:
            _get_course(db, quiz.course_id)
        db.add(quiz)
        db.flush()
        for position, question_data in enumerate(questions):
            if not isinstance(question_data, dict):
                db.rollback()
                _error(422, "invalid_quiz_question", "Every quiz question must be an object.")
            options = question_data.get("options") or []
            if not isinstance(options, list):
                db.rollback()
                _error(422, "invalid_quiz_question", "Every quiz question must contain an options array.")
            if any(not isinstance(option, dict) for option in options):
                db.rollback()
                _error(422, "invalid_quiz_question", "Every quiz option must be an object.")
            ids = [str(option.get("option_id")) for option in options]
            correct = str(question_data.get("correct_option_id", ""))
            if (
                not str(question_data.get("prompt", "")).strip()
                or len(options) != 4
                or len(set(ids)) != 4
                or any(not option_id.strip() or not str(option.get("text", "")).strip() for option_id, option in zip(ids, options))
                or correct not in ids
            ):
                db.rollback()
                _error(422, "invalid_quiz_question", "Every MCQ must contain four unique options and one correct option.")
            question = QuizQuestion(
                id=str(uuid4()),
                quiz_id=quiz.id,
                prompt=str(question_data.get("prompt", "")),
                topic=question_data.get("topic"),
                explanation=question_data.get("explanation"),
                correct_option_id=correct,
                position=position,
            )
            db.add(question)
            db.flush()
            for option_position, option_data in enumerate(options):
                db.add(
                    QuizOption(
                        id=str(uuid4()),
                        question_id=question.id,
                        option_id=ids[option_position],
                        text=str(option_data.get("text", "")),
                        position=option_position,
                    )
                )
        db.commit()
        state["created_links"].append(("Open quiz", f"/quizzes/{quiz.id}"))
        return
    if event.kind == "notification_create":
        severity = str(data.get("severity", "info"))
        title = str(data.get("title", "Study update"))
        message = str(data.get("message", "You have a new study update."))
        target_url = data.get("target_url")
        dedupe_key = data.get("dedupe_key")
        if not dedupe_key:
            dedupe_key = hashlib.sha256(
                json.dumps(
                    {"severity": severity, "title": title, "message": message, "target_url": target_url},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        if dedupe_key:
            existing = db.scalar(select(Notification).where(Notification.dedupe_key == str(dedupe_key), Notification.read_at.is_(None)))
            if existing:
                return
        db.add(
            Notification(
                severity=severity,
                title=title,
                message=message,
                target_url=target_url,
                dedupe_key=str(dedupe_key) if dedupe_key else None,
            )
        )
        db.commit()


async def _consume_supervisor(request: SupervisorRequest, db: Session, state: dict[str, Any]) -> None:
    event_count = 0
    log(f"Supervisor started request_type={request.type}")
    try:
        async for raw_event in get_supervisor().handle(request):
            event = _coerce_event(raw_event)
            event_count += 1
            if event.kind != "text_delta":
                log(f"Supervisor event request_type={request.type} kind={event.kind}")
            if request.type != "chat" and event.kind == "document_status":
                state["document_id"] = request.document.get("id") if request.document else None
            _apply_event(db, event, state)
        log(
            f"Supervisor completed request_type={request.type} events={event_count} "
            f"text_chunks={len(state['text'])} citations={len(state['citations'])} "
            f"created_links={len(state['created_links'])}"
        )
    except SupervisorUnavailable:
        raise APIError(503, "supervisor_unavailable", "The supervisor is not configured.")
    except APIError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary around user agent code
        log(f"Supervisor request failed: {exc}")
        raise APIError(502, "supervisor_error", "The supervisor failed to complete the request.", {"reason": str(exc)})


async def _process_document_ingest(document_id: str) -> None:
    log(f"Document ingestion started document_id={document_id}")
    db = next(get_db())
    try:
        document = db.get(Document, document_id)
        if not document:
            return
        payload = {
            "id": document.id,
            "course_id": document.course_id,
            "collection": document.collection,
            "filename": document.filename,
            "path": document.stored_path,
            "media_type": document.media_type,
        }
        # Course materials are indexed once, at upload time. Old exams stay on
        # the existing supervisor path and are intentionally outside the RAG index.
        if document.collection == "slides":
            CourseIngestor().ingest_document(payload)
            document.status = "ready"
            document.error = None
            document.updated_at = now_utc()
            db.commit()
            log(f"Document ingestion completed document_id={document_id} collection=slides")
            return
        state: dict[str, Any] = {"text": [], "citations": [], "created_links": [], "document_id": document_id}
        request = SupervisorRequest(
            type="document_ingest",
            document=payload,
        )
        await _consume_supervisor(request, db, state)
        if document.status == "processing":
            document.status = "ready"
            document.updated_at = now_utc()
            db.commit()
        log(f"Document ingestion completed document_id={document_id} collection={document.collection}")
    except APIError as exc:
        log(f"Document ingestion failed document_id={document_id} code={exc.code} error={exc.message}")
        document = db.get(Document, document_id)
        if document:
            document.status = "failed"
            document.error = exc.message
            document.updated_at = now_utc()
            db.commit()
    except Exception as exc:  # pragma: no cover - background dependency/runtime boundary
        log(f"Document ingestion failed: {exc}")
        document = db.get(Document, document_id)
        if document:
            document.status = "failed"
            document.error = str(exc)
            document.updated_at = now_utc()
            db.commit()
    finally:
        db.close()


def _progress_snapshot(db: Session) -> dict[str, Any]:
    tasks = db.scalars(select(StudyTask)).all()
    return {
        "total_tasks": len(tasks),
        "completed_tasks": sum(task.status == "done" for task in tasks),
        "incomplete_tasks": sum(task.status == "todo" for task in tasks),
        "tasks": [_task_out(task) for task in tasks],
    }


def _slide_count(document: Document) -> int | None:
    """Read a lightweight slide/page count for the planner's library view."""

    path = Path(document.stored_path)
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".pptx":
            with ZipFile(path) as archive:
                return sum(bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name)) for name in archive.namelist())
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            return len(PdfReader(str(path)).pages)
    except Exception:
        return None
    return None


def _shared_state_snapshot(db: Session) -> dict[str, Any]:
    """Project database records into the shared state expected by the agents."""

    semester = _active_semester(db)
    courses = db.scalars(select(Course).order_by(Course.code, Course.name)).all()
    documents = db.scalars(select(Document).where(Document.collection == "slides")).all()
    tasks = db.scalars(select(StudyTask).order_by(StudyTask.start_at)).all()
    quizzes = db.scalars(select(Quiz)).all()
    questions = db.scalars(select(QuizQuestion)).all()
    attempts = db.scalars(select(QuizAttempt).where(QuizAttempt.status == "completed")).all()
    answers = db.scalars(select(QuizAnswer)).all()

    documents_by_course: dict[str, list[Document]] = {}
    tasks_by_course: dict[str | None, list[StudyTask]] = {}
    questions_by_quiz: dict[str, list[QuizQuestion]] = {}
    answers_by_attempt: dict[str, dict[str, QuizAnswer]] = {}
    quizzes_by_id = {quiz.id: quiz for quiz in quizzes}
    for document in documents:
        documents_by_course.setdefault(document.course_id, []).append(document)
    for task in tasks:
        tasks_by_course.setdefault(task.course_id, []).append(task)
    for question in questions:
        questions_by_quiz.setdefault(question.quiz_id, []).append(question)
    for answer in answers:
        answers_by_attempt.setdefault(answer.attempt_id, {})[answer.question_id] = answer

    performance: dict[tuple[str | None, str], list[int]] = {}
    for attempt in attempts:
        quiz = quizzes_by_id.get(attempt.quiz_id)
        if not quiz:
            continue
        for question in questions_by_quiz.get(quiz.id, []):
            answer = answers_by_attempt.get(attempt.id, {}).get(question.id)
            if not answer:
                continue
            topic = str(question.topic or "General")
            bucket = performance.setdefault((quiz.course_id, topic), [0, 0])
            bucket[0] += int(answer.is_correct)
            bucket[1] += 1

    state_courses: list[dict[str, Any]] = []
    for course in courses:
        course_tasks = tasks_by_course.get(course.id, [])
        completed_tasks = [
            {"id": task.id, "title": task.title, "description": task.notes or "", "date": task.start_at.date().isoformat()}
            for task in course_tasks
            if task.status == "done"
        ]
        remaining_tasks = [
            {"id": task.id, "title": task.title, "description": task.notes or "", "date": task.start_at.date().isoformat()}
            for task in course_tasks
            if task.status == "todo"
        ]
        deadlines = [
            {"title": task.title, "date": task.start_at.date().isoformat(), "type": "assessment"}
            for task in course_tasks
            if any(word in task.title.casefold() for word in ("exam", "quiz", "midterm", "final", "assessment"))
        ]
        state_courses.append(
            {
                "id": course.id,
                "course_name": course.name,
                "code": course.code,
                "deadlines": deadlines,
                "materials": [
                    {
                        "title": document.filename,
                        "document_id": document.id,
                        "course_id": course.id,
                        "collection": document.collection,
                        "slides": _slide_count(document),
                        "status": document.status,
                    }
                    for document in documents_by_course.get(course.id, [])
                ],
                "completed_topics": completed_tasks,
                "current_plan": [
                    {
                        "title": f"{course.name} study plan",
                        "tasks": completed_tasks + remaining_tasks,
                        "completed_tasks": completed_tasks,
                        "remaining_tasks": remaining_tasks,
                    }
                ],
                "quiz_performance": [
                    {
                        "topic": topic,
                        "estimated_mastery_level": correct / total if total else 0.5,
                    }
                    for (course_id, topic), (correct, total) in performance.items()
                    if course_id == course.id
                ],
            }
        )
    return {
        "semester": {
            "id": semester.id,
            "name": semester.name,
            "start_date": semester.start_date.isoformat(),
            "end_date": semester.end_date.isoformat(),
            "timezone": semester.timezone,
        }
        if semester
        else None,
        "courses": state_courses,
    }


async def _run_progress_check() -> None:
    log("Progress check started")
    db = next(get_db())
    try:
        state: dict[str, Any] = {"text": [], "citations": [], "created_links": []}
        await _consume_supervisor(
            SupervisorRequest(type="progress_check", snapshot=_progress_snapshot(db)), db, state
        )
        log("Progress check completed")
    except APIError as exc:
        log(f"Progress check failed: {exc.message}")
    finally:
        db.close()


async def _progress_worker() -> None:
    while True:
        await asyncio.sleep(settings.progress_check_interval_minutes * 60)
        await _run_progress_check()


@app.get("/health")
def health():
    return {"status": "ok", "service": "nahaj-api", "version": app.version}


@app.get("/api/v1/semester", response_model=SemesterOut | None, dependencies=[Depends(require_api_key)])
def get_semester(db: Session = Depends(get_db)):
    return _active_semester(db)


@app.put("/api/v1/semester", response_model=SemesterOut, dependencies=[Depends(require_api_key)])
def put_semester(payload: SemesterInput, db: Session = Depends(get_db)):
    semester = _active_semester(db)
    if semester is None:
        semester = Semester()
        db.add(semester)
    for key, value in payload.model_dump().items():
        setattr(semester, key, value)
    for other in db.scalars(select(Semester).where(Semester.id != semester.id)).all():
        other.active = False
    semester.active = True
    semester.updated_at = now_utc()
    db.commit()
    db.refresh(semester)
    return semester


@app.get("/api/v1/courses", response_model=list[CourseOut], dependencies=[Depends(require_api_key)])
def list_courses(db: Session = Depends(get_db)):
    return db.scalars(select(Course).order_by(Course.code, Course.name)).all()


@app.post("/api/v1/courses", response_model=CourseOut, status_code=201, dependencies=[Depends(require_api_key)])
def create_course(payload: CourseCreate, db: Session = Depends(get_db)):
    semester = _active_semester(db)
    if not semester:
        _error(409, "semester_required", "Create an active semester before adding courses.")
    course = Course(semester_id=semester.id, **payload.model_dump())
    db.add(course)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        _error(409, "duplicate_course_code", "A course with this code already exists in the semester.")
    db.refresh(course)
    return course


@app.get("/api/v1/courses/{course_id}", response_model=CourseOut, dependencies=[Depends(require_api_key)])
def get_course(course_id: str, db: Session = Depends(get_db)):
    return _get_course(db, course_id)


@app.patch("/api/v1/courses/{course_id}", response_model=CourseOut, dependencies=[Depends(require_api_key)])
def update_course(course_id: str, payload: CourseUpdate, db: Session = Depends(get_db)):
    course = _get_course(db, course_id)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(course, key, value)
    course.updated_at = now_utc()
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        _error(409, "duplicate_course_code", "A course with this code already exists in the semester.")
    db.refresh(course)
    return course


@app.delete("/api/v1/courses/{course_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_course(course_id: str, db: Session = Depends(get_db)):
    course = _get_course(db, course_id)
    if course.documents or course.tasks or course.quizzes:
        _error(409, "course_not_empty", "Remove the course's documents, tasks, and quizzes before deleting it.")
    db.delete(course)
    db.commit()


@app.get("/api/v1/courses/{course_id}/documents", response_model=list[DocumentOut], dependencies=[Depends(require_api_key)])
def list_documents(course_id: str, collection: str | None = Query(default=None), db: Session = Depends(get_db)):
    _get_course(db, course_id)
    query = select(Document).where(Document.course_id == course_id).order_by(Document.created_at.desc())
    if collection:
        if collection not in ALLOWED_COLLECTIONS:
            _error(422, "invalid_collection", "collection must be slides or past_exam.")
        query = query.where(Document.collection == collection)
    return db.scalars(query).all()


@app.post("/api/v1/courses/{course_id}/documents", response_model=DocumentOut, status_code=202, dependencies=[Depends(require_api_key)])
async def upload_document(
    course_id: str,
    background_tasks: BackgroundTasks,
    collection: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _get_course(db, course_id)
    if collection not in ALLOWED_COLLECTIONS:
        _error(422, "invalid_collection", "collection must be slides or past_exam.")
    filename = (file.filename or "upload").strip()
    content = await file.read(settings.max_upload_bytes + 1)
    suffix, detected_media_type = _validate_upload(filename, file.content_type, content)
    digest = hashlib.sha256(content).hexdigest()
    duplicate = db.scalar(
        select(Document).where(
            Document.course_id == course_id,
            Document.collection == collection,
            Document.sha256 == digest,
        )
    )
    if duplicate:
        _error(409, "duplicate_document", "This file is already in the selected course collection.")
    document = Document(
        course_id=course_id,
        collection=collection,
        filename=filename,
        media_type=detected_media_type,
        size_bytes=len(content),
        sha256=digest,
        stored_path="",
        status="processing",
    )
    db.add(document)
    db.flush()
    path = _stored_path(document.id, course_id, collection, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    document.stored_path = str(path)
    db.commit()
    db.refresh(document)
    background_tasks.add_task(_process_document_ingest, document.id)
    return document


@app.get("/api/v1/documents/{document_id}", response_model=DocumentOut, dependencies=[Depends(require_api_key)])
def get_document(document_id: str, db: Session = Depends(get_db)):
    return _get_document(db, document_id)


@app.get("/api/v1/documents/{document_id}/content", dependencies=[Depends(require_api_key)])
def document_content(document_id: str, db: Session = Depends(get_db)):
    document = _get_document(db, document_id)
    path = Path(document.stored_path)
    if not path.exists():
        _error(404, "file_missing", "The original file is no longer available.")
    return FileResponse(path, media_type=document.media_type, filename=document.filename)


@app.post("/api/v1/documents/{document_id}/retry", response_model=DocumentOut, status_code=202, dependencies=[Depends(require_api_key)])
def retry_document(document_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    document = _get_document(db, document_id)
    if document.status != "failed":
        _error(409, "document_not_failed", "Only failed documents can be retried.")
    document.status = "processing"
    document.error = None
    document.updated_at = now_utc()
    db.commit()
    db.refresh(document)
    background_tasks.add_task(_process_document_ingest, document.id)
    return document


@app.delete("/api/v1/documents/{document_id}", status_code=204, dependencies=[Depends(require_api_key)])
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = _get_document(db, document_id)
    if document.collection == "slides":
        try:
            CourseIngestor().remove_document(document.course_id, document.id)
        except Exception as exc:  # Removing the database record should not depend on an index being available.
            log(f"Could not remove document {document.id} from the RAG index: {exc}")
    else:
        state: dict[str, Any] = {"text": [], "citations": [], "created_links": [], "document_id": document.id}
        await _consume_supervisor(
            SupervisorRequest(
                type="document_remove",
                document={"id": document.id, "course_id": document.course_id, "collection": document.collection},
            ),
            db,
            state,
        )
    path = Path(document.stored_path)
    db.delete(document)
    db.commit()
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
        path.parent.parent.rmdir()
    except OSError:
        pass


@app.get("/api/v1/plan", response_model=PlanOut, dependencies=[Depends(require_api_key)])
def get_plan(
    from_at: datetime | None = Query(default=None, alias="from"),
    to_at: datetime | None = Query(default=None, alias="to"),
    course_id: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
):
    query = select(StudyTask).order_by(StudyTask.start_at)
    if from_at:
        query = query.where(StudyTask.end_at >= _iso_datetime(from_at, "from"))
    if to_at:
        query = query.where(StudyTask.start_at <= _iso_datetime(to_at, "to"))
    if course_id:
        query = query.where(StudyTask.course_id == course_id)
    if status:
        if status not in {"todo", "done", "skipped"}:
            _error(422, "invalid_task_status", "status must be todo, done, or skipped.")
        query = query.where(StudyTask.status == status)
    tasks = db.scalars(query).all()
    total = len(tasks)
    done = sum(task.status == "done" for task in tasks)
    return {"semester": _active_semester(db), "tasks": tasks, "summary": {"total": total, "done": done, "completion_percent": (done / total * 100 if total else 0)}}


@app.post("/api/v1/tasks", response_model=TaskOut, status_code=201, dependencies=[Depends(require_api_key)])
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    if payload.course_id:
        _get_course(db, payload.course_id)
    values = payload.model_dump()
    values["start_at"] = _iso_datetime(values["start_at"], "start_at")
    values["end_at"] = _iso_datetime(values["end_at"], "end_at")
    task = StudyTask(**values)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@app.patch("/api/v1/tasks/{task_id}", response_model=TaskOut, dependencies=[Depends(require_api_key)])
def update_task(task_id: str, payload: TaskUpdate, db: Session = Depends(get_db)):
    task = _get_task(db, task_id)
    values = payload.model_dump(exclude_unset=True)
    if "course_id" in values and values["course_id"]:
        _get_course(db, values["course_id"])
    start_at = _iso_datetime(values.get("start_at", task.start_at), "start_at")
    end_at = _iso_datetime(values.get("end_at", task.end_at), "end_at")
    if end_at <= start_at:
        _error(422, "invalid_task_range", "A task end must be after its start.")
    values["start_at"] = start_at
    values["end_at"] = end_at
    for key, value in values.items():
        setattr(task, key, value)
    task.updated_at = now_utc()
    db.commit()
    db.refresh(task)
    return task


@app.delete("/api/v1/tasks/{task_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_task(task_id: str, db: Session = Depends(get_db)):
    task = _get_task(db, task_id)
    db.delete(task)
    db.commit()


@app.get("/api/v1/progress", response_model=ProgressOut, dependencies=[Depends(require_api_key)])
def get_progress(db: Session = Depends(get_db)):
    courses = db.scalars(select(Course).order_by(Course.code)).all()
    tasks = db.scalars(select(StudyTask)).all()
    completed_attempts = db.scalars(select(QuizAttempt).where(QuizAttempt.status == "completed")).all()
    quiz_scores = [attempt.score_percent for attempt in completed_attempts if attempt.score_percent is not None]
    course_progress = []
    for course in courses:
        course_tasks = [task for task in tasks if task.course_id == course.id]
        course_attempts = [attempt for attempt in completed_attempts if attempt.quiz.course_id == course.id]
        scores = [attempt.score_percent for attempt in course_attempts if attempt.score_percent is not None]
        course_progress.append(
            {
                "course_id": course.id,
                "code": course.code,
                "name": course.name,
                "completed_tasks": sum(task.status == "done" for task in course_tasks),
                "total_tasks": len(course_tasks),
                "quiz_average_percent": (sum(scores) / len(scores) if scores else None),
            }
        )
    return {
        "task_completion_percent": (sum(task.status == "done" for task in tasks) / len(tasks) * 100 if tasks else 0),
        "quiz_average_percent": (sum(quiz_scores) / len(quiz_scores) if quiz_scores else None),
        "courses": course_progress,
    }


@app.get("/api/v1/quizzes", response_model=list[QuizListOut], dependencies=[Depends(require_api_key)])
def list_quizzes(db: Session = Depends(get_db)):
    quizzes = db.scalars(select(Quiz).options(joinedload(Quiz.questions)).order_by(Quiz.created_at.desc())).unique().all()
    return [_quiz_list_item(db, quiz) for quiz in quizzes]


@app.get("/api/v1/quizzes/{quiz_id}", response_model=QuizDetailOut, dependencies=[Depends(require_api_key)])
def get_quiz(quiz_id: str, db: Session = Depends(get_db)):
    quiz = _get_quiz(db, quiz_id)
    return {**_quiz_list_item(db, quiz), "questions": [_question_public(question) for question in quiz.questions]}


@app.delete("/api/v1/quizzes/{quiz_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_quiz(quiz_id: str, db: Session = Depends(get_db)):
    quiz = _get_quiz(db, quiz_id)
    db.delete(quiz)
    db.commit()


@app.post("/api/v1/quizzes/{quiz_id}/attempts", response_model=AttemptStartOut, status_code=201, dependencies=[Depends(require_api_key)])
def start_attempt(quiz_id: str, db: Session = Depends(get_db)):
    quiz = _get_quiz(db, quiz_id)
    attempt = QuizAttempt(quiz_id=quiz.id, status="in_progress")
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return {"id": attempt.id, "quiz_id": quiz.id, "status": attempt.status, "started_at": attempt.started_at, "questions": [_question_public(question) for question in quiz.questions]}


@app.get("/api/v1/attempts/{attempt_id}", response_model=AttemptOut, dependencies=[Depends(require_api_key)])
def get_attempt(attempt_id: str, db: Session = Depends(get_db)):
    return _attempt_out(_get_attempt(db, attempt_id))


@app.post("/api/v1/attempts/{attempt_id}/answers", response_model=dict[str, Any], dependencies=[Depends(require_api_key)])
def submit_answer(attempt_id: str, payload: AnswerInput, db: Session = Depends(get_db)):
    attempt = _get_attempt(db, attempt_id)
    if attempt.status != "in_progress":
        _error(409, "attempt_completed", "This quiz attempt is already complete.")
    question = next((item for item in attempt.quiz.questions if item.id == payload.question_id), None)
    if question is None:
        _error(404, "question_not_found", "The question does not belong to this quiz.")
    if payload.selected_option_id not in {option.option_id for option in question.options}:
        _error(422, "invalid_option", "The selected option does not belong to this question.")
    answer = db.scalar(select(QuizAnswer).where(QuizAnswer.attempt_id == attempt.id, QuizAnswer.question_id == question.id))
    if answer is None:
        answer = QuizAnswer(attempt_id=attempt.id, question_id=question.id, selected_option_id=payload.selected_option_id, is_correct=payload.selected_option_id == question.correct_option_id)
        db.add(answer)
    else:
        answer.selected_option_id = payload.selected_option_id
        answer.is_correct = payload.selected_option_id == question.correct_option_id
        answer.answered_at = now_utc()
    db.commit()
    return {"question_id": question.id, "selected_option_id": payload.selected_option_id, "answered": True}


@app.post("/api/v1/attempts/{attempt_id}/complete", response_model=AttemptOut, dependencies=[Depends(require_api_key)])
def complete_attempt(attempt_id: str, db: Session = Depends(get_db)):
    attempt = _get_attempt(db, attempt_id)
    if attempt.status != "in_progress":
        return _attempt_out(attempt)
    answer_by_question = {answer.question_id: answer for answer in attempt.answers}
    missing = [question.id for question in attempt.quiz.questions if question.id not in answer_by_question]
    if missing:
        _error(409, "unanswered_questions", "Answer every question before completing the quiz.", {"question_ids": missing})
    total = len(attempt.quiz.questions)
    correct = sum(answer.is_correct for answer in answer_by_question.values())
    attempt.score_percent = round(correct / total * 100, 2) if total else 0
    attempt.weak_topics = sorted({question.topic for question in attempt.quiz.questions if not answer_by_question[question.id].is_correct and question.topic})
    attempt.status = "completed"
    attempt.completed_at = now_utc()
    db.commit()
    db.refresh(attempt)
    return _attempt_out(attempt)


@app.get("/api/v1/notifications", response_model=list[NotificationOut], dependencies=[Depends(require_api_key)])
def list_notifications(unread: bool = False, db: Session = Depends(get_db)):
    query = select(Notification).order_by(Notification.created_at.desc())
    if unread:
        query = query.where(Notification.read_at.is_(None))
    return db.scalars(query).all()


@app.patch("/api/v1/notifications/{notification_id}", response_model=NotificationOut, dependencies=[Depends(require_api_key)])
def update_notification(notification_id: str, payload: dict[str, Any], db: Session = Depends(get_db)):
    notification = db.get(Notification, notification_id)
    if not notification:
        _error(404, "notification_not_found", "Notification was not found.")
    if payload.get("read") is True:
        notification.read_at = now_utc()
    elif payload.get("read") is False:
        notification.read_at = None
    db.commit()
    db.refresh(notification)
    return notification


@app.post("/api/v1/notifications/read-all", status_code=204, dependencies=[Depends(require_api_key)])
def read_all_notifications(db: Session = Depends(get_db)):
    for notification in db.scalars(select(Notification).where(Notification.read_at.is_(None))).all():
        notification.read_at = now_utc()
    db.commit()


def _openai_chunk(completion_id: str, model: str, delta: dict[str, Any], finish_reason: str | None = None) -> str:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_openai_models():
    return {"object": "list", "data": [{"id": "nahaj-supervisor", "object": "model", "owned_by": "nahaj"}]}


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(payload: dict[str, Any], db: Session = Depends(get_db)):
    model = payload.get("model")
    if model != "nahaj-supervisor":
        _error(404, "model_not_found", "Only nahaj-supervisor is available.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        _error(422, "messages_required", "At least one chat message is required.")
    stream = bool(payload.get("stream", False))
    context = payload.get("nahaj_context") or {}
    if not isinstance(context, dict):
        _error(422, "invalid_nahaj_context", "nahaj_context must be an object.")
    document_ids = context.get("document_ids") or []
    if not isinstance(document_ids, list) or any(not isinstance(item, str) for item in document_ids):
        _error(422, "invalid_nahaj_context", "nahaj_context.document_ids must be a list of IDs.")
    request = SupervisorRequest(
        type="chat",
        messages=messages,
        context={
            "course_id": context.get("course_id"),
            "document_ids": document_ids,
            # Open WebUI sends a stable chat_id with each turn. Reuse it as
            # the LangGraph thread so approvals can resume the same workflow.
            "thread_id": str(
                context.get("thread_id")
                or payload.get("chat_id")
                or payload.get("conversation_id")
                or payload.get("session_id")
                or payload.get("id")
                or "nahaj-default"
            ),
            "semester": _active_semester(db).id if _active_semester(db) else None,
            "shared_state": _shared_state_snapshot(db),
        },
    )
    completion_id = f"chatcmpl_{uuid4().hex}"

    if not stream:
        state: dict[str, Any] = {"text": [], "citations": [], "created_links": []}
        await _consume_supervisor(request, db, state)
        content = "".join(state["text"]).strip()
        if state.get("plan_changed"):
            state["created_links"].append(("Open your plan", "/plan"))
        content += _citation_footer(state["citations"])
        if state["created_links"]:
            content += "\n\n" + "\n".join(f"[{label}]({url})" for label, url in state["created_links"])
        content = safe_final_response(content)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "sources": _source_payload(state["citations"]),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def stream_response():
        state: dict[str, Any] = {"text": [], "citations": [], "created_links": []}
        try:
            async for raw_event in get_supervisor().handle(request):
                event = _coerce_event(raw_event)
                _apply_event(db, event, state)
                if event.kind == "status":
                    yield _openai_chunk(completion_id, model, {"content": ""})
            if state.get("plan_changed"):
                state["created_links"].append(("Open your plan", "/plan"))
            footer = _citation_footer(state["citations"])
            if state["created_links"]:
                footer += "\n\n" + "\n".join(f"[{label}]({url})" for label, url in state["created_links"])
            content = "".join(state["text"]).strip() + footer
            yield _openai_chunk(completion_id, model, {"content": safe_final_response(content)})
            if state["citations"]:
                source_event = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "sources": _source_payload(state["citations"]),
                }
                yield f"data: {json.dumps(source_event, ensure_ascii=False)}\n\n"
            yield _openai_chunk(completion_id, model, {}, "stop")
            yield "data: [DONE]\n\n"
        except SupervisorUnavailable:
            yield f"data: {json.dumps({'error': {'code': 'supervisor_unavailable', 'message': 'The supervisor is not configured.'}})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # pragma: no cover - defensive stream boundary
            log(f"Streaming supervisor request failed: {exc}")
            yield f"data: {json.dumps({'error': {'code': 'supervisor_error', 'message': str(exc)}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
