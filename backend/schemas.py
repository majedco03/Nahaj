from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SemesterInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    start_date: date
    end_date: date
    timezone: str = Field(default="Asia/Riyadh", min_length=1, max_length=80)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class SemesterOut(SemesterInput, APIModel):
    id: str
    active: bool
    created_at: datetime
    updated_at: datetime


class CourseCreate(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    color: str = Field(default="#6366f1", pattern=r"^#[0-9A-Fa-f]{6}$")


class CourseUpdate(BaseModel):
    code: str | None = Field(default=None, min_length=1, max_length=80)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")


class CourseOut(CourseCreate, APIModel):
    id: str
    semester_id: str
    created_at: datetime
    updated_at: datetime


Collection = Literal["slides", "past_exam"]


class DocumentOut(APIModel):
    id: str
    course_id: str
    collection: Collection
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: str
    error: str | None
    created_at: datetime
    updated_at: datetime


class TaskCreate(BaseModel):
    course_id: str | None = None
    title: str = Field(min_length=1, max_length=300)
    notes: str | None = None
    start_at: datetime
    end_at: datetime
    status: Literal["todo", "done", "skipped"] = "todo"
    origin: Literal["student", "supervisor"] = "student"

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be after start_at")
        return self


class TaskUpdate(BaseModel):
    course_id: str | None = None
    title: str | None = Field(default=None, min_length=1, max_length=300)
    notes: str | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    status: Literal["todo", "done", "skipped"] | None = None


class TaskOut(TaskCreate, APIModel):
    id: str
    created_at: datetime
    updated_at: datetime


class PlanOut(APIModel):
    semester: SemesterOut | None
    tasks: list[TaskOut]
    summary: dict[str, Any]


class QuizOptionOut(APIModel):
    option_id: str
    text: str


class QuizQuestionOut(APIModel):
    id: str
    prompt: str
    topic: str | None
    explanation: str | None = None
    options: list[QuizOptionOut]


class QuizListOut(APIModel):
    id: str
    course_id: str | None
    title: str
    source_collections: list[str]
    question_count: int
    created_at: datetime
    latest_score: float | None
    best_score: float | None


class QuizDetailOut(QuizListOut):
    questions: list[QuizQuestionOut]


class AttemptStartOut(APIModel):
    id: str
    quiz_id: str
    status: str
    started_at: datetime
    questions: list[QuizQuestionOut]


class AnswerInput(BaseModel):
    question_id: str
    selected_option_id: str


class AnswerOut(APIModel):
    question_id: str
    selected_option_id: str
    answered_at: datetime


class ReviewQuestionOut(QuizQuestionOut):
    correct_option_id: str
    selected_option_id: str | None
    is_correct: bool


class AttemptOut(APIModel):
    id: str
    quiz_id: str
    status: str
    score_percent: float | None
    weak_topics: list[str]
    started_at: datetime
    completed_at: datetime | None
    questions: list[ReviewQuestionOut] | list[QuizQuestionOut]


class NotificationOut(APIModel):
    id: str
    severity: str
    title: str
    message: str
    target_url: str | None
    read_at: datetime | None
    created_at: datetime


class ProgressOut(BaseModel):
    task_completion_percent: float
    quiz_average_percent: float | None
    courses: list[dict[str, Any]]


class NahajContext(BaseModel):
    course_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    thread_id: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    nahaj_context: NahajContext | None = None
