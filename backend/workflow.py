"""LangGraph supervisor workflow for the Nahaj agentic system."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agents import RAG_agent, planning_agent, quiz_agent, supervisor_agent
from .guardrails import PROMPT_INJECTION_MESSAGE, has_prompt_injection, safe_final_response
from .supervisor_contract import SupervisorEvent, SupervisorRequest


class WorkflowState(TypedDict, total=False):
    request: dict[str, Any]
    shared_state: dict[str, Any]
    route: str
    pending: dict[str, Any] | None
    resume_answer: dict[str, Any] | None
    events: list[dict[str, Any]]


class _UnavailableRAG:
    """Keep the coordinator usable when optional retrieval dependencies are absent."""

    def retrieve(self, query: str, course_id: Any, top_k: int = 3) -> list[dict[str, Any]]:
        return []

    def generate(self, query: str, context: Sequence[Mapping[str, Any]]) -> str:
        return "Course retrieval is not available in this environment yet."


class LangGraphSupervisor:
    """Route one request through a specialist and pause for user approvals."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        rag: RAG_agent | None = None,
        planner: planning_agent | None = None,
        quiz: quiz_agent | None = None,
        checkpointer: Any | None = None,
    ):
        self.llm = llm
        if rag is not None:
            self.rag = rag
        else:
            try:
                self.rag = RAG_agent(llm=llm)
            except Exception:
                self.rag = _UnavailableRAG()
        self.quiz = quiz or quiz_agent(llm=llm)
        self.planner = planner or planning_agent(llm=llm, quiz_agent=self.quiz)
        try:
            self.router = supervisor_agent(llm) if llm is not None else None
        except Exception:
            self.router = None
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = self._build_graph()
        # LangGraph stores the pending payload in the checkpoint. This set is
        # only a small boundary marker so the next API request uses Command
        # (resume=...) instead of starting a new graph run.
        self._paused_threads: set[str] = set()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("planning", self._planning_node)
        builder.add_node("quiz", self._quiz_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("progress", self._progress_node)
        builder.add_node("document", self._document_node)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state.get("route", "rag"),
            {
                "rag": "rag",
                "planning": "planning",
                "quiz": "quiz",
                "progress": "progress",
                "document": "document",
            },
        )
        builder.add_edge("rag", END)
        builder.add_conditional_edges(
            "planning",
            lambda state: "approval" if state.get("pending") else END,
            {"approval": "approval", END: END},
        )
        builder.add_conditional_edges(
            "quiz",
            lambda state: "approval" if state.get("pending") else END,
            {"approval": "approval", END: END},
        )
        builder.add_conditional_edges(
            "approval",
            lambda state: str((state.get("pending") or {}).get("next") or END),
            {"planning": "planning", "quiz": "quiz", END: END},
        )
        for node in ("progress", "document"):
            builder.add_edge(node, END)
        return builder.compile(checkpointer=self.checkpointer)

    async def handle(self, request: SupervisorRequest):
        """Implement the API's async supervisor gateway contract."""

        if request.type == "chat" and has_prompt_injection(self._latest_user(request.messages)):
            yield SupervisorEvent("text_delta", {"text": PROMPT_INJECTION_MESSAGE})
            return

        context = dict(request.context or {})
        thread_id = str(context.get("thread_id") or "nahaj-default")
        shared_state = self._state_dict(context.get("shared_state"))
        context["shared_state"] = dict(shared_state)
        request_data = self._request_data(request)
        request_data["context"]["shared_state"] = dict(shared_state)
        config = {"configurable": {"thread_id": thread_id}}
        if thread_id in self._paused_threads:
            resume = {
                "text": self._latest_user(request.messages),
                "context": context,
                "shared_state": dict(shared_state),
            }
            graph_input: Any = Command(resume=resume)
        else:
            graph_input = {
                "request": request_data,
                "shared_state": dict(shared_state),
                "pending": None,
                "resume_answer": None,
                "events": [],
            }

        result = await asyncio.to_thread(self.graph.invoke, graph_input, config)
        for event in result.get("events", []) if isinstance(result, Mapping) else []:
            yield self._validated_event(event)

        interrupts = result.get("__interrupt__", ()) if isinstance(result, Mapping) else ()
        if interrupts:
            for item in interrupts:
                value = getattr(item, "value", item)
                value = dict(value) if isinstance(value, Mapping) else {"message": str(value)}
                self._paused_threads.add(thread_id)
                yield SupervisorEvent("status", {"status": "approval_required", **value})
                if value.get("message"):
                    yield SupervisorEvent("text_delta", {"text": safe_final_response(value["message"])})
        else:
            self._paused_threads.discard(thread_id)

    def _supervisor_node(self, state: WorkflowState) -> dict[str, str]:
        request = state.get("request", {})
        request_type = str(request.get("type") or "chat")
        if request_type == "progress_check":
            return {"route": "progress"}
        if request_type in {"document_ingest", "document_remove"}:
            return {"route": "document"}

        query = self._latest_user(request.get("messages") or [])
        if self.router is not None:
            try:
                selected = self.router.supervise(query).get("agent", "")
                if selected in {"RAG_agent", "planning_agent", "quiz_agent"}:
                    return {"route": selected.removesuffix("_agent")}
            except Exception:
                pass
        return {"route": self._keyword_route(query)}

    def _rag_node(self, state: WorkflowState) -> dict[str, Any]:
        request = state.get("request", {})
        context = request.get("context") or {}
        query = self._latest_user(request.get("messages") or [])
        course_id = context.get("course_id") or self._first_course_id(state.get("shared_state") or {})
        if not course_id:
            return {
                "events": [self._text_event("Select a course so I can search its material.")],
                "pending": None,
                "resume_answer": None,
            }
        try:
            chunks = self.rag.retrieve(query, course_id, top_k=3)
            answer = self.rag.generate(query, chunks)
        except Exception as exc:
            answer = f"I could not search the course material: {exc}"
            chunks = []
        events = [self._text_event(answer)]
        for chunk in chunks:
            events.append(
                self._event_dict(
                    "citation",
                    {
                        "document_id": chunk.get("document_id"),
                        "label": chunk.get("filename", "Course material"),
                        "page": chunk.get("page"),
                        "slide": chunk.get("slide"),
                        "excerpt": str(chunk.get("text", ""))[:500],
                    },
                )
            )
        return {"events": events, "pending": None, "resume_answer": None}

    def _approval_node(self, state: WorkflowState) -> dict[str, Any]:
        """Pause at one stable graph node and return the user's answer on resume."""

        pending = state.get("pending") or {}
        answer = interrupt(dict(pending))
        if isinstance(answer, Mapping):
            return {"resume_answer": dict(answer)}
        return {"resume_answer": {"text": str(answer)}}

    def _planning_node(self, state: WorkflowState) -> dict[str, Any]:
        shared_state = state.get("shared_state") or {}
        pending = state.get("pending") or {}
        answer = state.get("resume_answer")
        if isinstance(answer, Mapping) and str(pending.get("action") or ""):
            shared_state = answer.get("shared_state") if isinstance(answer.get("shared_state"), Mapping) else shared_state
            working_state = dict(state)
            working_state["shared_state"] = dict(shared_state)
            action = str(pending.get("action") or "")
            text = str(answer.get("text") or "")
            if action == "calendar_approval":
                plan = pending.get("plan") or {}
                if self._affirmative(text):
                    return self._calendar_result(plan, working_state)
                if self._negative(text):
                    return {
                        "events": [self._text_event("Okay, I kept the plan as a draft.")],
                        "shared_state": dict(shared_state),
                        "pending": None,
                        "resume_answer": None,
                    }
                return self._keep_pending(pending, shared_state)
            if action == "quiz_approval":
                tool_request = pending.get("request") or {}
                if self._affirmative(text):
                    result = self.planner.run_approved_quick_quiz(tool_request, approved=True, state=shared_state)
                    quiz = result.get("quiz") if isinstance(result, Mapping) else None
                    if isinstance(quiz, Mapping) and quiz.get("status") == "ready":
                        return self._quiz_result(quiz, working_state)
                    # The planner receives a high-level shared-state view, so
                    # fetch the approved quiz's source chunks through RAG when
                    # that view does not contain document text.
                    payload = dict(tool_request.get("payload") or {})
                    payload.setdefault("course_id", self._first_course_id(shared_state))
                    payload.setdefault("course", self._course_name(shared_state, payload.get("course_id")))
                    fallback = self._generate_quiz(payload, working_state)
                    if any(
                        isinstance(event, Mapping)
                        and event.get("kind") == "quiz_create"
                        for event in fallback.get("events", [])
                    ):
                        return fallback
                    return {
                        "events": [self._text_event(str((quiz or result).get("message", "The quick quiz could not be prepared.")))],
                        "shared_state": dict(shared_state),
                        "pending": None,
                        "resume_answer": None,
                    }
                if self._negative(text):
                    result = self.planner.prepare_plan(shared_state, request_quick_quiz=False)
                    return self._plan_result(result, working_state)
                return self._keep_pending(pending, shared_state)
            if action == "plan_input":
                result = self.planner.prepare_plan(
                    shared_state,
                    answers=self._plan_answers(text) or None,
                    request_quick_quiz=True,
                )
                return self._plan_result(result, working_state)

        result = self.planner.prepare_plan(shared_state, request_quick_quiz=True)
        return self._plan_result(result, state)

    def _plan_result(self, result: Mapping[str, Any], state: WorkflowState) -> dict[str, Any]:
        shared_state = state.get("shared_state") or {}
        status = result.get("status")
        if status == "needs_user_input":
            messages = " ".join(str(item.get("message", "")) for item in result.get("questions", []) if isinstance(item, Mapping)).strip()
            questions = [dict(item) for item in result.get("questions", []) if isinstance(item, Mapping)]
            return {
                "events": [],
                "shared_state": dict(shared_state),
                "pending": {
                    "action": "plan_input",
                    "next": "planning",
                    "message": messages or "What information should I use for the plan?",
                    "questions": questions,
                },
                "resume_answer": None,
            }
        if status == "needs_quiz_approval":
            request = result.get("tool_request") or {}
            message = str(request.get("message") or "May I give you a quick quiz before finalizing the plan?")
            return {
                "events": [],
                "shared_state": dict(shared_state),
                "pending": {
                    "action": "quiz_approval",
                    "next": "planning",
                    "message": message,
                    "request": dict(request),
                },
                "resume_answer": None,
            }
        if status == "draft":
            message = self._plan_message(result)
            return {
                "events": [],
                "shared_state": dict(shared_state),
                "pending": {
                    "action": "calendar_approval",
                    "next": "planning",
                    "message": message,
                    "plan": dict(result),
                },
                "resume_answer": None,
            }
        return {
            "events": [self._text_event(str(result.get("message") or "I could not create a plan."))],
            "shared_state": dict(shared_state),
            "pending": None,
            "resume_answer": None,
        }

    def _calendar_result(self, plan: Mapping[str, Any], state: WorkflowState) -> dict[str, Any]:
        result = self.planner.add_to_calendar(plan, approved=True)
        events = self._calendar_events(result.get("event_day_pairs", []), state.get("shared_state") or {})
        return {
            "events": events + [self._text_event(f"Added {len(events)} plan items to your calendar.")],
            "shared_state": dict(state.get("shared_state") or {}),
            "pending": None,
            "resume_answer": None,
        }

    def _quiz_node(self, state: WorkflowState) -> dict[str, Any]:
        pending = state.get("pending") or {}
        answer = state.get("resume_answer")
        shared_state = state.get("shared_state") or {}
        if isinstance(answer, Mapping) and str(pending.get("action") or "") == "quiz_approval":
            shared_state = answer.get("shared_state") if isinstance(answer.get("shared_state"), Mapping) else shared_state
            working_state = dict(state)
            working_state["shared_state"] = dict(shared_state)
            payload = pending.get("payload") if isinstance(pending.get("payload"), Mapping) else {}
            text = str(answer.get("text") or "")
            if self._affirmative(text):
                return self._generate_quiz(payload, working_state)
            if self._negative(text):
                return {
                    "events": [self._text_event("Okay, I will not create a quiz yet.")],
                    "shared_state": dict(shared_state),
                    "pending": None,
                    "resume_answer": None,
                }
            return self._keep_pending(pending, shared_state)

        request = state.get("request", {})
        context = request.get("context") or {}
        payload = {
            "query": self._latest_user(request.get("messages") or []),
            "course_id": context.get("course_id"),
            "course": context.get("course"),
            "count": 3,
        }
        return {
            "events": [],
            "pending": {
                "action": "quiz_approval",
                "next": "quiz",
                "message": "I can prepare a quiz from your course material. May I create it?",
                "payload": payload,
            },
            "resume_answer": None,
        }

    def _generate_quiz(self, payload: Mapping[str, Any], state: WorkflowState) -> dict[str, Any]:
        request = state.get("request", {})
        context = request.get("context") or {}
        shared_state = state.get("shared_state") or {}
        course_id = payload.get("course_id") or context.get("course_id") or self._first_course_id(shared_state)
        topics = payload.get("topics")
        if isinstance(topics, str):
            topics = [topics]
        query = str(payload.get("query") or " ".join(str(item) for item in (topics or [])) or "key course concepts")
        chunks = []
        if course_id:
            try:
                chunks = self.rag.retrieve(query, course_id, top_k=max(3, int(payload.get("count", 3)) * 2))
            except Exception:
                chunks = []
        course_name = str(payload.get("course") or self._course_name(shared_state, course_id) or "").strip() or None
        material = [
            {
                "title": chunk.get("section") or chunk.get("filename") or "Course material",
                "course": course_name,
                "content": chunk.get("text", ""),
            }
            for chunk in chunks
            if chunk.get("text")
        ]
        result = self.quiz.generate_quiz(
            state=shared_state,
            material=material or None,
            course=course_name,
            count=payload.get("count", 3),
            topics=topics,
        )
        if result.get("status") != "ready":
            return {
                "events": [self._text_event(str(result.get("message", "I need course material before I can make a quiz.")))],
                "shared_state": dict(shared_state),
                "pending": None,
                "resume_answer": None,
            }
        return self._quiz_result(result, state, course_id=course_id, course_name=course_name)

    def _quiz_result(
        self,
        quiz: Mapping[str, Any],
        state: WorkflowState,
        course_id: str | None = None,
        course_name: str | None = None,
    ) -> dict[str, Any]:
        request = state.get("request", {})
        context = request.get("context") or {}
        course_id = course_id or context.get("course_id") or self._first_course_id(state.get("shared_state") or {})
        course_name = course_name or self._course_name(state.get("shared_state") or {}, course_id)
        answer_key = {str(item.get("id")): item for item in quiz.get("answer_key", []) if isinstance(item, Mapping)}
        questions = []
        for question in quiz.get("questions", []):
            if not isinstance(question, Mapping):
                continue
            options = []
            for index, option in enumerate(question.get("options") or []):
                if isinstance(option, Mapping):
                    option_id = str(option.get("option_id") or self._option_id(index))
                    text = str(option.get("text") or "")
                else:
                    option_id = self._option_id(index)
                    text = str(option)
                options.append({"option_id": option_id, "text": text})
            answer = answer_key.get(str(question.get("id")), {})
            questions.append(
                {
                    "prompt": question.get("prompt", ""),
                    "topic": question.get("topic"),
                    "explanation": answer.get("explanation", ""),
                    "correct_option_id": answer.get("correct_option_id", "A"),
                    "options": options,
                }
            )
        event = {
            "title": f"{course_name + ' ' if course_name else ''}practice quiz",
            "course_id": course_id,
            "source_collections": ["slides"],
            "questions": questions,
        }
        return {
            "events": [
                self._event_dict("quiz_create", event),
                self._text_event(f"Created a quiz with {len(questions)} questions."),
            ],
            "shared_state": dict(state.get("shared_state") or {}),
            "pending": None,
            "resume_answer": None,
        }

    def _progress_node(self, state: WorkflowState) -> dict[str, Any]:
        request = state.get("request", {})
        snapshot = request.get("snapshot") or {}
        tasks = [item for item in snapshot.get("tasks", []) if isinstance(item, Mapping)]
        completed = sum(1 for item in tasks if item.get("status") == "done")
        total = len(tasks)
        message = f"Progress check: {completed} of {total} tasks completed."
        events = [self._text_event(message)]
        overdue = [item for item in tasks if item.get("status") == "todo" and self._past(item.get("end_at"))]
        if overdue:
            events.append(
                self._event_dict(
                    "notification_create",
                    {
                        "title": "Overdue study tasks",
                        "message": f"You have {len(overdue)} overdue study task(s).",
                        "severity": "warning",
                        "dedupe_key": f"overdue-{date.today().isoformat()}",
                    },
                )
            )
        return {"events": events, "pending": None, "resume_answer": None}

    def _document_node(self, state: WorkflowState) -> dict[str, Any]:
        request = state.get("request", {})
        document = request.get("document") or {}
        action = str(request.get("type") or "")
        if action == "document_ingest":
            message = f"Recorded {document.get('filename', 'the document')} in the course library."
            events = [self._text_event(message), self._event_dict("document_status", {"document_id": document.get("id"), "status": "ready"})]
        else:
            events = [self._text_event(f"Removed {document.get('filename', 'the document')} from the course library.")]
        return {"events": events, "pending": None, "resume_answer": None}

    @staticmethod
    def _keep_pending(pending: Mapping[str, Any], shared_state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "events": [],
            "shared_state": dict(shared_state),
            "pending": dict(pending),
            "resume_answer": None,
        }

    @staticmethod
    def _request_data(request: SupervisorRequest) -> dict[str, Any]:
        return {
            "type": request.type,
            "request_id": request.request_id,
            "messages": list(request.messages or []),
            "context": dict(request.context or {}),
            "document": dict(request.document or {}) if request.document else None,
            "snapshot": dict(request.snapshot or {}) if request.snapshot else None,
        }

    @staticmethod
    def _state_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if value is None:
            return {}
        return {
            key: getattr(value, key)
            for key in ("semester", "courses", "course_id", "nahaj_context", "retrieved_context")
            if hasattr(value, key)
        }

    @staticmethod
    def _latest_user(messages: Sequence[Any]) -> str:
        for message in reversed(list(messages or [])):
            if isinstance(message, Mapping):
                if str(message.get("role", "user")) != "user":
                    continue
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "")
            if isinstance(content, list):
                content = " ".join(str(item.get("text", item)) if isinstance(item, Mapping) else str(item) for item in content)
            if content:
                return str(content).strip()
        return ""

    @staticmethod
    def _keyword_route(query: str) -> str:
        text = query.casefold()
        if any(word in text for word in ("plan", "schedule", "calendar", "study plan", "organize")):
            return "planning"
        if any(word in text for word in ("quiz", "test me", "practice question", "mock exam")):
            return "quiz"
        return "rag"

    @staticmethod
    def _affirmative(text: str) -> bool:
        value = text.strip().casefold()
        return value in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "approve", "approved", "confirm", "confirmed"} or value.startswith(("yes ", "approve ", "confirm "))

    @staticmethod
    def _negative(text: str) -> bool:
        value = text.strip().casefold()
        return value in {"no", "n", "nope", "not now", "cancel", "decline", "declined"} or value.startswith(("no ", "don't ", "do not ", "cancel "))

    @staticmethod
    def _plan_answers(text: str) -> dict[str, Any]:
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if len(dates) >= 2:
            return {"semester_dates": {"start_date": dates[0], "end_date": dates[1]}}
        return {}

    @staticmethod
    def _plan_message(plan: Mapping[str, Any]) -> str:
        summary = plan.get("summary") or {}
        lines = [
            f"I drafted a plan for {summary.get('courses_planned', 0)} course(s) with {summary.get('total_tasks', 0)} study task(s).",
        ]
        for item in (plan.get("event_day_pairs") or [])[:5]:
            if isinstance(item, Mapping):
                event = item.get("event") or {}
                lines.append(f"- {event.get('title', 'Study task')} on {item.get('day')}")
        lines.append("May I add these events to your calendar?")
        return "\n".join(lines)

    @staticmethod
    def _event_dict(kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return {"kind": kind, "data": dict(data)}

    @classmethod
    def _text_event(cls, text: str) -> dict[str, Any]:
        return cls._event_dict("text_delta", {"text": text})

    @staticmethod
    def _event(event: Any) -> SupervisorEvent:
        if isinstance(event, SupervisorEvent):
            return event
        return SupervisorEvent(kind=str(event.get("kind", "")), data=dict(event.get("data") or {}))

    @classmethod
    def _validated_event(cls, event: Any) -> SupervisorEvent:
        normalized = cls._event(event)
        if normalized.kind == "text_delta":
            normalized.data["text"] = safe_final_response(normalized.data.get("text", ""))
        return normalized

    @staticmethod
    def _first_course_id(state: Mapping[str, Any]) -> str | None:
        for course in state.get("courses", []) or []:
            if isinstance(course, Mapping) and course.get("id"):
                return str(course["id"])
        return None

    @staticmethod
    def _course_name(state: Mapping[str, Any], course_id: str | None) -> str | None:
        for course in state.get("courses", []) or []:
            if isinstance(course, Mapping) and (not course_id or str(course.get("id")) == str(course_id)):
                return str(course.get("course_name") or course.get("name") or "").strip() or None
        return None

    @staticmethod
    def _calendar_events(event_day_pairs: Sequence[Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
        course_ids = {
            str(course.get("course_name") or course.get("name")): course.get("id")
            for course in state.get("courses", []) or []
            if isinstance(course, Mapping) and course.get("id")
        }
        existing_ids: dict[tuple[str | None, str, str], str] = {}
        existing_titles: dict[tuple[str | None, str], str] = {}
        for course in state.get("courses", []) or []:
            if not isinstance(course, Mapping):
                continue
            course_id = str(course.get("id")) if course.get("id") else None
            plans = course.get("current_plan") or []
            if isinstance(plans, Mapping):
                plans = [plans]
            for plan in plans:
                if not isinstance(plan, Mapping):
                    continue
                for task in plan.get("tasks") or []:
                    if isinstance(task, Mapping) and task.get("id") and task.get("title") and task.get("date"):
                        existing_ids[(course_id, str(task["title"]), str(task["date"]))] = str(task["id"])
                        existing_titles[(course_id, str(task["title"]))] = str(task["id"])
        events = []
        for item in event_day_pairs or []:
            if not isinstance(item, Mapping):
                continue
            event = item.get("event") or {}
            try:
                day = date.fromisoformat(str(item.get("day")))
            except ValueError:
                continue
            start = datetime.combine(day, time(9), tzinfo=timezone.utc)
            end = datetime.combine(day, time(10), tzinfo=timezone.utc)
            course_id = event.get("course_id") or course_ids.get(str(event.get("course") or ""))
            title = str(event.get("title") or "Study task")
            task_id = existing_ids.get((str(course_id) if course_id else None, title, str(item.get("day"))))
            task_id = task_id or existing_titles.get((str(course_id) if course_id else None, title))
            events.append(
                {
                    "kind": "task_upsert",
                    "data": {
                        "id": task_id,
                        "course_id": course_id,
                        "title": title,
                        "notes": str(event.get("description") or ""),
                        "start_at": start.isoformat(),
                        "end_at": end.isoformat(),
                        "status": "todo",
                        "origin": "supervisor",
                    },
                }
            )
        return events

    @staticmethod
    def _option_id(index: int) -> str:
        return ("A", "B", "C", "D")[index] if 0 <= index < 4 else str(index + 1)

    @staticmethod
    def _past(value: Any) -> bool:
        if not value:
            return False
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed < datetime.now(timezone.utc)


def build_workflow(supervisor: LangGraphSupervisor):
    """Return the compiled coordinator graph for an initialized supervisor."""

    return supervisor.graph


def create_supervisor() -> LangGraphSupervisor:
    """Factory used by ``SUPERVISOR_FACTORY=backend.workflow:create_supervisor``."""

    llm = None
    try:
        from .config import config

        llm = config().llm
    except Exception:
        # The graph remains usable with deterministic routing/fallbacks when
        # OpenRouter credentials are not configured.
        pass
    return LangGraphSupervisor(llm=llm)


create_gateway = create_supervisor
