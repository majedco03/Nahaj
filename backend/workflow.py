"""LangGraph supervisor workflow for the Nahaj agentic system."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agents import RAG_agent, communication_agent, planning_agent, progress_agent, quiz_agent, supervisor_agent
from .guardrails import PROMPT_INJECTION_MESSAGE, has_prompt_injection, safe_final_response
from .logger import log
from .supervisor_contract import SupervisorEvent, SupervisorRequest


class WorkflowState(TypedDict, total=False):
    request: dict[str, Any]
    shared_state: dict[str, Any]
    communication_checked: bool
    understood: bool
    task: str
    route: str
    pending: dict[str, Any] | None
    resume_answer: dict[str, Any] | None
    events: list[dict[str, Any]]


class LangGraphSupervisor:
    """Route one request through a specialist and pause for user approvals."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        rag: RAG_agent | None = None,
        communicator: communication_agent | None = None,
        planner: planning_agent | None = None,
        progress: progress_agent | None = None,
        quiz: quiz_agent | None = None,
        checkpointer: Any | None = None,
    ):
        self.llm = llm
        self.rag = rag if rag is not None else RAG_agent(llm=llm)
        self.quiz = quiz or quiz_agent(llm=llm)
        self.planner = planner or planning_agent(llm=llm)
        self.progress = progress or progress_agent(llm=llm)
        self.communicator = communicator or communication_agent(llm)
        self.router = supervisor_agent(llm) if llm is not None else None
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = self._build_graph()
        # LangGraph stores the pending payload in the checkpoint. This mapping
        # links a conversation to the unique graph run waiting for its answer.
        self._paused_threads: dict[str, dict[str, Any]] = {}

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("communication", self._communication_node)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("planning", self._planning_node)
        builder.add_node("quiz", self._quiz_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("progress", self._progress_node)
        builder.add_node("document", self._document_node)
        builder.add_edge(START, "communication")
        builder.add_conditional_edges(
            "communication",
            lambda state: "supervisor" if state.get("understood") else END,
            {"supervisor": "supervisor", END: END},
        )
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
        paused = self._paused_threads.get(thread_id) if request.type == "chat" else None
        if paused:
            user_answer = self._latest_user(request.messages)
            decision = self.communicator.communicate(
                user_answer,
                pending=paused.get("pending") if isinstance(paused.get("pending"), Mapping) else None,
                history=self._recent_turns(request.messages),
            )
            log(
                f"Communication decision request_id={request.request_id} thread_id={thread_id} "
                f"understood={bool(decision.get('understood'))} "
                f"answers_pending={bool(decision.get('answers_pending'))} "
                f"source={decision.get('source', 'unknown')}"
            )
            if not decision.get("understood"):
                if not str(decision.get("response") or "").strip():
                    raise RuntimeError("The communication model returned an empty response.")
                yield SupervisorEvent(
                    "text_delta",
                    {"text": safe_final_response(decision["response"])},
                )
                return
            if decision.get("answers_pending"):
                graph_thread_id = str(paused["graph_thread_id"])
                graph_input: Any = Command(
                    resume={
                        # Approval matching must use the student's exact answer. The
                        # communication model may rewrite "yes" into a task sentence.
                        "text": user_answer,
                        "context": context,
                        "shared_state": dict(shared_state),
                    }
                )
            else:
                self._paused_threads.pop(thread_id, None)
                graph_thread_id = f"{thread_id}:{request.request_id}"
                graph_input = {
                    "request": request_data,
                    "shared_state": dict(shared_state),
                    "communication_checked": True,
                    "understood": True,
                    "task": str(decision.get("task") or self._latest_user(request.messages)),
                    "pending": None,
                    "resume_answer": None,
                    "events": [],
                }
        else:
            graph_thread_id = f"{thread_id}:{request.request_id}"
            graph_input = {
                "request": request_data,
                "shared_state": dict(shared_state),
                "communication_checked": False,
                "understood": False,
                "task": "",
                "pending": None,
                "resume_answer": None,
                "events": [],
            }
        config = {"configurable": {"thread_id": graph_thread_id}}

        result = await asyncio.to_thread(self.graph.invoke, graph_input, config)
        for event in result.get("events", []) if isinstance(result, Mapping) else []:
            yield self._validated_event(event)

        interrupts = result.get("__interrupt__", ()) if isinstance(result, Mapping) else ()
        if interrupts:
            for item in interrupts:
                value = getattr(item, "value", item)
                value = dict(value) if isinstance(value, Mapping) else {"message": str(value)}
                self._paused_threads[thread_id] = {
                    "graph_thread_id": graph_thread_id,
                    "pending": value,
                }
                yield SupervisorEvent("status", {"status": "approval_required", **value})
                if value.get("message"):
                    yield SupervisorEvent("text_delta", {"text": safe_final_response(value["message"])})
        else:
            current = self._paused_threads.get(thread_id)
            if current and current.get("graph_thread_id") == graph_thread_id:
                self._paused_threads.pop(thread_id, None)

    def _communication_node(self, state: WorkflowState) -> dict[str, Any]:
        if state.get("communication_checked"):
            return {"understood": True}
        request = state.get("request", {})
        if request.get("type") != "chat":
            return {"communication_checked": True, "understood": True}
        messages = request.get("messages") or []
        query = self._latest_user(messages)
        decision = self.communicator.communicate(query, history=self._recent_turns(messages))
        understood = bool(decision.get("understood"))
        log(
            f"Communication decision request_id={request.get('request_id', 'unknown')} "
            f"thread_id={(request.get('context') or {}).get('thread_id', 'nahaj-default')} "
            f"understood={understood} answers_pending=False "
            f"source={decision.get('source', 'unknown')}"
        )
        if not understood:
            response = str(decision.get("response") or "").strip()
            if not response:
                raise RuntimeError("The communication model returned an empty response.")
            return {
                "understood": False,
                "task": "",
                "events": [self._text_event(response)],
                "pending": None,
                "resume_answer": None,
            }
        return {
            "communication_checked": True,
            "understood": True,
            "task": str(decision.get("task") or query),
        }

    def _supervisor_node(self, state: WorkflowState) -> dict[str, str]:
        request = state.get("request", {})
        request_type = str(request.get("type") or "chat")
        if request_type == "progress_check":
            return {"route": "progress"}
        if request_type in {"document_ingest", "document_remove"}:
            return {"route": "document"}

        messages = request.get("messages") or []
        original = self._latest_user(messages)
        query = str(state.get("task") or original)
        if self.router is None:
            raise RuntimeError("OpenRouter LLM is required for coordinator routing.")
        # The rewritten task can lose the detail that decides the route, so the
        # coordinator sees the student's own words and the turns before them.
        decision = self.router.supervise(query, original=original, history=self._recent_turns(messages))
        selected = decision.get("agent", "")
        if selected not in {"RAG_agent", "planning_agent", "quiz_agent"}:
            raise RuntimeError(f"The coordinator returned an invalid agent: {selected!r}")
        route = selected.removesuffix("_agent").lower()
        log(
            f"Coordinator decision request_id={request.get('request_id', 'unknown')} "
            f"route={route} source=model reason={str(decision.get('reason') or '').strip()!r}"
        )
        return {"route": route}

    def _rag_node(self, state: WorkflowState) -> dict[str, Any]:
        request = state.get("request", {})
        context = request.get("context") or {}
        query = str(state.get("task") or self._latest_user(request.get("messages") or []))
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
                # Anything else is the student telling us how to change the draft.
                revised = self.planner.prepare_plan(shared_state, request=text)
                if self._same_draft(plan, revised):
                    # Re-presenting an unchanged draft reads as the agent ignoring
                    # the student, so the result says the change could not be made.
                    revised = dict(revised)
                    revised["unchanged"] = True
                return self._plan_result(revised, working_state)
            if action == "plan_input":
                result = self.planner.prepare_plan(
                    shared_state,
                    answers=self._plan_answers(text) or None,
                    request=text,
                )
                return self._plan_result(result, working_state)

        result = self.planner.prepare_plan(shared_state, request=self._task_text(state))
        return self._plan_result(result, state)

    def _plan_result(self, result: Mapping[str, Any], state: WorkflowState) -> dict[str, Any]:
        shared_state = state.get("shared_state") or {}
        status = result.get("status")
        if status == "needs_user_input":
            questions = [dict(item) for item in result.get("questions", []) if isinstance(item, Mapping)]
            message = self._planning_response(
                result,
                state,
                "Ask only for the missing information listed in the result.",
            )
            return {
                "events": [],
                "shared_state": dict(shared_state),
                "pending": {
                    "action": "plan_input",
                    "next": "planning",
                    "message": message,
                    "questions": questions,
                },
                "resume_answer": None,
            }
        if status == "draft":
            instruction = (
                "Summarize the draft plan with its dates and ask for explicit approval "
                "before adding it to the calendar."
            )
            if result.get("unchanged"):
                instruction = (
                    "The requested change produced the same plan. Say plainly that it could "
                    "not be applied, name the constraint that prevented it using design and "
                    "summary, such as the finish date, the daily session limit, or a deadline, "
                    "and offer a concrete alternative such as dropping material or allowing "
                    "more sessions per day. Do not present this as a new plan."
                )
            elif str(result.get("design", {}).get("horizon_met")) == "False":
                instruction = (
                    "Summarize the draft plan with its dates. Say clearly that it does not fit "
                    "the finish date the student asked for, give the date it actually ends, and "
                    "ask whether to approve it or change the scope. "
                )
            message = self._planning_response(result, state, instruction)
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
            "events": [self._text_event(self._planning_response(result, state, "Explain the planning result clearly."))],
            "shared_state": dict(shared_state),
            "pending": None,
            "resume_answer": None,
        }

    @staticmethod
    def _same_draft(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        """Whether a revision produced exactly the plan the student just rejected."""

        def signature(plan: Mapping[str, Any]) -> list[tuple[str, str]]:
            pairs = []
            for item in plan.get("event_day_pairs") or []:
                if isinstance(item, Mapping):
                    event = item.get("event") or {}
                    pairs.append((str(event.get("title") or ""), str(item.get("day") or "")))
            return sorted(pairs)

        before = signature(previous)
        return bool(before) and before == signature(current)

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
        """Create the requested quiz. Quiz creation is not an approval step."""

        request = state.get("request", {})
        context = request.get("context") or {}
        messages = request.get("messages") or []
        query = str(state.get("task") or self._latest_user(messages))
        scope = self._quiz_scope(query, messages, state.get("shared_state") or {})
        payload = {
            "query": query,
            "course_id": context.get("course_id"),
            "course": scope.get("course") or context.get("course"),
            "count": scope.get("count") or self._requested_count(query),
            "topics": scope.get("topics") or None,
        }
        return self._generate_quiz(payload, state)

    def _quiz_material(
        self,
        payload: Mapping[str, Any],
        state: WorkflowState,
    ) -> tuple[str | None, str | None, list[dict[str, Any]]]:
        """Find the course and the source chunks a quiz should be built from."""

        request = state.get("request", {})
        context = request.get("context") or {}
        shared_state = state.get("shared_state") or {}
        topics = payload.get("topics")
        if isinstance(topics, str):
            topics = [topics]
        query = str(payload.get("query") or " ".join(str(item) for item in (topics or [])) or "key course concepts")
        course_id = self._resolve_course_id(
            query,
            shared_state,
            payload.get("course_id") or context.get("course_id"),
        )
        course_id, chunks = self._course_material(
            query,
            course_id,
            shared_state,
            max(3, int(payload.get("count", 3)) * 2),
        )
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
        return course_id, course_name, material

    def _generate_quiz(self, payload: Mapping[str, Any], state: WorkflowState) -> dict[str, Any]:
        shared_state = state.get("shared_state") or {}
        topics = payload.get("topics")
        if isinstance(topics, str):
            topics = [topics]
        course_id, course_name, material = self._quiz_material(payload, state)
        result = self.quiz.generate_quiz(
            state=shared_state,
            material=material or None,
            course=course_name,
            count=payload.get("count", 3),
            topics=topics,
        )
        if result.get("status") != "ready":
            return {
                "events": [self._text_event(self._material_problem(shared_state, result, bool(material)))],
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
        questions = self._normalised_questions(quiz)
        event = {
            "title": f"{course_name + ' ' if course_name else ''}practice quiz",
            "course_id": course_id,
            "source_collections": ["slides"],
            "questions": questions,
        }
        return {
            "events": [
                self._event_dict("quiz_create", event),
                self._text_event(self._quiz_message(quiz, state, course_name, len(questions))),
                *(
                    [
                        self._text_event(
                            f"Your course material only supported {len(questions)} of the "
                            f"{quiz.get('requested_count')} questions you asked for."
                        )
                    ]
                    if quiz.get("short_of_request")
                    else []
                ),
            ],
            "shared_state": dict(state.get("shared_state") or {}),
            "pending": None,
            "resume_answer": None,
        }

    @classmethod
    def _normalised_questions(cls, quiz: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Pair every question with its option ids and its answer."""

        answer_key = {str(item.get("id")): item for item in quiz.get("answer_key", []) if isinstance(item, Mapping)}
        questions: list[dict[str, Any]] = []
        for question in quiz.get("questions", []):
            if not isinstance(question, Mapping):
                continue
            options = []
            for index, option in enumerate(question.get("options") or []):
                if isinstance(option, Mapping):
                    option_id = str(option.get("option_id") or cls._option_id(index))
                    text = str(option.get("text") or "")
                else:
                    option_id = cls._option_id(index)
                    text = str(option)
                options.append({"option_id": option_id, "text": text})
            answer = answer_key.get(str(question.get("id")), {})
            questions.append(
                {
                    "prompt": question.get("prompt", ""),
                    "topic": question.get("topic"),
                    "explanation": answer.get("explanation", ""),
                    "correct_option_id": str(answer.get("correct_option_id", "A")).upper(),
                    "options": options,
                }
            )
        return questions

    @staticmethod
    def _library(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every course the student has, with the material recorded against it."""

        library: list[dict[str, Any]] = []
        for course in state.get("courses", []) or []:
            if not isinstance(course, Mapping) or not course.get("id"):
                continue
            library.append(
                {
                    "id": str(course["id"]),
                    "name": str(course.get("course_name") or course.get("name") or "").strip(),
                    "code": str(course.get("code") or "").strip(),
                    "materials": [item for item in course.get("materials") or [] if isinstance(item, Mapping)],
                }
            )
        return library

    @classmethod
    def _resolve_course_id(cls, query: str, state: Mapping[str, Any], explicit: Any = None) -> str | None:
        """Pick the course the student means rather than the first one on file."""

        library = cls._library(state)
        if not library:
            return str(explicit) if explicit else None
        if explicit and any(course["id"] == str(explicit) for course in library):
            return str(explicit)
        text = query.casefold()

        def score(course: Mapping[str, Any]) -> int:
            value = 0
            for label in (course["name"], course["code"]):
                if label and label.casefold() in text:
                    value += 10
            for material in course["materials"]:
                title = str(material.get("title") or "").strip().casefold()
                if not title:
                    continue
                if title in text:
                    value += 5
                value += sum(1 for word in set(re.findall(r"\w{4,}", title)) if word in text)
            return value

        best = max(library, key=lambda course: (score(course), len(course["materials"])))
        if score(best) == 0:
            # Nothing in the request names a course, so prefer one that has material.
            best = next((course for course in library if course["materials"]), library[0])
        return best["id"]

    def _course_material(
        self,
        query: str,
        course_id: str | None,
        state: Mapping[str, Any],
        top_k: int,
    ) -> tuple[str | None, list[Mapping[str, Any]]]:
        """Read the chosen course's material, then any other course that has some."""

        order = [course_id] + [
            course["id"] for course in self._library(state) if course["id"] != course_id
        ]
        for candidate in order:
            if not candidate:
                continue
            try:
                chunks = self.rag.retrieve(query, candidate, top_k=top_k)
            except Exception as exc:
                # A retrieval failure is not the same as an empty library, so it is
                # recorded instead of being reported to the student as missing material.
                log(f"Quiz retrieval failed course_id={candidate} error={exc}")
                continue
            usable = [
                chunk
                for chunk in chunks or []
                if isinstance(chunk, Mapping) and str(chunk.get("text") or "").strip()
            ]
            if usable:
                return candidate, usable
            log(f"Quiz retrieval returned no usable chunks course_id={candidate}")
        return course_id, []

    @classmethod
    def _material_problem(
        cls,
        state: Mapping[str, Any],
        result: Mapping[str, Any],
        retrieved: bool,
    ) -> str:
        """Say what is actually wrong instead of always asking for material."""

        documents = [material for course in cls._library(state) for material in course["materials"]]
        if not documents:
            return (
                "I could not find any course material to build a quiz from. "
                "Upload your slides for the course and I will create one."
            )
        unready = sorted(
            {
                str(material.get("title") or "").strip()
                for material in documents
                if str(material.get("status") or "ready").casefold() != "ready"
            }
            - {""}
        )
        if unready:
            return (
                "Your course material is still being prepared ("
                + ", ".join(unready[:3])
                + "). Ask me again shortly and I will build the quiz."
            )
        if not retrieved:
            titles = sorted({str(material.get("title") or "").strip() for material in documents} - {""})
            return (
                "I have your course material on file ("
                + ", ".join(titles[:3])
                + ") but could not read its search index, so I cannot build a quiz from it yet. "
                "Re-uploading the file will rebuild the index."
            )
        return str(
            result.get("message")
            or "I could not read your course material well enough to build a quiz."
        )

    def _quiz_message(
        self,
        quiz: Mapping[str, Any],
        state: WorkflowState,
        course_name: str | None,
        count: int,
    ) -> str:
        """Let the quiz specialist hand over the finished quiz."""

        request = state.get("request", {})
        query = str(state.get("task") or self._latest_user(request.get("messages") or []))
        try:
            return self.quiz.describe_quiz(query, quiz, course_name)
        except Exception:
            # The quiz itself is already built and carried by the quiz_create
            # event, so a failed hand-over message must not discard it.
            return f"Created a quiz with {count} questions."

    def _quiz_scope(self, query: str, messages: Sequence[Any], shared_state: Mapping[str, Any]) -> dict[str, Any]:
        """Let the quiz specialist read its own scope out of the conversation."""

        courses = [course["name"] for course in self._library(shared_state) if course["name"]]
        try:
            scope = self.quiz.scope(query, history=self._recent_turns(messages), courses=courses)
        except Exception as exc:
            log(f"Quiz scope failed error={exc}")
            return {}
        return scope if isinstance(scope, Mapping) else {}

    @staticmethod
    def _requested_count(query: str) -> int:
        """Read a question count out of the request, defaulting to three."""

        match = re.search(r"\b(\d{1,2})\s*(?:questions?|q)\b", query, re.IGNORECASE)
        if not match:
            return 3
        return max(1, min(int(match.group(1)), 20))

    def _progress_node(self, state: WorkflowState) -> dict[str, Any]:
        request = state.get("request", {})
        snapshot = request.get("snapshot") or {}
        tasks = [item for item in snapshot.get("tasks", []) if isinstance(item, Mapping)]
        completed = sum(1 for item in tasks if item.get("status") == "done")
        total = len(tasks)
        review = {
            "total_tasks": total,
            "completed_tasks": completed,
            "incomplete_tasks": total - completed,
            "tasks": tasks,
        }
        message = self.progress.review(review)
        events = [self._text_event(message)]
        overdue = [item for item in tasks if item.get("status") == "todo" and self._past(item.get("end_at"))]
        if overdue:
            events.append(
                self._event_dict(
                    "notification_create",
                    {
                        "title": "Overdue study tasks",
                        "message": message,
                        "severity": "warning",
                        "dedupe_key": f"overdue-{date.today().isoformat()}",
                    },
                )
            )
        return {"events": events, "pending": None, "resume_answer": None}

    def _task_text(self, state: WorkflowState) -> str:
        """What the student asked for, as the planner should receive it."""

        request = state.get("request", {})
        return str(state.get("task") or self._latest_user(request.get("messages") or []))

    def _planning_response(
        self,
        result: Mapping[str, Any],
        state: WorkflowState,
        instruction: str,
    ) -> str:
        return self.planner.respond(self._task_text(state), result, instruction)

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

    @classmethod
    def _recent_turns(cls, messages: Sequence[Any], limit: int = 6) -> list[dict[str, str]]:
        """The turns before the current one, so a short follow-up keeps its context."""

        turns: list[dict[str, str]] = []
        for message in list(messages or [])[:-1][-limit:]:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user")
                content = message.get("content", "")
            else:
                role = str(getattr(message, "type", "user"))
                content = getattr(message, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", item)) if isinstance(item, Mapping) else str(item) for item in content
                )
            text = str(content).strip()
            if text:
                turns.append({"role": role, "content": text[:300]})
        return turns

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
    def _affirmative(text: str) -> bool:
        value = text.strip().casefold()
        return value in {
            "yes",
            "y",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "approve",
            "approved",
            "confirm",
            "confirmed",
            "go ahead",
            "add them",
            "add it",
            "add to calendar",
        } or value.startswith(("yes ", "approve ", "confirm "))

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
        # A plan can place several sessions on one day, so each one takes the next
        # hour instead of every task starting at nine.
        used_hours: dict[date, int] = {}
        for item in event_day_pairs or []:
            if not isinstance(item, Mapping):
                continue
            event = item.get("event") or {}
            try:
                day = date.fromisoformat(str(item.get("day")))
            except ValueError:
                continue
            hour = min(9 + 2 * used_hours.get(day, 0), 21)
            used_hours[day] = used_hours.get(day, 0) + 1
            start = datetime.combine(day, time(hour), tzinfo=timezone.utc)
            end = datetime.combine(day, time(hour + 1), tzinfo=timezone.utc)
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


def create_supervisor() -> LangGraphSupervisor:
    """Factory used by ``SUPERVISOR_FACTORY=backend.workflow:create_supervisor``."""

    from .config import config

    llm = config().llm
    return LangGraphSupervisor(llm=llm)
