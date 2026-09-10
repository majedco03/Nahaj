"""Tests for how the quiz route finds course material."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from backend.agents import communication_agent, planning_agent, quiz_agent, supervisor_agent
from backend.supervisor_contract import SupervisorRequest
from backend.workflow import LangGraphSupervisor


TWO_COURSES = {
    "courses": [
        {
            "id": "c-hist",
            "course_name": "History",
            "code": "HIS101",
            "materials": [{"title": "Rome.pdf", "status": "ready"}],
        },
        {
            "id": "c-algo",
            "course_name": "Algorithms",
            "code": "CS220",
            "materials": [{"title": "Graph Theory.pdf", "status": "ready"}],
        },
    ]
}


class _Communicator:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        text = messages[-1]["content"].rsplit("Latest message:", 1)[-1].strip()
        return {"understood": True, "task": text, "response": "", "answers_pending": False}


class _Router:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return {"agent": "quiz_agent"}


class _Quiz:
    def generate_quiz(self, **kwargs) -> dict[str, Any]:
        material = kwargs.get("material")
        if not material:
            return {
                "status": "needs_material",
                "message": "Add course material with text or sections before generating a quiz.",
            }
        return {
            "status": "ready",
            "questions": [
                {"id": "q1", "prompt": "?", "topic": material[0]["content"], "options": ["a", "b", "c", "d"]}
            ],
            "answer_key": [{"id": "q1", "correct_option_id": "A", "explanation": "x"}],
        }

    def describe_quiz(self, _query, _quiz=None, course=None) -> str:
        return f"Quiz ready from {course}."


class _Retriever:
    """A RAG stand-in whose collections can be missing, like a partial index."""

    def __init__(self, index: Mapping[str, list[str]]):
        self.index = dict(index)
        self.asked: list[str] = []

    def retrieve(self, _query, course_id, top_k=3):
        self.asked.append(str(course_id))
        if str(course_id) not in self.index:
            raise RuntimeError(f"no collection for {course_id}")
        return [{"text": text, "section": text} for text in self.index[str(course_id)]][:top_k]


def gateway(retriever: _Retriever) -> LangGraphSupervisor:
    gw = LangGraphSupervisor(
        rag=retriever,
        quiz=_Quiz(),
        planner=planning_agent(llm=None),
        communicator=communication_agent(llm=_Communicator()),
    )
    gw.router = supervisor_agent(_Router())
    return gw


def ask(gw: LangGraphSupervisor, message: str, state: Mapping[str, Any]) -> list[Any]:
    request = SupervisorRequest(
        type="chat",
        messages=[{"role": "user", "content": message}],
        context={"thread_id": "quiz-thread", "shared_state": dict(state)},
    )

    async def collect():
        return [event async for event in gw.handle(request)]

    return asyncio.run(collect())


def texts(events) -> str:
    return " ".join(str(event.data.get("text", "")) for event in events if event.kind == "text_delta")


def test_the_named_course_is_searched_not_merely_the_first_one():
    retriever = _Retriever({"c-hist": ["Rome fell."], "c-algo": ["A graph has vertices."]})

    events = ask(gateway(retriever), "Quiz me on graph theory", TWO_COURSES)

    assert retriever.asked == ["c-algo"]
    assert any(event.kind == "quiz_create" for event in events)


def test_a_course_is_matched_by_its_document_title():
    retriever = _Retriever({"c-hist": ["Rome fell."], "c-algo": ["A graph has vertices."]})

    ask(gateway(retriever), "test me on Rome", TWO_COURSES)

    assert retriever.asked == ["c-hist"]


def test_retrieval_falls_back_to_another_course_with_material():
    retriever = _Retriever({"c-algo": ["A graph has vertices."]})

    events = ask(gateway(retriever), "make me a quiz", TWO_COURSES)

    assert retriever.asked == ["c-hist", "c-algo"]
    assert any(event.kind == "quiz_create" for event in events)


def test_an_unreadable_index_is_not_reported_as_missing_material():
    events = ask(gateway(_Retriever({})), "quiz me", TWO_COURSES)

    message = texts(events)
    assert "could not read its search index" in message
    assert "Upload your slides" not in message


def test_material_that_is_still_indexing_says_so():
    state = {
        "courses": [
            {"id": "c1", "course_name": "Algorithms", "materials": [{"title": "Graphs.pdf", "status": "processing"}]}
        ]
    }

    events = ask(gateway(_Retriever({})), "quiz me", state)

    assert "still being prepared" in texts(events)


def test_an_empty_library_asks_for_an_upload():
    state = {"courses": [{"id": "c1", "course_name": "Algorithms", "materials": []}]}

    events = ask(gateway(_Retriever({})), "quiz me", state)

    assert "Upload your slides" in texts(events)


def test_an_explicit_course_id_wins_over_the_query():
    retriever = _Retriever({"c-hist": ["Rome fell."], "c-algo": ["A graph has vertices."]})
    gw = gateway(retriever)
    request = SupervisorRequest(
        type="chat",
        messages=[{"role": "user", "content": "Quiz me on graph theory"}],
        context={"thread_id": "t", "shared_state": dict(TWO_COURSES), "course_id": "c-hist"},
    )

    async def collect():
        return [event async for event in gw.handle(request)]

    asyncio.run(collect())

    assert retriever.asked == ["c-hist"]


class _ScopeModel:
    """A quiz specialist that reads a count out of the request."""

    def __init__(self, count: int, topics=None, course=""):
        self.count = count
        self.topics = topics or []
        self.course = course
        self.brief = ""

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        from backend.agents import QuizScope

        self.brief = messages[-1]["content"]
        return QuizScope(count=self.count, topics=self.topics, course=self.course)


class _CountingQuiz(quiz_agent):
    """Records the count it was asked for and returns exactly that many."""

    def __init__(self, scope_model):
        super().__init__(llm=scope_model)
        self.asked_for = None

    def describe_quiz(self, _query, _quiz=None, course=None) -> str:
        return f"Quiz ready from {course}."

    def generate_quiz(self, **kwargs):
        count = kwargs.get("count", 3)
        self.asked_for = count
        return {
            "status": "ready",
            "requested_count": count,
            "short_of_request": False,
            "questions": [
                {"id": f"q{index}", "prompt": "?", "topic": "t", "options": ["a", "b", "c", "d"]}
                for index in range(count)
            ],
            "answer_key": [
                {"id": f"q{index}", "correct_option_id": "A", "explanation": ""} for index in range(count)
            ],
        }


def _counting_gateway(quiz):
    gw = LangGraphSupervisor(
        rag=_Retriever({"c-algo": ["A graph has vertices."], "c-hist": ["Rome fell."]}),
        quiz=quiz,
        planner=planning_agent(llm=None),
        communicator=communication_agent(llm=_Communicator()),
    )
    gw.router = supervisor_agent(_Router())
    return gw


def test_the_requested_number_of_questions_is_produced():
    quiz = _CountingQuiz(_ScopeModel(15))

    events = ask(_counting_gateway(quiz), "give me 15 mcqs", TWO_COURSES)

    assert quiz.asked_for == 15
    created = [event for event in events if event.kind == "quiz_create"]
    assert len(created[0].data["questions"]) == 15


def test_the_scope_model_sees_the_conversation_and_the_courses():
    scope_model = _ScopeModel(10)
    quiz = _CountingQuiz(scope_model)
    gw = _counting_gateway(quiz)
    request = SupervisorRequest(
        type="chat",
        messages=[
            {"role": "user", "content": "quiz me on graph theory"},
            {"role": "assistant", "content": "3-question quiz ready."},
            {"role": "user", "content": "give me more from the same one"},
        ],
        context={"thread_id": "scope", "shared_state": dict(TWO_COURSES)},
    )

    async def collect():
        return [event async for event in gw.handle(request)]

    asyncio.run(collect())

    assert "quiz me on graph theory" in scope_model.brief
    assert "Algorithms" in scope_model.brief


def test_a_broken_scope_call_still_produces_a_quiz():
    class _Broken(_Quiz):
        def scope(self, *_args, **_kwargs):
            raise RuntimeError("scope model unavailable")

    events = ask(_counting_gateway(_Broken()), "give me a 10 question quiz", TWO_COURSES)

    assert any(event.kind == "quiz_create" for event in events)


def test_falling_short_of_the_request_is_said_out_loud():
    class _Short(_Quiz):
        def scope(self, *_args, **_kwargs):
            return {"count": 10, "topics": [], "course": ""}

        def generate_quiz(self, **_kwargs):
            return {
                "status": "ready",
                "requested_count": 10,
                "short_of_request": True,
                "questions": [{"id": "q1", "prompt": "?", "topic": "t", "options": ["a", "b", "c", "d"]}],
                "answer_key": [{"id": "q1", "correct_option_id": "A", "explanation": ""}],
            }

    events = ask(_counting_gateway(_Short()), "give me 10 questions", TWO_COURSES)

    assert "only supported 1 of the 10 questions" in texts(events)
