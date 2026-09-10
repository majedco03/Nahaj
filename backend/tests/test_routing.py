"""Tests for what the coordinator sees before it routes."""

from __future__ import annotations

import asyncio

from backend.agents import (
    RAG_agent,
    communication_agent,
    planning_agent,
    quiz_agent,
    supervisor_agent,
)
from backend.supervisor_contract import SupervisorRequest
from backend.workflow import LangGraphSupervisor


class _RouterModel:
    """Routes on the original wording, and records the brief it was given."""

    def __init__(self):
        self.brief = ""

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.brief = messages[-1]["content"]
        original = self.brief.split("<original_message>")[1].split("</original_message>")[0].casefold()
        if "plan" in original or "schedule" in original:
            return {"agent": "planning_agent", "reason": "the student wants their plan changed"}
        if "quiz" in original:
            return {"agent": "quiz_agent", "reason": "a quiz was requested"}
        return {"agent": "RAG_agent", "reason": "a course question"}


class _LossyCommunicator:
    """Paraphrases the way the real communication agent does, losing detail."""

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        text = messages[-1]["content"].rsplit("Latest message:", 1)[-1].strip()
        rewritten = "Update quiz coverage for chapter 4" if "quiz on sunday" in text.casefold() else text
        return {"understood": True, "task": rewritten, "response": "", "answers_pending": False}


class _Narrator:
    def invoke(self, _messages):
        return {"content": "ok"}


def _gateway(router_model: _RouterModel) -> LangGraphSupervisor:
    gateway = LangGraphSupervisor(
        rag=RAG_agent.__new__(RAG_agent),
        quiz=quiz_agent(llm=None),
        planner=planning_agent(llm=_Narrator()),
        communicator=communication_agent(llm=_LossyCommunicator()),
    )
    gateway.router = supervisor_agent(router_model)
    return gateway


CONVERSATION = [
    {"role": "user", "content": "create a study plan"},
    {"role": "assistant", "content": "Draft study plan: Networks (CEN445). Approve?"},
    {
        "role": "user",
        "content": "My quiz on sunday only covers chapter 4 (CLO#1.1), I want the plan to only focus on it.",
    },
]


def _run(gateway: LangGraphSupervisor, messages) -> list:
    request = SupervisorRequest(
        type="chat",
        messages=messages,
        context={"thread_id": "route-thread", "shared_state": {"courses": []}},
    )

    async def collect():
        return [event async for event in gateway.handle(request)]

    return asyncio.run(collect())


def test_the_coordinator_sees_the_original_message_and_the_rewritten_task():
    router_model = _RouterModel()

    _run(_gateway(router_model), CONVERSATION)

    brief = router_model.brief
    assert "My quiz on sunday only covers chapter 4" in brief.split("</original_message>")[0]
    assert "Update quiz coverage for chapter 4" in brief.split("<rewritten_task>")[1].split("</rewritten_task>")[0]


def test_the_coordinator_sees_the_turns_before_the_current_one():
    router_model = _RouterModel()

    _run(_gateway(router_model), CONVERSATION)

    history = router_model.brief.split("<recent_conversation>")[1].split("</recent_conversation>")[0]
    assert "create a study plan" in history
    assert "Draft study plan" in history
    # The current message is supplied on its own, not repeated in the history.
    assert "quiz on sunday" not in history


def test_a_misleading_paraphrase_does_not_decide_the_route():
    router_model = _RouterModel()
    gateway = _gateway(router_model)
    routes: list[str] = []
    node = gateway._supervisor_node

    def spy(state):
        result = node(state)
        routes.append(result["route"])
        return result

    gateway._supervisor_node = spy
    gateway.graph = gateway._build_graph()

    _run(gateway, CONVERSATION)

    # The rewrite says "quiz"; the student asked for their plan to change.
    assert routes == ["planning"]


def test_the_coordinator_reports_why_it_routed():
    decision = supervisor_agent(_RouterModel()).supervise(
        "Update quiz coverage",
        original="please change my study plan",
        history=[{"role": "user", "content": "create a study plan"}],
    )

    assert decision["agent"] == "planning_agent"
    assert decision["reason"] == "the student wants their plan changed"


def test_recent_turns_are_bounded_and_exclude_the_current_message():
    messages = [{"role": "user", "content": f"message {index} " + "x" * 500} for index in range(10)]

    turns = LangGraphSupervisor._recent_turns(messages)

    assert len(turns) == 6
    assert all(len(turn["content"]) <= 300 for turn in turns)
    assert "message 9" not in " ".join(turn["content"] for turn in turns)


def test_routing_without_an_original_falls_back_to_the_task():
    router_model = _RouterModel()

    decision = supervisor_agent(router_model).supervise("build me a schedule")

    assert decision["agent"] == "planning_agent"
    assert "build me a schedule" in router_model.brief.split("</original_message>")[0]
