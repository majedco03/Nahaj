"""Tests for the planning agent's schedule design."""

from __future__ import annotations

import copy
from collections import Counter
from datetime import date

import pytest

from backend.agents import PlanDesign, planning_agent


TODAY = "2026-09-10"

BASE_STATE: dict = {
    "semester": {"start_date": "2026-09-01", "end_date": "2026-12-15", "sessions_per_day": 2},
    "courses": [
        {
            "course_name": "Algorithms",
            "deadlines": [{"title": "Midterm", "date": "2026-10-05", "type": "exam"}],
            "materials": [
                {"title": "Graph Theory", "page_count": 200, "content": "..."},
                {"title": "Intro Deck", "slide_count": 10, "content": "..."},
            ],
            "completed_topics": [],
            "quiz_performance": [],
            "current_plan": [],
        }
    ],
}


def state(**overrides) -> dict:
    value = copy.deepcopy(BASE_STATE)
    course = value["courses"][0]
    for key, item in overrides.items():
        course[key] = item
    return value


def sessions_for(result, prefix: str) -> list[dict]:
    return [row for row in result["schedule"] if row["title"].startswith(prefix)]


class _DesignModel:
    """A model that returns one prepared schedule and records the brief it saw."""

    def __init__(self, design: PlanDesign):
        self.design = design
        self.brief = ""

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.brief = messages[-1]["content"]
        return self.design


class _BrokenModel:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        raise RuntimeError("OpenRouter is unavailable")


@pytest.mark.parametrize(
    ("pages", "expected"),
    [(10, 1), (60, 5), (200, 6)],
)
def test_document_length_decides_how_many_sessions_a_document_gets(pages, expected):
    result = planning_agent(llm=None).plan(
        state(materials=[{"title": "Doc", "page_count": pages, "content": "..."}]),
        today=TODAY,
    )

    assert len(sessions_for(result, "Study Doc")) == expected


def test_weak_mastery_deepens_and_promotes_a_topic():
    weak = planning_agent(llm=None).plan(
        state(quiz_performance=[{"topic": "Graph Theory", "estimated_mastery_level": 0.1}]),
        today=TODAY,
    )
    strong = planning_agent(llm=None).plan(
        state(quiz_performance=[{"topic": "Graph Theory", "estimated_mastery_level": 0.95}]),
        today=TODAY,
    )

    assert len(sessions_for(weak, "Study Graph Theory")) > len(sessions_for(strong, "Study Graph Theory"))
    assert sessions_for(weak, "Study Graph Theory")[0]["priority_score"] > sessions_for(strong, "Study Graph Theory")[0]["priority_score"]
    # Only the weak topic earns its own revision task.
    assert sessions_for(weak, "Review Graph Theory")
    assert not sessions_for(strong, "Review Graph Theory")


def test_completed_material_is_left_out_of_the_plan():
    result = planning_agent(llm=None).plan(
        state(completed_topics=[{"title": "Graph Theory", "date": "2026-09-09"}]),
        today=TODAY,
    )

    assert not sessions_for(result, "Study Graph Theory")
    assert sessions_for(result, "Study Intro Deck")


def test_the_plan_never_exceeds_the_daily_session_capacity():
    result = planning_agent(llm=None).plan(state(), today=TODAY)

    load = Counter(row["date"] for row in result["schedule"])
    assert max(load.values()) <= 2


def test_every_session_stays_inside_the_semester_window():
    result = planning_agent(llm=None).plan(state(), today=TODAY)

    assert all(TODAY <= row["date"] <= "2026-12-15" for row in result["schedule"])


def test_pace_evidence_reports_overdue_work():
    result = planning_agent(llm=None).plan(
        state(
            current_plan=[
                {
                    "title": "Algorithms study plan",
                    "completed_tasks": [{"title": "Read syllabus", "date": "2026-09-02"}],
                    "remaining_tasks": [{"title": "Problem set 1", "date": "2026-09-04"}],
                }
            ]
        ),
        today=TODAY,
    )

    assert result["pace"]["overdue_tasks"] == 1
    assert result["pace"]["status"] == "behind"
    assert result["summary"]["completion_ratio"] == 0.5


def test_the_model_designs_the_schedule_and_sees_the_full_library():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    graph_id = next(item["id"] for item in candidates if item["title"] == "Study Graph Theory")
    model = _DesignModel(
        PlanDesign(
            sessions=[
                {"candidate_id": graph_id, "day": "2026-09-12", "session": 1, "of": 3, "focus": "pages 1-70"},
                {"candidate_id": graph_id, "day": "2026-09-19", "session": 2, "of": 3, "focus": "pages 71-140"},
                {"candidate_id": graph_id, "day": "2026-09-26", "session": 3, "of": 3, "focus": "pages 141-200"},
            ],
            rationale="Graph Theory is the longest document.",
        )
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    assert result["design"]["source"] == "model"
    assert result["design"]["rationale"] == "Graph Theory is the longest document."
    # The model asked for three sessions, so the plan has three.
    assert [row["date"] for row in sessions_for(result, "Study Graph Theory")] == [
        "2026-09-12",
        "2026-09-19",
        "2026-09-26",
    ]
    assert "pages 71-140" in " ".join(
        pair["event"]["description"] for pair in result["event_day_pairs"]
    )
    # The brief carries the inputs the planner is supposed to reason over.
    for key in ("'pages'", "'mastery'", "'current_plan'", "'completed_tasks'", "'remaining_tasks'", "'pace'"):
        assert key in model.brief
    # The computed size reaches the model as advice, not as a fixed instruction.
    assert "'suggested_sessions'" in model.brief


def test_a_session_for_an_unknown_candidate_is_rejected():
    model = _DesignModel(
        PlanDesign(sessions=[{"candidate_id": "not-a-candidate", "day": "2026-09-12", "session": 1, "of": 1, "focus": ""}])
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    assert all(row["candidate_id"] != "not-a-candidate" for row in result["schedule"])
    assert result["design"]["source"] == "rules"


def test_a_date_outside_the_semester_is_rescheduled():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    deck_id = next(item["id"] for item in candidates if item["title"] == "Study Intro Deck")
    model = _DesignModel(
        PlanDesign(sessions=[{"candidate_id": deck_id, "day": "2027-05-01", "session": 1, "of": 1, "focus": ""}])
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    placed = sessions_for(result, "Study Intro Deck")
    assert placed and TODAY <= placed[0]["date"] <= "2026-12-15"


def test_a_candidate_the_model_skipped_is_still_scheduled_at_full_effort():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    deck_id = next(item["id"] for item in candidates if item["title"] == "Study Intro Deck")
    model = _DesignModel(
        PlanDesign(sessions=[{"candidate_id": deck_id, "day": "2026-09-12", "session": 1, "of": 1, "focus": ""}])
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    assert result["design"]["backfilled_candidates"] >= 1
    assert len(sessions_for(result, "Study Graph Theory")) == 6


def test_planning_falls_back_to_rules_when_the_model_fails():
    result = planning_agent(llm=_BrokenModel()).plan(state(), today=TODAY)

    assert result["design"]["source"] == "rules"
    assert result["summary"]["total_tasks"] > 0


def test_replanning_does_not_multiply_the_plan():
    working = state()
    planner = planning_agent(llm=None)

    counts = [planner.plan(working, today=TODAY)["summary"]["total_tasks"] for _ in range(4)]

    assert len(set(counts)) == 1


def test_a_replan_reports_what_changed():
    working = state()
    planner = planning_agent(llm=None)
    planner.plan(working, today=TODAY)

    revision = planner.plan(working, today=TODAY)["revision"]

    assert revision["is_revision"] is True
    assert revision["added"] == 0
    assert revision["dropped"] == 0


def test_the_model_decides_the_depth_of_a_document():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    graph_id = next(item["id"] for item in candidates if item["title"] == "Study Graph Theory")
    suggested = next(item["effort_units"] for item in candidates if item["id"] == graph_id)
    model = _DesignModel(
        PlanDesign(sessions=[{"candidate_id": graph_id, "day": "2026-09-12", "session": 1, "of": 2, "focus": ""}])
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    # Two sessions were asked for, so two are scheduled, whatever the arithmetic suggested.
    assert suggested == 6
    assert len(sessions_for(result, "Study Graph Theory")) == 2


def test_sessions_the_model_asked_for_but_did_not_place_are_filled_in():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    graph_id = next(item["id"] for item in candidates if item["title"] == "Study Graph Theory")
    model = _DesignModel(
        PlanDesign(sessions=[{"candidate_id": graph_id, "day": "2026-09-12", "session": 1, "of": 4, "focus": ""}])
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    assert len(sessions_for(result, "Study Graph Theory")) == 4
    assert result["design"]["topped_up_sessions"] == 3


def test_the_students_request_reaches_the_model():
    model = _DesignModel(PlanDesign(sessions=[]))

    planning_agent(llm=model).plan(state(), today=TODAY, request="only chapter 4")

    assert "only chapter 4" in model.brief


def test_the_model_can_narrow_the_plan_to_the_requested_material():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    keep = next(item["id"] for item in candidates if item["title"] == "Study Graph Theory")
    model = _DesignModel(
        PlanDesign(
            sessions=[{"candidate_id": keep, "day": "2026-09-12", "session": 1, "of": 6, "focus": ""}],
            excluded=[
                {"candidate_id": item["id"], "reason": "outside the requested scope"}
                for item in candidates
                if item["id"] != keep
            ],
        )
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY, request="only Graph Theory")

    assert not sessions_for(result, "Study Intro Deck")
    assert not sessions_for(result, "Prepare for Midterm")
    assert len(sessions_for(result, "Study Graph Theory")) == 6
    assert {item["title"] for item in result["design"]["excluded"]} == {"Study Intro Deck", "Prepare for Midterm"}


def test_excluding_everything_is_ignored():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    model = _DesignModel(
        PlanDesign(
            sessions=[],
            excluded=[{"candidate_id": item["id"], "reason": "no"} for item in candidates],
        )
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY)

    assert result["summary"]["total_tasks"] > 0
    assert result["design"]["excluded"] == []


def test_a_replan_keeps_a_documents_depth():
    working = state()
    planner = planning_agent(llm=None)
    first = planner.plan(working, today=TODAY)

    second = planner.plan(working, today=TODAY)

    # The page count lives on the document, not the task, and must survive a replan.
    assert len(sessions_for(second, "Study Graph Theory")) == len(sessions_for(first, "Study Graph Theory")) == 6


def test_a_reply_to_the_approval_question_revises_the_draft():
    """The third turn of a real conversation: not yes, not no, but an instruction."""

    from backend.workflow import LangGraphSupervisor

    seen: list[str] = []

    class _Planner:
        def prepare_plan(self, _state, **kwargs):
            seen.append(str(kwargs.get("request") or ""))
            return {"status": "draft", "summary": {"courses_planned": 1, "total_tasks": 1}, "event_day_pairs": []}

        def respond(self, _query, _result, _instruction):
            return "Revised draft. May I add it to your calendar?"

    gateway = object.__new__(LangGraphSupervisor)
    gateway.planner = _Planner()
    node_state = {
        "request": {"context": {}, "messages": []},
        "shared_state": {"courses": []},
        "pending": {"action": "calendar_approval", "next": "planning", "plan": {}},
        "resume_answer": {"text": "revise it so it only includes chapter 4 CLO#1.1"},
    }

    result = LangGraphSupervisor._planning_node(gateway, node_state)

    assert seen == ["revise it so it only includes chapter 4 CLO#1.1"]
    assert result["pending"]["action"] == "calendar_approval"
    assert result["pending"]["message"] == "Revised draft. May I add it to your calendar?"


def test_a_finish_date_compresses_the_plan_instead_of_repeating_it():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    model = _DesignModel(
        PlanDesign(
            sessions=[
                {"candidate_id": item["id"], "day": "2026-09-10", "session": 1, "of": item["effort_units"], "focus": ""}
                for item in candidates
            ],
            horizon="2026-09-12",
            sessions_per_day=6,
        )
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY, request="I need to finish before the 12th")

    assert result["design"]["horizon"] == "2026-09-12"
    assert result["design"]["horizon_met"] is True
    assert result["summary"]["last_task_date"] <= "2026-09-12"


def test_an_impossible_finish_date_is_reported_rather_than_hidden():
    working = state()
    candidates = planning_agent(llm=None)._collect_candidates(
        working["courses"], date(2026, 9, 10), date(2026, 12, 15)
    )[0]
    model = _DesignModel(
        PlanDesign(
            sessions=[
                {"candidate_id": item["id"], "day": "2026-09-10", "session": 1, "of": 8, "focus": ""}
                for item in candidates
            ],
            horizon="2026-09-10",
            sessions_per_day=99,
        )
    )

    result = planning_agent(llm=model).plan(state(), today=TODAY, request="finish it all today")

    # The daily load is capped, so the work spills and the result says so.
    assert result["design"]["horizon"] == "2026-09-10"
    assert result["design"]["horizon_met"] is False
    assert result["design"]["sessions_per_day"] <= 8


def test_a_revision_that_changes_nothing_is_reported_instead_of_repeated():
    from backend.workflow import LangGraphSupervisor

    draft = {
        "status": "draft",
        "summary": {"courses_planned": 1, "total_tasks": 1},
        "event_day_pairs": [{"event": {"title": "Study Chapter 6"}, "day": "2026-09-19"}],
    }
    captured: dict = {}

    class _Planner:
        def prepare_plan(self, _state, **_kwargs):
            return dict(draft)

        def respond(self, _query, result, instruction):
            captured["result"] = result
            captured["instruction"] = instruction
            return "That change could not be applied."

    gateway = object.__new__(LangGraphSupervisor)
    gateway.planner = _Planner()
    node_state = {
        "request": {"context": {}, "messages": []},
        "shared_state": {"courses": []},
        "pending": {"action": "calendar_approval", "next": "planning", "plan": dict(draft)},
        "resume_answer": {"text": "thats not suitable, I need to finish before 12"},
    }

    LangGraphSupervisor._planning_node(gateway, node_state)

    assert captured["result"]["unchanged"] is True
    assert "could not be applied" in captured["instruction"]


def test_a_revision_that_does_change_the_plan_is_not_flagged():
    from backend.workflow import LangGraphSupervisor

    before = {
        "status": "draft",
        "event_day_pairs": [{"event": {"title": "Study Chapter 6"}, "day": "2026-09-19"}],
    }
    after = {
        "status": "draft",
        "summary": {},
        "event_day_pairs": [{"event": {"title": "Study Chapter 6"}, "day": "2026-09-11"}],
    }
    captured: dict = {}

    class _Planner:
        def prepare_plan(self, _state, **_kwargs):
            return dict(after)

        def respond(self, _query, result, instruction):
            captured["result"] = result
            captured["instruction"] = instruction
            return "Revised."

    gateway = object.__new__(LangGraphSupervisor)
    gateway.planner = _Planner()
    LangGraphSupervisor._planning_node(
        gateway,
        {
            "request": {"context": {}, "messages": []},
            "shared_state": {"courses": []},
            "pending": {"action": "calendar_approval", "next": "planning", "plan": dict(before)},
            "resume_answer": {"text": "move it earlier"},
        },
    )

    assert "unchanged" not in captured["result"]
    assert "could not be applied" not in captured["instruction"]


def test_a_missed_finish_date_changes_how_the_draft_is_presented():
    from backend.workflow import LangGraphSupervisor

    captured: dict = {}

    class _Planner:
        def respond(self, _query, _result, instruction):
            captured["instruction"] = instruction
            return "narrated"

    gateway = object.__new__(LangGraphSupervisor)
    gateway.planner = _Planner()
    result = {
        "status": "draft",
        "design": {"horizon": "2026-09-12", "horizon_met": False},
        "summary": {"last_task_date": "2026-09-19"},
        "event_day_pairs": [],
    }

    LangGraphSupervisor._plan_result(gateway, result, {"request": {"messages": []}, "shared_state": {}})

    assert "does not fit the finish date" in captured["instruction"]
