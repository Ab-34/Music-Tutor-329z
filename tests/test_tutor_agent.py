"""The tutor agent, driven by a scripted fake LLM (no network): tools, decisions, saved progress."""
import json

import pytest

from examples.rondo_learner import SimLearner
from tutor import llm
from tutor.lesson import rondo_piece
from tutor.notes import save_midi
from tutor.tools import memory
from tutor import tutor_agent as T

PIECE = rondo_piece()


@pytest.fixture(autouse=True)
def tmp_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "ROOT", tmp_path / "memory")


def _call(name, args, i):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(steps):
    """A fake llm.chat that replays one list of tool calls per model turn."""
    seen = []

    def chat(messages, **kw):
        assert kw["tag"] == "tutor"
        seen.append(messages)
        calls = [_call(n, a, f"{len(seen)}_{j}") for j, (n, a) in enumerate(steps[len(seen) - 1])]
        return {"choices": [{"message": {"content": "", "tool_calls": calls}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "cost_usd": 0.002, "cached": False}
    return chat, seen


def take(tmp_path, level, assignment, reps=0, seed=0):
    path = tmp_path / f"{level}_{reps}.mid"
    save_midi(SimLearner(level, seed).play(T.assignment_target(PIECE, assignment), reps), path)
    return str(path)


def test_first_assignment_is_the_whole_piece_at_full_level():
    assert T.current_assignment("new_learner", PIECE) == {"scope": "whole", "level": "full"}


def test_far_off_take_starts_a_beginner_lesson_and_progress_is_saved(tmp_path, monkeypatch):
    chat, seen = scripted([
        [("get_assignment", {}), ("get_learner_history", {})],
        [("transcribe_recording", {})],
        [("compare_to_assignment", {})],
        [("score_performance", {})],
        [("preview_version", {"scope": "0", "level": "beginner"})],
        [("finish_turn", {"decision": "beginner_lesson", "feedback": "Let's start with bars 1-2.",
                          "next_scope": "0", "next_level": "beginner", "reason": "score in the chunk band"})],
    ])
    monkeypatch.setattr(llm, "chat", chat)
    rec = take(tmp_path, "struggling", T.FIRST_ASSIGNMENT)
    res = T.run_tutor_turn("ana", rec, PIECE)
    assert res["decision"] == "beginner_lesson" and res["rubric"]["band"] == "chunk"
    assert [t["tool"] for t in res["trace"]] == ["get_assignment", "get_learner_history", "transcribe_recording",
                                                  "compare_to_assignment", "score_performance", "preview_version",
                                                  "finish_turn"]
    assert all(t["ok"] for t in res["trace"])
    assert T.current_assignment("ana", PIECE) == {"scope": 0, "level": "beginner"}
    assert memory.read_progress("ana", PIECE.name)["history"][0]["score"] == res["rubric"]["total"]
    assert memory.read("ana", PIECE.name)  # mistakes logged
    # the mistakes the tool reported are located by bar and beat
    tool_out = json.loads(next(m["content"] for m in seen[3] if m.get("role") == "tool" and "candidates" in m["content"]))
    assert all(c["description"].startswith(("Bar", "Pickup")) for c in tool_out["candidates"])


def test_agent_can_keep_only_the_mistakes_it_believes(tmp_path, monkeypatch):
    chat, _ = scripted([
        [("transcribe_recording", {})], [("compare_to_assignment", {})],
        [("finish_turn", {"decision": "polish", "feedback": "Only bar 3.", "mistake_ids": [0],
                          "next_scope": "whole", "next_level": "full", "reason": "close"})],
    ])
    monkeypatch.setattr(llm, "chat", chat)
    res = T.run_tutor_turn("ben", take(tmp_path, "intermediate", T.FIRST_ASSIGNMENT), PIECE)
    assert len(res["mistakes"]) == 1
    assert sum(res["rubric"]["counts"].values()) == 1      # scored on the one mistake it kept


def test_invalid_next_assignment_is_reported_back_and_can_be_fixed(tmp_path, monkeypatch):
    chat, seen = scripted([
        [("transcribe_recording", {})],
        [("finish_turn", {"decision": "next_chunk", "feedback": "x", "next_scope": "9", "next_level": "full",
                          "reason": "r"})],
        [("finish_turn", {"decision": "next_chunk", "feedback": "x", "next_scope": "1", "next_level": "beginner",
                          "reason": "r"})],
    ])
    monkeypatch.setattr(llm, "chat", chat)
    res = T.run_tutor_turn("cy", take(tmp_path, "nearly_there", {"scope": 0, "level": "full"}), PIECE)
    assert [t["ok"] for t in res["trace"]] == [True, False, True]
    assert "chunk must be" in res["trace"][1]["error"]
    assert res["next_assignment"] == {"scope": 1, "level": "beginner"}


def test_turn_without_finish_keeps_the_assignment(tmp_path, monkeypatch):
    chat, _ = scripted([[("transcribe_recording", {})]] * 3)
    monkeypatch.setattr(llm, "chat", chat)
    res = T.run_tutor_turn("dee", take(tmp_path, "nearly_there", T.FIRST_ASSIGNMENT), PIECE, max_turns=3)
    assert res["error"] and res["next_assignment"] == T.FIRST_ASSIGNMENT
    assert T.current_assignment("dee", PIECE) == T.FIRST_ASSIGNMENT


def test_every_turn_saves_a_full_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TRANSCRIPTS", tmp_path / "sessions")
    chat, _ = scripted([
        [("transcribe_recording", {})], [("compare_to_assignment", {})],
        [("finish_turn", {"decision": "polish", "feedback": "Nice.", "next_scope": "whole", "next_level": "full",
                          "reason": "close"})],
    ])
    monkeypatch.setattr(llm, "chat", chat)
    res = T.run_tutor_turn("eve", take(tmp_path, "nearly_there", T.FIRST_ASSIGNMENT), PIECE)
    [tr] = T.load_transcripts("eve", tmp_path / "sessions")
    assert tr["decision"] == "polish" and tr["attempt"] == 1 and res["transcript"]
    roles = [m["role"] for m in tr["messages"]]
    assert roles[:2] == ["system", "user"] and roles.count("tool") == 3       # every tool result is kept
    assert tr["transcription"] and tr["trace"][-1]["tool"] == "finish_turn"
