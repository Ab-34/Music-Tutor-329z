"""Rubric, chunks, simulated learner, policies and guardrails for the practice loop (no network)."""
import json

import pytest

from examples.rondo_learner import SimLearner
from tutor import lesson as L
from tutor import llm, rubric
from tutor.notes import Note
from tutor.tools.align import find_errors

PIECE = L.rondo_piece()


# ---- rubric -----------------------------------------------------------------

def test_rubric_perfect_and_weights():
    notes = [Note(60 + i, i * 0.5, i * 0.5 + 0.4) for i in range(40)]   # allowed = max(0.05*40, 2) = 2
    assert rubric.score(notes, []).total == 100
    r = rubric.score(notes, [{"type": "wrong_pitch", "time": 1.0, "pitch": 61},
                             {"type": "timing", "time": 2.0, "pitch": 64}])
    assert r.parts == {"right_notes": 50, "complete": 100, "rhythm": 50, "clean": 100}
    assert r.total == round(0.35 * 50 + 0.30 * 100 + 0.20 * 50 + 0.15 * 100)
    assert r.counts["right_notes"] == 1


def test_bands():
    assert rubric.band(85) == "polish" and rubric.band(84) == "simplify"
    assert rubric.band(50) == "simplify" and rubric.band(49) == "chunk"


def test_rondo_levels_land_in_their_bands():
    from gui.sources import rondo_takes
    from tutor.notes import load_midi
    bands = {}
    for t in rondo_takes():
        bands.setdefault(t.level, set()).add(rubric.score(load_midi(t.ref), t.gt).band)
    assert bands == {"perfect": {"polish"}, "nearly_there": {"polish"}, "intermediate": {"simplify"},
                     "struggling": {"chunk"}}


# ---- chunks and rungs -------------------------------------------------------

def test_four_chunks_cover_every_note_once():
    cs = L.chunks(PIECE)
    assert [c.label for c in cs] == ["Pickup + bars 1–2", "Bars 3–4", "Bars 5–6", "Bars 7–8"]
    assert sum(len(c.notes) for c in cs) == len(PIECE.melody)
    assert {id(n) for c in cs for n in c.notes} == {id(n) for n in PIECE.melody}


def test_rungs_get_harder():
    from examples.rondo_learner import difficulty
    for c in L.chunks(PIECE):
        d = [difficulty(L.target(PIECE, c, r).notes) for r in range(len(L.RUNGS))]
        assert d[0] <= d[1] < d[2], (c.label, d)
        assert L.target(PIECE, c, 0).notes[0].onset == pytest.approx(0, abs=1e-3)  # rebased to 0 (onsets are rounded to 0.1 ms)


def test_target_bar_beat_maps_back_to_the_score():
    c = L.chunks(PIECE)[1]                       # bars 3-4
    tgt = L.target(PIECE, c, 1)                  # slowed down
    assert tgt.bar_beat(tgt.notes[0].onset) == (3, 1)


# ---- simulated learner ------------------------------------------------------

def _score(level, chunk, rung, reps, seed=0):
    tgt = L.target(PIECE, L.chunks(PIECE)[chunk], rung)
    att = SimLearner(level, seed).play(tgt, reps)
    return rubric.score(tgt.notes, find_errors(tgt.notes, att)).total


def test_learner_is_deterministic_improves_and_levels_differ():
    tgt = L.target(PIECE, L.chunks(PIECE)[0], 1)
    a, b = SimLearner("struggling", 3).play(tgt, 2), SimLearner("struggling", 3).play(tgt, 2)
    assert [(n.pitch, n.onset) for n in a] == [(n.pitch, n.onset) for n in b]
    mean = lambda level, reps: sum(_score(level, c, 1, reps, s) for c in range(4) for s in range(4)) / 16
    assert mean("struggling", 7) > mean("struggling", 0)
    assert mean("nearly_there", 0) > mean("intermediate", 0) > mean("struggling", 0)
    assert _score("perfect", 2, 2, 0) == 100


# ---- guardrails and policies ------------------------------------------------

def test_guardrails():
    top = len(L.RUNGS) - 1
    assert L.guard(L.State(chunk=0, rung=top, n_chunks=4), "harder")[0] == "next_chunk"
    assert L.guard(L.State(chunk=3, rung=top, n_chunks=4), "harder")[0] == "finish"
    assert L.guard(L.State(chunk=0, rung=0, n_chunks=4), "easier")[0] == "repeat"
    assert L.guard(L.State(chunk=1, rung=1, n_chunks=4), "finish")[0] == "repeat"
    capped = L.State(chunk=1, rung=1, n_chunks=4, attempts_on_chunk=L.MAX_ATTEMPTS_PER_CHUNK)
    assert L.guard(capped, "repeat") == ("next_chunk", f"attempt cap ({L.MAX_ATTEMPTS_PER_CHUNK} per chunk)")


@pytest.mark.parametrize("level", ["nearly_there", "intermediate"])
def test_rules_lesson_finishes(level):
    s = L.Lesson(PIECE, SimLearner(level, 0), L.RulesPolicy(), log_dir=None).run()
    assert s["finished"] and s["chunks_passed"] >= 3 and s["cost_usd"] == 0


def _fake_chat(reply):
    def chat(messages, **kw):
        assert kw["tag"] == "coach" and "suggested_action" in messages[-1]["content"]
        return {"choices": [{"message": {"content": reply}}], "cost_usd": 0.001, "cached": False}
    return chat


def test_coach_is_followed_and_logged(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", _fake_chat(json.dumps(
        {"action": "harder", "feedback": "Good. Now all the notes.", "reason": "steady"})))
    lesson = L.Lesson(PIECE, SimLearner("struggling", 0), L.CoachPolicy(), log_dir=tmp_path)
    row = lesson.step()
    assert row["action"] == "harder" and lesson.state.rung == 1 and not row["fallback"]
    assert row["chosen"] != row["suggested"]           # rules would have repeated a low score
    assert json.loads(lesson.log.read_text().splitlines()[0])["feedback"] == "Good. Now all the notes."


@pytest.mark.parametrize("reply", ["not json", json.dumps({"action": "teleport", "feedback": "", "reason": ""})])
def test_coach_falls_back_to_rules(monkeypatch, reply):
    monkeypatch.setattr(llm, "chat", _fake_chat(reply))
    lesson = L.Lesson(PIECE, SimLearner("struggling", 0), L.CoachPolicy(), log_dir=None)
    row = lesson.step()
    assert row["fallback"] and row["action"] == row["suggested"]


def test_coach_harder_at_top_rung_moves_on(monkeypatch):
    monkeypatch.setattr(llm, "chat", _fake_chat(json.dumps({"action": "harder", "feedback": "", "reason": ""})))
    lesson = L.Lesson(PIECE, SimLearner("nearly_there", 0), L.CoachPolicy(), log_dir=None)
    lesson.state.rung = len(L.RUNGS) - 1
    row = lesson.step()
    assert row["action"] == "next_chunk" and row["guard"] == "top level passed" and lesson.state.chunk == 1
