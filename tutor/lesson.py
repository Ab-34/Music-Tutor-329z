"""Chunked practice loop: split a piece into chunks, climb each chunk's ladder, move on.

    lesson = Lesson(rondo_piece(), learner, CoachPolicy())   # or RulesPolicy()
    while not lesson.done:
        row = lesson.step()        # one attempt: play -> align -> rubric -> decide -> apply

Each chunk has RUNGS (easiest first). A policy picks an action after every attempt:
repeat / easier / harder / next_chunk / finish. Guardrails turn impossible actions into the
nearest possible one and cap attempts per chunk. Every attempt is logged to
runs/lessons/<lesson_id>.jsonl.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tutor import llm, rubric
from tutor.notes import Note, pitch_name
from tutor.tools.align import find_errors
from tutor.tools.simplify import drop_ornaments, slow_down, thin_to_grid

ROOT = Path(__file__).resolve().parent.parent
LESSON_DIR = ROOT / "runs" / "lessons"
ACTIONS = ["repeat", "easier", "harder", "next_chunk", "finish"]
MAX_ATTEMPTS_PER_CHUNK = 8
SLOW_BPM = 60

RUNGS = [
    ("simplified", "Simplified: quick runs thinned to their main notes, no grace notes, 60 bpm"),
    ("full_slow", "All the notes, slowly (60 bpm)"),
    ("full", "All the notes at full tempo (the goal)"),
]


# ---- piece, chunks, targets -------------------------------------------------

@dataclass
class Piece:
    name: str
    melody: list[Note]                 # the one-note line being learned, at full tempo
    sec_per_beat: float
    pickup_beats: float = 0.0
    beats_per_bar: int | None = None   # None: no meter, chunk by time windows

    @property
    def bpm(self) -> float:
        return 60 / self.sec_per_beat

    def bar_beat(self, t: float) -> tuple[int, int]:
        """Piece time (s, full tempo) -> (bar, beat). Bar 0 is the pickup."""
        if not self.beats_per_bar:
            return int(t // 4) + 1, 1
        b = t / self.sec_per_beat - self.pickup_beats + 1e-3
        if b < 0:
            return 0, 1
        return int(b // self.beats_per_bar) + 1, int(b % self.beats_per_bar) + 1

    def where(self, t: float) -> str:
        if not self.beats_per_bar:
            return f"{t:.1f} s"
        bar, beat = self.bar_beat(t)
        return "Pickup" if bar == 0 else f"Bar {bar}, beat {beat}"


def rondo_piece() -> Piece:
    from examples.rondo_demo import BEATS_PER_BAR, PICKUP_BEATS, SEC_PER_BEAT
    from examples.rondo_practice import full_melody
    return Piece("Rondo alla Turca", full_melody(), SEC_PER_BEAT, PICKUP_BEATS, BEATS_PER_BAR)


@dataclass
class Chunk:
    index: int
    label: str
    notes: list[Note]       # piece time
    start: float            # piece time where the chunk starts (a beat boundary)


def chunks(piece: Piece, bars_per_chunk: int = 2, window_s: float = 4.0) -> list[Chunk]:
    groups: dict[int, list[Note]] = {}
    for n in piece.melody:
        if piece.beats_per_bar:
            bar = piece.bar_beat(n.onset)[0]
            k = max(bar - 1, 0) // bars_per_chunk          # the pickup joins the first chunk
        else:
            k = int(n.onset // window_s)
        groups.setdefault(k, []).append(n)
    out = []
    for i, k in enumerate(sorted(groups)):
        ns = groups[k]
        if piece.beats_per_bar:
            first, last = k * bars_per_chunk + 1, (k + 1) * bars_per_chunk
            start = 0.0 if k == 0 else (piece.pickup_beats + (first - 1) * piece.beats_per_bar) * piece.sec_per_beat
            label = ("Pickup + bars" if k == 0 and piece.pickup_beats else "Bars") + f" {first}–{last}"
        else:
            start, label = k * window_s, f"{k * window_s:.0f}–{(k + 1) * window_s:.0f} s"
        out.append(Chunk(i, label, ns, start))
    return out


@dataclass
class Target:
    chunk: Chunk
    rung: int
    notes: list[Note]       # rebased to 0, at the rung's tempo
    factor: float           # target time = (piece time - chunk.start) * factor
    piece: Piece

    @property
    def rung_name(self) -> str:
        return RUNGS[self.rung][0]

    def to_piece_time(self, t: float) -> float:
        return t / self.factor + self.chunk.start

    def bar_beat(self, t: float) -> tuple[int, int]:
        return self.piece.bar_beat(self.to_piece_time(t))

    def where(self, t: float) -> str:
        return self.piece.where(self.to_piece_time(t))


def target(piece: Piece, chunk: Chunk, rung: int) -> Target:
    notes = [Note(n.pitch, max(0.0, n.onset - chunk.start), n.offset - chunk.start, n.velocity) for n in chunk.notes]
    name = RUNGS[rung][0]
    if name == "simplified":
        notes = thin_to_grid(drop_ornaments(notes), grid=piece.sec_per_beat / 2)
    factor = 1.0 if name == "full" else piece.bpm / SLOW_BPM
    if factor != 1.0:
        notes = slow_down(notes, factor)
    return Target(chunk, rung, notes, factor, piece)


def describe_error(e: dict, where: Callable[[float], str]) -> str:
    loc, p = where(float(e["time"])), pitch_name(int(e["pitch"]))
    if e["type"] == "wrong_pitch":
        return f"{loc}: played {pitch_name(int(e['played_pitch']))} instead of {p}"
    if e["type"] == "missed":
        return f"{loc}: skipped the {p}"
    if e["type"] == "timing":
        late = e.get("offset_ms", 0) > 0
        return f"{loc}: the {p} came {abs(e.get('offset_ms', 0))} ms {'late' if late else 'early'}"
    return f"{loc}: an extra {p} that isn't in the music"


# ---- policies ---------------------------------------------------------------

@dataclass
class State:
    chunk: int = 0
    rung: int = 0
    n_chunks: int = 1
    attempts_on_chunk: int = 0
    done: bool = False


def rules_action(state: State, rub: rubric.Rubric, rung_scores: list[int]) -> tuple[str, str]:
    """The baseline: pass -> harder; two fails in a row at this rung -> easier; else repeat."""
    if rub.total >= rubric.POLISH_AT:
        return "harder", f"scored {rub.total} (pass mark {rubric.POLISH_AT})"
    if len(rung_scores) >= 2 and all(s < rubric.CHUNK_BELOW for s in rung_scores[-2:]):
        return "easier", f"below {rubric.CHUNK_BELOW} twice in a row at this level"
    return "repeat", f"scored {rub.total}, not yet {rubric.POLISH_AT}"


def template_feedback(rub: rubric.Rubric, descs: list[str]) -> str:
    if not descs:
        return "Clean: every note right and on time."
    return "Fix these: " + "; ".join(descs[:4]) + ("; …" if len(descs) > 4 else "") + "."


class RulesPolicy:
    name = "rules"

    def decide(self, state: State, tgt: Target, rub: rubric.Rubric, descs: list[str], history: list[dict]) -> dict:
        rung_scores = [h["rubric"]["total"] for h in history if h["chunk"] == state.chunk and h["rung"] == state.rung]
        action, reason = rules_action(state, rub, rung_scores + [rub.total])
        return {"action": action, "suggested": action, "reason": reason, "feedback": template_feedback(rub, descs),
                "fallback": False, "cost_usd": 0.0}


COACH_SYSTEM = f"""You are a patient piano coach running a practice loop for a learner who plays one note at a time.
The piece is split into chunks; each chunk has levels (rungs), easiest first. After each attempt you see a rubric
score (0-100) and the mistakes, and you choose the next step:
- repeat: same chunk, same level
- easier: same chunk, one level easier
- harder: same chunk, one level harder (from the top level this moves on to the next chunk)
- next_chunk: start the next chunk at the easiest level
- finish: end the lesson (only once the last chunk is done)
Pass mark is {rubric.POLISH_AT}. Default to the suggested action. You may choose differently when the history
justifies it, e.g. repeat after a pass if the same weak spot keeps coming back, move on at a slightly lower score after
steady progress, or go easier sooner if the learner is stuck. Always explain your choice in one sentence.
Feedback: at most two short sentences, addressed to the learner, naming the bar, beat and notes to fix.
Reply with JSON only."""

COACH_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ACTIONS}, "feedback": {"type": "string"},
                   "reason": {"type": "string"}},
    "required": ["action", "feedback", "reason"],
    "additionalProperties": False,
}


class CoachPolicy:
    name = "llm"

    def __init__(self, model: str | None = None):
        self.model = model
        self.rules = RulesPolicy()

    def decide(self, state: State, tgt: Target, rub: rubric.Rubric, descs: list[str], history: list[dict]) -> dict:
        base = self.rules.decide(state, tgt, rub, descs, history)
        recent = [{"level": RUNGS[h["rung"]][0], "score": h["rubric"]["total"], "action": h["action"]}
                  for h in history if h["chunk"] == state.chunk][-5:]
        info = {
            "chunk": f"{state.chunk + 1} of {state.n_chunks} ({tgt.chunk.label})",
            "level": f"{state.rung + 1} of {len(RUNGS)}: {RUNGS[state.rung][1]}",
            "attempts_on_this_chunk": state.attempts_on_chunk,
            "rubric": {"total": rub.total, "band": rub.band, "parts": rub.parts, "worst_bars": rub.worst_bars},
            "mistakes": descs,
            "earlier_attempts_on_this_chunk": recent,
            "suggested_action": base["suggested"],
            "suggestion_reason": base["reason"],
        }
        try:
            resp = llm.chat([{"role": "system", "content": COACH_SYSTEM},
                             {"role": "user", "content": json.dumps(info, indent=1)}],
                            response_format={"type": "json_schema",
                                             "json_schema": {"name": "coach", "strict": True, "schema": COACH_SCHEMA}},
                            model=self.model, max_tokens=4000, tag="coach")
            out = json.loads(llm.text_of(resp))
            if out.get("action") not in ACTIONS:
                raise ValueError(f"invalid action {out.get('action')!r}")
            return {"action": out["action"], "suggested": base["suggested"], "reason": out["reason"],
                    "feedback": out["feedback"], "fallback": False, "cost_usd": resp["cost_usd"]}
        except llm.BudgetExceeded:
            raise
        except Exception as e:  # bad JSON / invalid action / API error -> follow the rules
            base.update(fallback=True, reason=f"coach failed ({type(e).__name__}: {e}); used the rules: {base['reason']}")
            return base


# ---- the controller ---------------------------------------------------------

def guard(state: State, action: str) -> tuple[str, str]:
    """Turn an impossible action into the nearest possible one; enforce the attempt cap."""
    top, last = len(RUNGS) - 1, state.chunk == state.n_chunks - 1
    note = ""
    if action == "harder" and state.rung == top:
        action, note = "next_chunk", "top level passed"
    if action == "easier" and state.rung == 0:
        action, note = "repeat", "already at the easiest level"
    if action == "finish" and not (last and state.rung == top):
        action, note = "repeat", "can't finish before the last chunk's top level"
    if action not in ("next_chunk", "finish") and state.attempts_on_chunk >= MAX_ATTEMPTS_PER_CHUNK:
        action, note = "next_chunk", f"attempt cap ({MAX_ATTEMPTS_PER_CHUNK} per chunk)"
    if action == "next_chunk" and last:
        action, note = "finish", note or "last chunk"
    return action, note


class Lesson:
    def __init__(self, piece: Piece, learner, policy, lesson_id: str | None = None, log_dir: Path | None = LESSON_DIR):
        self.piece, self.learner, self.policy = piece, learner, policy
        self.chunks = chunks(piece)
        self.state = State(n_chunks=len(self.chunks))
        self.rows: list[dict] = []
        self.audio: list[tuple[list[Note], list[Note]]] = []   # (target, attempt) per row, for playback
        self.id = lesson_id or f"{getattr(learner, 'level', 'learner')}_{policy.name}_" \
                               f"s{getattr(learner, 'seed', 0)}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.log = (log_dir / f"{self.id}.jsonl") if log_dir else None
        self.stopped: str | None = None

    @property
    def done(self) -> bool:
        return self.state.done or self.stopped is not None

    def step(self) -> dict:
        s = self.state
        tgt = target(self.piece, self.chunks[s.chunk], s.rung)
        attempt = self.learner.play(tgt, s.attempts_on_chunk)
        errors = find_errors(tgt.notes, attempt)
        rub = rubric.score(tgt.notes, errors, attempt, tgt.bar_beat)
        descs = [describe_error(e, tgt.where) for e in errors]
        s.attempts_on_chunk += 1
        try:
            d = self.policy.decide(s, tgt, rub, descs, self.rows)
        except llm.BudgetExceeded as e:
            self.stopped = str(e)
            raise
        action, note = guard(s, d["action"])
        row = {"attempt": len(self.rows) + 1, "chunk": s.chunk, "chunk_label": tgt.chunk.label, "rung": s.rung,
               "rung_name": tgt.rung_name, "rubric": rub.to_dict(), "errors": errors, "mistakes": descs,
               "chosen": d["action"], "action": action, "guard": note, "suggested": d["suggested"],
               "fallback": d["fallback"], "feedback": d["feedback"], "reason": d["reason"], "cost_usd": d["cost_usd"]}
        self._apply(action)
        self.rows.append(row)
        self.audio.append((tgt.notes, attempt))
        self._write(row)
        if self.done:
            self._write({"summary": self.summary()})
        return row

    def _apply(self, action: str) -> None:
        s = self.state
        if action == "easier":
            s.rung -= 1
        elif action == "harder":
            s.rung += 1
        elif action == "next_chunk":
            s.chunk, s.rung, s.attempts_on_chunk = s.chunk + 1, 0, 0
        elif action == "finish":
            s.done = True

    def run(self, max_steps: int = 200) -> dict:
        for _ in range(max_steps):
            if self.done:
                break
            try:
                self.step()
            except llm.BudgetExceeded:
                break
        if not self.done:
            self.stopped = self.stopped or f"step limit ({max_steps})"
            self._write({"summary": self.summary()})
        return self.summary()

    def summary(self) -> dict:
        per_chunk = {}
        for c in self.chunks:
            rows = [r for r in self.rows if r["chunk"] == c.index]
            full = [r["rubric"]["total"] for r in rows if r["rung_name"] == "full"]
            per_chunk[c.label] = {"attempts": len(rows), "final_full_score": full[-1] if full else None,
                                  "passed": bool(full) and full[-1] >= rubric.POLISH_AT}
        wasted = sum(1 for a, b in zip(self.rows, self.rows[1:])
                     if a["action"] == "repeat" and b["rubric"]["total"] <= a["rubric"]["total"])
        fulls = [v["final_full_score"] for v in per_chunk.values() if v["final_full_score"] is not None]
        return {
            "lesson_id": self.id, "policy": self.policy.name, "level": getattr(self.learner, "level", None),
            "seed": getattr(self.learner, "seed", None), "finished": self.state.done, "stopped": self.stopped,
            "attempts": len(self.rows), "chunks_passed": sum(v["passed"] for v in per_chunk.values()),
            "n_chunks": len(self.chunks), "all_passed": all(v["passed"] for v in per_chunk.values()),
            "mean_final_full_score": round(sum(fulls) / len(fulls), 1) if fulls else None,
            "wasted_repeats": wasted, "coach_overrides": sum(r["chosen"] != r["suggested"] for r in self.rows),
            "fallbacks": sum(r["fallback"] for r in self.rows),
            "cost_usd": round(sum(r["cost_usd"] for r in self.rows), 4), "per_chunk": per_chunk,
        }

    def _write(self, row: dict) -> None:
        if not self.log:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")
