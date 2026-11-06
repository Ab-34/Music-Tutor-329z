"""The tutor: one agent that listens to a recording and decides what the learner does next.

    result = run_tutor_turn("chinmay", "take.wav")          # or a .mid
    result["decision"], result["feedback"], result["next_assignment"]

Every learner has a current *assignment*: a scope (the whole excerpt, or one 2-bar chunk)
and a level (beginner / intermediate / full). A recording is compared with that assignment.
The agent transcribes it, checks the transcription, aligns it, scores it with the rubric,
reads the learner's history, can preview easier or harder versions, and then calls
`finish_turn` with its feedback and the next assignment. Progress and mistakes are saved
per learner, so the next recording (or the next session) continues where this one left off.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tutor import rubric
from tutor.agent import tool_loop
from tutor.lesson import RUNGS, Chunk, Piece, Target, chunks, describe_error, rondo_piece, target
from tutor.notes import Note, pitch_name
from tutor.tools import memory
from tutor.tools.align import find_errors
from tutor.tools.score_text import notes_to_text
from tutor.tools.simplify import difficulty
from tutor.tools.transcribe import transcribe

LEVELS = {"beginner": 0, "intermediate": 1, "full": 2}
LEVEL_TEXT = {
    "beginner": "beginner version: quick runs thinned to their main notes, no grace notes, 60 bpm",
    "intermediate": "intermediate version: all the notes, slowly (60 bpm)",
    "full": "full version: all the notes at full tempo (90 bpm)",
}
DECISIONS = ["polish", "easier_version", "beginner_lesson", "repeat", "easier", "harder", "next_chunk",
             "piece_learned"]
FIRST_ASSIGNMENT = {"scope": "whole", "level": "full"}
TRANSCRIPTS = Path(__file__).resolve().parent.parent / "runs" / "tutor_sessions"


def save_transcript(learner: str, turn: dict, root: Path | None = None) -> Path:
    """One JSON file per turn: runs/tutor_sessions/<learner>/<timestamp>.json (see load_transcripts)."""
    import time
    folder = (root or TRANSCRIPTS) / "".join(c for c in learner if c.isalnum() or c in "-_")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{turn.get('started', time.strftime('%Y%m%d-%H%M%S'))}.json"
    conv = lambda x: x.item() if hasattr(x, "item") else (x.__dict__ if hasattr(x, "__dict__") else str(x))
    path.write_text(json.dumps(turn, indent=1, default=conv))
    return path


def load_transcripts(learner: str, root: Path | None = None) -> list[dict]:
    folder = (root or TRANSCRIPTS) / "".join(c for c in learner if c.isalnum() or c in "-_")
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def scope_chunk(piece: Piece, scope) -> Chunk:
    if scope in (None, "whole"):
        return Chunk(-1, "the whole excerpt (pickup + bars 1–8)", piece.melody, 0.0)
    cs = chunks(piece)
    i = int(scope)
    if not 0 <= i < len(cs):
        raise ValueError(f"chunk must be 0..{len(cs) - 1} or 'whole'")
    return cs[i]


def assignment_target(piece: Piece, a: dict) -> Target:
    if a["level"] not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}")
    return target(piece, scope_chunk(piece, a["scope"]), LEVELS[a["level"]])


def describe_assignment(piece: Piece, a: dict) -> str:
    c = scope_chunk(piece, a["scope"])
    where = c.label if a["scope"] == "whole" else f"chunk {int(a['scope']) + 1} of {len(chunks(piece))} ({c.label})"
    return f"{where}, {LEVEL_TEXT[a['level']]}"


SYSTEM = f"""You are a piano tutor. A self-taught learner is learning the opening of Mozart's Rondo alla Turca,
right hand only, one note at a time. They have an assignment (a passage at a level), they played it, and you have
their recording. Decide what they do next, the way a good teacher would.

How to work:
1. get_assignment and get_learner_history: what they were asked to play and how earlier attempts went.
2. transcribe_recording. For audio, run self_check. Automatic transcription misses very short grace notes, and it can
   add spurious notes at the very start or end, or one octave up or down. Don't blame the learner for those.
   A MIDI recording is exact.
3. compare_to_assignment for the candidate mistakes (located by bar and beat). Decide which ones are real.
4. score_performance with the mistakes you believe. The rubric gives 0-100 and a band:
   polish >= {rubric.POLISH_AT}, simplify {rubric.CHUNK_BELOW}-{rubric.POLISH_AT - 1}, chunk < {rubric.CHUNK_BELOW}.
5. Decide, then call finish_turn exactly once.

Assignments: scope is "whole" (the whole excerpt) or a chunk number 0-3 (2-bar chunks), and the level is one of:
  beginner = fewer notes (quick runs thinned to their main notes, no grace notes), slowly at 60 bpm
  intermediate = all the notes, slowly at 60 bpm
  full = all the notes at full tempo, 90 bpm The goal is the whole excerpt at full level. Use preview_version to look at a version
before you assign it.

Guidance (use judgement; say why when you deviate):
- Whole excerpt at full level:
  - polish band: decision "polish". Give precise feedback on each mistake and keep the same assignment.
    If the score is 100 or close, use "piece_learned".
  - simplify band: decision "easier_version". Assign the whole excerpt at intermediate or beginner level.
  - chunk band: decision "beginner_lesson". Assign chunk 0 at beginner level.
- Inside a chunk lesson:
  - pass (>= {rubric.POLISH_AT}): "harder" (next level), or "next_chunk" at full level (next chunk, beginner level).
    After the last chunk at full level, assign the whole excerpt at full level.
  - not yet: "repeat". If stuck below {rubric.CHUNK_BELOW} twice at this level: "easier".
- Whole excerpt at a lower level: pass -> one level up; far off -> beginner_lesson.
- Look at the history. If the same mistake keeps coming back, say so and focus on it.
Feedback goes to the learner: 2-4 short sentences, warm and concrete, naming the bar, beat and notes. Refer to a
chunk by its bars ("bars 3-4"), never by its number. Being slower than the target is fine while learning, so don't
scold slow playing
(for example "Bar 3, beat 1: you played D5, it's B5"). Say what to do next and why."""

TOOLS = [
    ("get_assignment", "What the learner was asked to play: scope, level, tempo, notes.", {}),
    ("get_learner_history", "Earlier attempts on this piece (assignment, score, decision) and recurring mistakes.", {}),
    ("transcribe_recording", "Transcribe the recording into notes. MIDI recordings are read exactly.",
     {"onset_threshold": {"type": "number", "description": "0-1, lower = more notes (default 0.5)"},
      "frame_threshold": {"type": "number", "description": "0-1 (default 0.3)"}}),
    ("self_check", "Resynthesize the transcription and compare with the recording per 2 s segment (audio only). "
                   "chroma_distance > ~0.15 means the transcription is unreliable there.", {}),
    ("compare_to_assignment", "Align the transcription to the assignment; candidate mistakes by bar and beat.",
     {"timing_tol_ms": {"type": "number", "description": "default 40"}}),
    ("score_performance", "Rubric score (0-100, parts, band) for the mistakes you believe are real. Give their ids "
                          "from compare_to_assignment; leave out to score all candidates.",
     {"mistake_ids": {"type": "array", "items": {"type": "integer"}}}),
    ("preview_version", "Look at a version before assigning it: notes, tempo, difficulty.",
     {"scope": {"type": "string", "description": "'whole' or a chunk number '0'-'3'"},
      "level": {"type": "string", "enum": list(LEVELS)}}),
    ("finish_turn", "Your decision, feedback for the learner, and the next assignment. Call exactly once, last.",
     {"decision": {"type": "string", "enum": DECISIONS},
      "feedback": {"type": "string"},
      "mistake_ids": {"type": "array", "items": {"type": "integer"}, "description": "the real mistakes"},
      "next_scope": {"type": "string", "description": "'whole' or a chunk number '0'-'3'"},
      "next_level": {"type": "string", "enum": list(LEVELS)},
      "reason": {"type": "string", "description": "one sentence: why this next step"}}),
]
REQUIRED = {"preview_version": ["scope", "level"],
            "finish_turn": ["decision", "feedback", "next_scope", "next_level", "reason"]}
OAI_TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": props, "required": REQUIRED.get(n, [])}}} for n, d, props in TOOLS]


def _scope(v):
    return "whole" if str(v).strip().lower() in ("whole", "none", "") else int(v)


@dataclass
class TutorSession:
    learner: str
    recording: str
    piece: Piece
    assignment: dict
    progress: dict
    perf: list[Note] | None = None
    candidates: list[dict] = field(default_factory=list)
    rubric: rubric.Rubric | None = None
    result: dict | None = None
    trace: list[dict] = field(default_factory=list)

    @property
    def is_audio(self) -> bool:
        return Path(self.recording).suffix.lower() not in {".mid", ".midi"}

    @property
    def tgt(self) -> Target:
        return assignment_target(self.piece, self.assignment)

    # ---- tools --------------------------------------------------------------
    def get_assignment(self):
        t = self.tgt
        return {"assignment": self.assignment, "description": describe_assignment(self.piece, self.assignment),
                "bpm": round(self.piece.bpm / t.factor), "n_notes": len(t.notes),
                "notes": notes_to_text(t.notes), "recording_kind": "audio" if self.is_audio else "MIDI (exact)"}

    def get_learner_history(self):
        hist = self.progress.get("history", [])[-8:]
        recurring = sorted(memory.read(self.learner, self.piece.name), key=lambda e: -e["count"])[:6]
        return {"attempts_so_far": len(self.progress.get("history", [])),
                "recent_attempts": [{k: h.get(k) for k in ("assignment", "score", "band", "decision")} for h in hist],
                "recurring_mistakes": [{"type": e["type"], "note": pitch_name(e["pitch"]) if e.get("pitch") else None,
                                        "where": e.get("note", ""), "times": e["count"]} for e in recurring]}

    def transcribe_recording(self, onset_threshold=0.5, frame_threshold=0.3):
        self.perf = transcribe(self.recording, backend="basic_pitch" if self.is_audio else "oracle",
                               onset_threshold=onset_threshold, frame_threshold=frame_threshold)
        return {"n_notes": len(self.perf), "text": notes_to_text(self.perf)}

    def self_check(self):
        if not self.is_audio:
            return {"info": "MIDI recording; nothing to check"}
        self._need_perf()
        from tutor.tools.self_check import self_check
        return {"segments": self_check(self.recording, self.perf)}

    def compare_to_assignment(self, timing_tol_ms=40):
        self._need_perf()
        t = self.tgt
        errs = find_errors(t.notes, self.perf, timing_tol=timing_tol_ms / 1000)
        grace = {round(n.onset, 3) for n in t.notes if n.offset - n.onset < 0.1}
        self.candidates = [dict(e, id=i, description=describe_error(e, t.where),
                                grace_note=e["type"] == "missed" and round(float(e["time"]), 3) in grace)
                           for i, e in enumerate(errs)]
        span = lambda ns: (max(n.onset for n in ns) - min(n.onset for n in ns)) if len(ns) > 1 else 0
        ratio = span(self.perf) / span(t.notes) if span(t.notes) and span(self.perf) else None
        tempo = None if ratio is None else ("about the right tempo" if abs(ratio - 1) < 0.07 else
                                            f"about {abs(ratio - 1):.0%} {'slower' if ratio > 1 else 'faster'} "
                                            "than the target tempo")
        return {"n_target_notes": len(t.notes), "n_played_notes": len(self.perf), "tempo": tempo,
                "candidates": [{k: c[k] for k in ("id", "type", "description", "grace_note")} for c in self.candidates]}

    def _pick(self, ids):
        if ids is None:
            return [c for c in self.candidates]
        ids = set(ids)
        return [c for c in self.candidates if c["id"] in ids]

    def score_performance(self, mistake_ids=None):
        t = self.tgt
        self.rubric = rubric.score(t.notes, self._pick(mistake_ids), self.perf, t.bar_beat)
        return self.rubric.to_dict()

    def preview_version(self, scope, level):
        a = {"scope": _scope(scope), "level": level}
        t = assignment_target(self.piece, a)
        return {"description": describe_assignment(self.piece, a), "n_notes": len(t.notes),
                "bpm": round(self.piece.bpm / t.factor), "notes_per_sec": difficulty(t.notes)["notes_per_sec"],
                "notes": notes_to_text(t.notes)}

    def finish_turn(self, decision, feedback, next_scope, next_level, reason, mistake_ids=None):
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        nxt = {"scope": _scope(next_scope), "level": next_level}
        assignment_target(self.piece, nxt)  # validates
        mistakes = self._pick(mistake_ids) if self.candidates else []
        if self.rubric is None or mistake_ids is not None:
            t = self.tgt
            self.rubric = rubric.score(t.notes, mistakes, self.perf, t.bar_beat)
        self.result = {"decision": decision, "feedback": feedback, "reason": reason, "mistakes": mistakes,
                       "next_assignment": nxt}
        return {"ok": True}

    def _need_perf(self):
        if self.perf is None:
            raise RuntimeError("no transcription yet; call transcribe_recording first")


def current_assignment(learner: str, piece: Piece | None = None) -> dict:
    piece = piece or rondo_piece()
    return memory.read_progress(learner, piece.name).get("assignment") or dict(FIRST_ASSIGNMENT)


def run_tutor_turn(learner: str, recording: str, piece: Piece | None = None, on_event=None,
                   max_turns: int = 16, save: bool = True) -> dict:
    import time
    started = time.strftime("%Y%m%d-%H%M%S")
    piece = piece or rondo_piece()
    progress = memory.read_progress(learner, piece.name)
    assignment = progress.get("assignment") or dict(FIRST_ASSIGNMENT)
    s = TutorSession(learner, recording, piece, assignment, progress)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Learner: {learner}. Their assignment: "
                                            f"{describe_assignment(piece, assignment)}. "
                                            f"They just sent a {'recording' if s.is_audio else 'MIDI take'}. "
                                            "Listen, decide, and call finish_turn."}]
    usage, stop = tool_loop(messages, OAI_TOOLS, s, done=lambda: s.result is not None,
                            nudge="Please call finish_turn with your decision.", tag="tutor",
                            max_turns=max_turns, emit=on_event)
    out = {"assignment": assignment, "assignment_text": describe_assignment(piece, assignment),
           "trace": s.trace, "usage": usage, "perf": s.perf, "target": s.tgt.notes if s.perf is not None else None}
    if s.result is None:
        out.update(error=stop or "the tutor did not finish its turn", decision=None, next_assignment=assignment)
        if save:
            out["transcript"] = str(save_transcript(learner, {
                "learner": learner, "piece": piece.name, "started": started, "recording": str(recording),
                "assignment": assignment, "error": out["error"], "usage": usage, "messages": messages,
                "trace": s.trace}))
        return out
    out.update(s.result, rubric=s.rubric.to_dict() if s.rubric else None,
               next_assignment_text=describe_assignment(piece, s.result["next_assignment"]))
    if save:
        mistakes = [{**m, "note": m["description"]} for m in s.result["mistakes"]]
        if mistakes:  # mistake log keyed on piece time so chunks and whole-piece attempts line up
            t = s.tgt
            memory.write(learner, piece.name, [dict(m, time=round(t.to_piece_time(float(m["time"])), 3))
                                               for m in mistakes])
        progress.setdefault("history", []).append({
            "assignment": assignment, "score": out["rubric"]["total"] if out["rubric"] else None,
            "band": out["rubric"]["band"] if out["rubric"] else None, "decision": s.result["decision"],
            "feedback": s.result["feedback"], "reason": s.result["reason"],
            "next_assignment": s.result["next_assignment"], "recording": str(recording)})
        progress["assignment"] = s.result["next_assignment"]
        memory.write_progress(learner, piece.name, progress)
    if save:
        out["transcript"] = str(save_transcript(learner, {
            "learner": learner, "piece": piece.name, "started": started, "recording": str(recording),
            "attempt": len(progress.get("history", [])), "assignment": assignment,
            "decision": out.get("decision"), "rubric": out.get("rubric"), "feedback": out.get("feedback"),
            "reason": out.get("reason"), "next_assignment": out.get("next_assignment"),
            "mistakes": out.get("mistakes"), "error": out.get("error"), "usage": usage,
            "transcription": [[n.pitch, round(n.onset, 3), round(n.offset, 3)] for n in (s.perf or [])],
            "messages": messages, "trace": s.trace}))
    return out
