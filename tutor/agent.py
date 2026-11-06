"""The orchestrator: an LLM tool-use loop over the tutor's tools.

The agent decides HOW to listen (which transcription backend, which segments,
what thresholds; whether to self-check and re-transcribe), then which error
candidates are real, then writes feedback that uses the learner's history.
It ends by calling `submit_report`.

We write the loop by hand (instead of the SDK tool runner) so every tool call
is logged into a trace -- the traces are what the error analysis is built on.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tutor import llm
from tutor.notes import Note, load_midi
from tutor.tools import memory
from tutor.tools.align import find_errors
from tutor.tools.score_text import notes_to_text
from tutor.tools.transcribe import BACKENDS, transcribe

SYSTEM = """You are a piano tutor's listening assistant. You compare a learner's \
performance against a reference score and report the learner's mistakes.

Error types: wrong_pitch (played a different note), missed (skipped a note), \
extra (played a note not in the score), timing (right note, onset off by >40 ms).
Times are seconds on the reference clock.

How to work:
- Transcribe the performance. You choose the backend and settings. Automatic \
transcription makes its own mistakes (dropped notes in fast passages or dense \
chords, spurious octave/harmonic notes), so do not trust every candidate blindly.
- If the input is audio, use self_check to find segments where the transcription \
is unreliable, and consider re-transcribing those segments with different thresholds.
- Use find_error_candidates, then inspect the score and performance text around \
suspicious candidates to decide which are real learner errors.
- Read the learner's mistake log and mention recurring problems in feedback.
- Finish by calling submit_report exactly once. Only include errors you believe \
are real. Keep feedback short, concrete and encouraging (times, note names, what to practise)."""

ERROR_ITEM = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["wrong_pitch", "missed", "extra", "timing"]},
        "time": {"type": "number"},
        "pitch": {"type": "integer", "description": "MIDI pitch from the reference (for extra: the played pitch)"},
        "note": {"type": "string", "description": "optional short explanation"},
    },
    "required": ["type", "time", "pitch"],
}

TOOLS = [
    {
        "name": "transcribe_performance",
        "description": "Transcribe the learner's performance into notes (replaces the current transcription for the "
                       "given window, or the whole piece if no window). Returns a note count and the transcription as text.",
        "input_schema": {"type": "object", "properties": {
            "backend": {"type": "string", "enum": BACKENDS},
            "start": {"type": "number"}, "end": {"type": "number"},
            "onset_threshold": {"type": "number", "description": "0-1, lower = more notes (default 0.5)"},
            "frame_threshold": {"type": "number", "description": "0-1 (default 0.3)"},
        }, "required": ["backend"]},
    },
    {
        "name": "self_check",
        "description": "Resynthesize the current transcription and compare it to the recording per 2 s segment. "
                       "High chroma_distance (> ~0.15) means the transcription is probably wrong there. Audio input only.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_error_candidates",
        "description": "Align the current transcription to the score and list candidate errors.",
        "input_schema": {"type": "object", "properties": {
            "timing_tol_ms": {"type": "number", "description": "default 40"},
            "match_tol_ms": {"type": "number", "description": "default 300"},
        }},
    },
    {
        "name": "get_score_text",
        "description": "Reference score notes in a time window, as text (NoteName@seconds).",
        "input_schema": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}}},
    },
    {
        "name": "get_performance_text",
        "description": "Transcribed performance notes in a time window, as text.",
        "input_schema": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}}},
    },
    {
        "name": "read_mistake_log",
        "description": "The learner's past mistakes (all pieces, or this piece only).",
        "input_schema": {"type": "object", "properties": {"this_piece_only": {"type": "boolean"}}},
    },
    {
        "name": "submit_report",
        "description": "Final answer: the learner's real errors and feedback text. Call exactly once, last.",
        "input_schema": {"type": "object", "properties": {
            "errors": {"type": "array", "items": ERROR_ITEM},
            "feedback": {"type": "string"},
        }, "required": ["errors", "feedback"]},
    },
]


@dataclass
class Session:
    ref_path: str
    perf_path: str
    learner: str = "default"
    piece: str = ""
    ref: list[Note] = field(default_factory=list)
    perf: list[Note] | None = None
    report: dict | None = None
    trace: list[dict] = field(default_factory=list)

    @property
    def is_audio(self) -> bool:
        return Path(self.perf_path).suffix.lower() not in {".mid", ".midi"}

    # ---- tool implementations -------------------------------------------------
    def transcribe_performance(self, backend, start=None, end=None, onset_threshold=0.5, frame_threshold=0.3):
        if not self.is_audio:
            backend = "oracle"
        new = transcribe(self.perf_path, backend=backend, start=start or 0.0, end=end,
                         onset_threshold=onset_threshold, frame_threshold=frame_threshold)
        if self.perf is not None and (start is not None or end is not None):
            lo, hi = start or 0.0, end if end is not None else 1e9
            self.perf = sorted([n for n in self.perf if not lo <= n.onset < hi] + new, key=lambda n: n.onset)
        else:
            self.perf = new
        return {"backend": backend, "n_notes": len(self.perf), "text": notes_to_text(self.perf)}

    def self_check(self):
        if not self.is_audio:
            return {"info": "performance is MIDI; nothing to self-check"}
        self._need_perf()
        from tutor.tools.self_check import self_check
        return {"segments": self_check(self.perf_path, self.perf)}

    def find_error_candidates(self, timing_tol_ms=40, match_tol_ms=300):
        self._need_perf()
        errs = find_errors(self.ref, self.perf, match_tol=match_tol_ms / 1000, timing_tol=timing_tol_ms / 1000)
        return {"n": len(errs), "candidates": errs}

    def get_score_text(self, start=0.0, end=None):
        return {"text": notes_to_text(self.ref, start, end)}

    def get_performance_text(self, start=0.0, end=None):
        self._need_perf()
        return {"text": notes_to_text(self.perf, start, end)}

    def read_mistake_log(self, this_piece_only=False):
        return {"log": memory.read(self.learner, self.piece if this_piece_only else None)}

    def submit_report(self, errors, feedback):
        self.report = {"errors": errors, "feedback": feedback}
        return {"ok": True}

    def _need_perf(self):
        if self.perf is None:
            raise RuntimeError("no transcription yet; call transcribe_performance first")


# OpenAI-style function tools, as OpenRouter expects them.
OAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
TOOL_NAMES = {t["name"] for t in TOOLS}

# Per-run spending cap (USD), so one runaway loop can't eat the whole budget.
RUN_BUDGET_USD = float(os.environ.get("TUTOR_RUN_BUDGET_USD", "0.50"))


def run_agent(ref_path: str, perf_path: str, learner: str = "default", piece: str | None = None,
              max_turns: int = 20, write_memory: bool = True, on_event=None) -> dict:
    """Run one assessment. `on_event(dict)`, if given, is called after every model call
    ({"kind": "llm", ...}) and every tool call ({"kind": "tool", ...}) -- the GUI uses it
    to show the trace live. It does not change the result."""
    emit = on_event or (lambda ev: None)
    s = Session(ref_path, perf_path, learner, piece or Path(ref_path).parent.name or Path(ref_path).stem)
    s.ref = load_midi(ref_path)
    kind = "audio recording" if s.is_audio else "MIDI performance (already symbolic; transcription is exact)"
    duration = max((n.offset for n in s.ref), default=0)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                 f"Learner: {learner}. Piece: {s.piece} ({duration:.1f} s, {len(s.ref)} notes). "
                 f"Performance is a {kind}. Assess it and submit your report."}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cached_calls": 0}
    stop_error = None

    for turn in range(max_turns):
        if usage["cost_usd"] >= RUN_BUDGET_USD:
            stop_error = f"run budget of ${RUN_BUDGET_USD:.2f} reached (TUTOR_RUN_BUDGET_USD)"
            break
        resp = llm.chat(messages, tools=OAI_TOOLS, tag="agent")
        u = resp.get("usage") or {}
        usage["input_tokens"] += u.get("prompt_tokens", 0)
        usage["output_tokens"] += u.get("completion_tokens", 0)
        usage["cost_usd"] += resp["cost_usd"]
        usage["cached_calls"] += resp["cached"]
        msg = llm.message_of(resp)
        emit({"kind": "llm", "turn": turn + 1, "cost_usd": resp["cost_usd"], "cached": resp["cached"],
              "total_cost_usd": usage["cost_usd"], "text": (msg.get("content") or "")[:500]})
        if msg.get("refusal"):
            s.trace.append({"event": "refusal"})
            break
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         **({"tool_calls": [{"id": c["id"], "type": "function", "function": c["function"]}
                                            for c in calls]} if calls else {})})
        if not calls:
            if s.report is None:  # nudge if it answered in prose
                messages.append({"role": "user", "content": "Please call submit_report with your findings."})
                continue
            break
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                if name not in TOOL_NAMES:
                    raise ValueError(f"unknown tool {name!r}")
                out = json.dumps(getattr(s, name)(**args))
                s.trace.append({"tool": name, "input": args, "ok": True})
                emit({"kind": "tool", "name": name, "input": args, "ok": True, "output_preview": out[:600]})
            except Exception as e:  # report tool errors back to the model
                out = f"ERROR: {e}"
                s.trace.append({"tool": name, "input": c["function"].get("arguments"), "ok": False, "error": str(e)})
                emit({"kind": "tool", "name": name, "input": c["function"].get("arguments"), "ok": False,
                      "error": str(e)})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        if s.report is not None:
            break

    report = s.report or {"errors": [], "feedback": "", "error": stop_error or "agent did not submit a report"}
    if write_memory and report.get("errors"):
        report["memory"] = memory.write(learner, s.piece, report["errors"])
    report["trace"] = s.trace
    report["usage"] = usage
    return report
