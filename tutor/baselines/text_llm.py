"""B2: one LLM call, no tools. Gets score and transcription as text, returns errors as JSON."""
from __future__ import annotations

import json

from tutor import llm
from tutor.notes import load_midi
from tutor.tools.score_text import notes_to_text
from tutor.tools.transcribe import transcribe

SCHEMA = {
    "type": "object",
    "properties": {
        "errors": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["wrong_pitch", "missed", "extra", "timing"]},
                "time": {"type": "number"},
                "pitch": {"type": "integer"},
            },
            "required": ["type", "time", "pitch"],
            "additionalProperties": False,
        }},
        "feedback": {"type": "string"},
    },
    "required": ["errors", "feedback"],
    "additionalProperties": False,
}

PROMPT = """Compare the learner's performance to the reference score and list the learner's mistakes.
Types: wrong_pitch, missed, extra, timing (onset off by >40 ms). Use reference-clock times in seconds and
MIDI pitch numbers of the reference note (for extra: the played note). Note names use C4 = MIDI 60.

REFERENCE SCORE:
{ref}

PERFORMANCE (automatic transcription, may contain transcription errors):
{perf}"""


def run(ref_path: str, perf_path: str, backend: str = "basic_pitch", **_) -> dict:
    ref = load_midi(ref_path)
    perf = transcribe(perf_path, backend=backend)
    resp = llm.create(
        messages=[{"role": "user", "content": PROMPT.format(ref=notes_to_text(ref), perf=notes_to_text(perf))}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if resp.stop_reason == "refusal":
        return {"errors": [], "feedback": "", "error": "refusal"}
    out = json.loads(llm.text_of(resp))
    out["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return out
