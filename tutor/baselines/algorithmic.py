"""B0: fixed pipeline, no LLM. transcribe -> align -> diff, every candidate reported."""
from __future__ import annotations

from tutor.notes import load_midi
from tutor.tools.align import find_errors
from tutor.tools.transcribe import transcribe


def run(ref_path: str, perf_path: str, backend: str = "basic_pitch", **_) -> dict:
    ref = load_midi(ref_path)
    perf = transcribe(perf_path, backend=backend)
    return {"errors": find_errors(ref, perf), "feedback": ""}
