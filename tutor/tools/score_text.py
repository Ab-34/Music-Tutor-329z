"""Compact text view of notes so the LLM can reason over the score symbolically.

One line per time window: `[t0-t1] C4 E4 G4 | C4@0.00 E4@0.25 ...`
MAESTRO MIDI is performance MIDI without reliable bar lines, so we window by
seconds rather than measures. If you have a real score with a time signature,
switch to measure windows via pretty_midi's get_downbeats().
"""
from __future__ import annotations

from tutor.notes import Note, pitch_name


def notes_to_text(notes: list[Note], start: float = 0.0, end: float | None = None, window: float = 2.0) -> str:
    sel = [n for n in notes if n.onset >= start and (end is None or n.onset < end)]
    if not sel:
        return "(no notes)"
    stop = end if end is not None else sel[-1].onset + 1e-6
    lines = []
    t = start
    while t < stop:
        w = [n for n in sel if t <= n.onset < t + window]
        body = " ".join(f"{pitch_name(n.pitch)}@{n.onset:.2f}" for n in w) or "-"
        lines.append(f"[{t:.1f}-{t + window:.1f}s] {body}")
        t += window
    return "\n".join(lines)
