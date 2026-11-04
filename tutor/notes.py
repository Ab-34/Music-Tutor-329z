"""Shared note representation and MIDI I/O.

Everything in the system passes around lists of `Note`. Times are seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pretty_midi


@dataclass
class Note:
    pitch: int        # MIDI pitch, 21..108 on piano
    onset: float      # seconds
    offset: float     # seconds
    velocity: int = 80

    def to_dict(self) -> dict:
        return asdict(self)


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(p: int) -> str:
    return f"{PITCH_NAMES[p % 12]}{p // 12 - 1}"


def load_midi(path: str | Path, start: float = 0.0, end: float | None = None) -> list[Note]:
    """Load all non-drum notes, optionally cropped to [start, end) and re-zeroed."""
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start < start or (end is not None and n.start >= end):
                continue
            notes.append(Note(n.pitch, n.start - start, n.end - start, n.velocity))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def save_midi(notes: list[Note], path: str | Path, program: int = 0) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.onset, max(n.offset, n.onset + 0.03)))
    pm.instruments.append(inst)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def notes_to_arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    """(intervals [N,2], pitches_hz [N]) in the format mir_eval expects."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n.onset, max(n.offset, n.onset + 0.01)] for n in notes])
    hz = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return iv, hz
