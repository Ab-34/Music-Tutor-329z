"""Audio -> notes.

Backends:
  - "basic_pitch": Spotify Basic Pitch (instrument-agnostic, Bittner et al. 2022).
  - "oracle":      the performance is already MIDI; read it directly. Gives an
                   upper bound that isolates assessment from transcription error.
Add a piano-specific backend (e.g. Onsets & Frames / piano_transcription_inference)
here as another branch; the agent chooses among them by name.
"""
from __future__ import annotations

from pathlib import Path

from tutor.notes import Note, load_midi

BACKENDS = ["basic_pitch", "oracle"]


def transcribe(
    path: str | Path,
    backend: str = "basic_pitch",
    start: float = 0.0,
    end: float | None = None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 58.0,
) -> list[Note]:
    path = Path(path)
    if backend == "oracle" or path.suffix.lower() in {".mid", ".midi"}:
        return load_midi(path, start, end)
    if backend == "basic_pitch":
        return _basic_pitch(path, start, end, onset_threshold, frame_threshold, min_note_ms)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def _basic_pitch(path, start, end, onset_threshold, frame_threshold, min_note_ms) -> list[Note]:
    from basic_pitch import ICASSP_2022_MODEL_PATH  # audio extra
    from basic_pitch.inference import predict

    _, midi, _ = predict(
        str(path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
    )
    notes = []
    for inst in midi.instruments:
        for n in inst.notes:
            if n.start < start or (end is not None and n.start >= end):
                continue
            notes.append(Note(n.pitch, n.start, n.end, n.velocity))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes
