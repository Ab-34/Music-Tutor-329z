"""Self-check a transcription by resynthesizing it and comparing to the input audio.

Returns per-segment chroma distance; high distance = transcription likely wrong
there, so the agent should re-transcribe that segment with other settings or
distrust error candidates inside it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tutor.notes import Note
from tutor.tools.synth import SR, synthesize


def self_check(audio_path: str | Path, notes: list[Note], segment_s: float = 2.0) -> list[dict]:
    import librosa  # audio extra

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y_hat = synthesize(notes, sr=SR)
    n = max(len(y), len(y_hat))
    y = np.pad(y, (0, n - len(y)))
    y_hat = np.pad(y_hat, (0, n - len(y_hat)))
    hop = 512
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=hop)
    c_hat = librosa.feature.chroma_cqt(y=y_hat, sr=SR, hop_length=hop)
    frames_per_seg = int(segment_s * SR / hop)
    out = []
    for i in range(0, c.shape[1], frames_per_seg):
        a = c[:, i : i + frames_per_seg].mean(axis=1)
        b = c_hat[:, i : i + frames_per_seg].mean(axis=1)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        out.append({
            "start": round(i * hop / SR, 2),
            "end": round((i + frames_per_seg) * hop / SR, 2),
            "chroma_distance": round(1 - cos, 3),
        })
    return out
