"""Rule-based simplification steps and difficulty proxies.

The agent (or a human) composes these into a ladder of easier arrangements.
Each step takes and returns a list of notes.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from tutor.notes import Note

CHORD_TOL = 0.05  # notes starting within 50 ms count as one chord


def _chords(notes: list[Note]) -> list[list[Note]]:
    groups: list[list[Note]] = []
    for n in sorted(notes, key=lambda n: n.onset):
        if groups and n.onset - groups[-1][0].onset <= CHORD_TOL:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def melody_only(notes: list[Note]) -> list[Note]:
    """Keep the highest note of every chord (skyline melody)."""
    return [max(g, key=lambda n: n.pitch) for g in _chords(notes)]


def drop_inner_voices(notes: list[Note]) -> list[Note]:
    """Keep only the top and bottom note of every chord."""
    out = []
    for g in _chords(notes):
        g = sorted(g, key=lambda n: n.pitch)
        out.extend([g[0], g[-1]] if len(g) > 1 else g)
    return out


def bass_roots(notes: list[Note], split: int = 60) -> list[Note]:
    """Melody in the right hand + only the lowest left-hand note of each chord."""
    rh = [n for n in notes if n.pitch >= split]
    lh = [min(g, key=lambda n: n.pitch) for g in _chords([n for n in notes if n.pitch < split])]
    return sorted(melody_only(rh) + lh, key=lambda n: n.onset)


def drop_ornaments(notes: list[Note], min_dur: float = 0.08) -> list[Note]:
    """Remove grace notes and other very short ornaments (shorter than `min_dur` seconds)."""
    return [n for n in notes if n.offset - n.onset >= min_dur]


def thin_to_grid(notes: list[Note], grid: float = 0.5, origin: float = 0.0) -> list[Note]:
    """For a single-note line: keep only the first note that starts in each `grid`-second cell, so
    fast runs become their main notes (e.g. four sixteenths -> two eighths with grid = an eighth).
    A kept note is held for at least one cell, but never past the next kept note."""
    kept, seen = [], set()
    for n in sorted(notes, key=lambda n: n.onset):
        cell = int((n.onset - origin) / grid + 1e-3)  # tolerate onsets rounded a hair early
        if cell not in seen:
            seen.add(cell)
            kept.append(Note(n.pitch, n.onset, n.offset, n.velocity))
    for a, b in zip(kept, kept[1:]):
        a.offset = min(max(a.offset, a.onset + grid), b.onset)
    if kept:
        kept[-1].offset = max(kept[-1].offset, kept[-1].onset + grid)
    return kept


def quantize(notes: list[Note], grid: float = 0.25) -> list[Note]:
    out = []
    for n in notes:
        on = round(n.onset / grid) * grid
        dur = max(grid, round((n.offset - n.onset) / grid) * grid)
        out.append(Note(n.pitch, on, on + dur, n.velocity))
    return out


def slow_down(notes: list[Note], factor: float = 1.5) -> list[Note]:
    return [Note(n.pitch, n.onset * factor, n.offset * factor, n.velocity) for n in notes]


STEPS = {
    "drop_inner_voices": drop_inner_voices,
    "bass_roots": bass_roots,
    "melody_only": melody_only,
    "drop_ornaments": drop_ornaments,
    "thin_to_grid": thin_to_grid,
    "quantize": quantize,
    "slow_down": slow_down,
}


def difficulty(notes: list[Note]) -> dict:
    """Cheap difficulty proxies. Should decrease monotonically down a good ladder."""
    if not notes:
        return {"notes_per_sec": 0, "mean_polyphony": 0, "pitch_range": 0, "max_leap": 0}
    dur = float(max(n.offset for n in notes) - min(n.onset for n in notes)) or 1.0
    chords = _chords(notes)
    tops = [max(g, key=lambda n: n.pitch).pitch for g in chords]
    leaps = np.abs(np.diff(tops)) if len(tops) > 1 else np.array([0])
    return {
        "notes_per_sec": round(len(notes) / dur, 2),
        "mean_polyphony": round(float(np.mean([len(g) for g in chords])), 2),
        "pitch_range": max(n.pitch for n in notes) - min(n.pitch for n in notes),
        "max_leap": int(leaps.max()),
    }
