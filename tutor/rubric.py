"""Performance rubric: a deterministic 0-100 score from the aligned errors, plus a band.

    r = score(target_notes, errors, perf=perf_notes, bar_beat=lambda t: (bar, beat))
    r.total, r.parts, r.band      # e.g. 68, {"right_notes": 81, ...}, "simplify"

Each part loses all its points once TOLERANCE of the notes (at least MIN_TOLERANCE notes) have
that kind of mistake: "1 - errors / notes" alone is far too lenient, since even a poor take
plays most notes right. The bands decide what the tutor does next:
    polish   (>= POLISH_AT)    precise note-by-note feedback
    simplify (>= CHUNK_BELOW)  a simplified version of the same passage
    chunk    (<  CHUNK_BELOW)  break the piece into chunks and start a practice loop
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

from tutor.notes import Note

POLISH_AT = 85
CHUNK_BELOW = 50
TOLERANCE = 0.05       # a part hits 0 when 5% of the notes have that kind of mistake...
MIN_TOLERANCE = 2      # ...but never fewer than 2 notes (short passages)

# part name -> (error type, weight)
PARTS = {
    "right_notes": ("wrong_pitch", 0.35),
    "complete": ("missed", 0.30),
    "rhythm": ("timing", 0.20),
    "clean": ("extra", 0.15),
}
PART_LABELS = {"right_notes": "Right notes", "complete": "Complete", "rhythm": "Rhythm", "clean": "Clean"}
BAND_ACTIONS = {
    "polish": "Precise feedback on each remaining mistake",
    "simplify": "Practise a simplified version of this passage",
    "chunk": "Break the piece into small chunks and learn them one at a time",
}


@dataclass
class Rubric:
    total: int
    parts: dict[str, int]
    band: str
    counts: dict[str, int]
    n_notes: int
    tempo_ratio: float | None = None
    worst_bars: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = ", ".join(f"{PART_LABELS[k].lower()} {v}" for k, v in self.parts.items())
        return f"{self.total}/100 ({self.band}): {parts}"


def band(total: float) -> str:
    if total >= POLISH_AT:
        return "polish"
    return "simplify" if total >= CHUNK_BELOW else "chunk"


def score(target: list[Note], errors: list[dict], perf: list[Note] | None = None,
          bar_beat: Callable[[float], tuple[int, int]] | None = None) -> Rubric:
    n = max(len(target), 1)
    allowed = max(TOLERANCE * n, MIN_TOLERANCE)
    counts = Counter(e["type"] for e in errors)
    parts = {name: round(100 * max(0.0, 1 - counts[etype] / allowed)) for name, (etype, _) in PARTS.items()}
    total = round(sum(PARTS[name][1] * parts[name] for name in PARTS))

    tempo = None
    if perf and len(perf) > 1 and len(target) > 1:
        span = lambda ns: max(x.onset for x in ns) - min(x.onset for x in ns)
        tempo = round(span(perf) / span(target), 2) if span(target) > 0 else None

    worst = []
    if bar_beat and errors:
        per_bar: dict[int, Counter] = defaultdict(Counter)
        for e in errors:
            per_bar[bar_beat(float(e["time"]))[0]][e["type"]] += 1
        worst = [{"bar": b, "errors": sum(c.values()), "types": dict(c)}
                 for b, c in sorted(per_bar.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))][:3]
    return Rubric(total, parts, band(total), {k: counts[v] for k, (v, _) in PARTS.items()}, len(target), tempo, worst)
