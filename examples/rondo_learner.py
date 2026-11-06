"""Simulated learners who get better with practice, for the chunked practice loop.

    learner = SimLearner("struggling", seed=0)
    attempt = learner.play(target, reps_on_chunk)      # a list of Notes

Per note, the chance of a mistake is
    base[level] * difficulty(target) * exp(-rate[level] * reps_on_chunk)
so a learner improves with each attempt at a chunk (at any level), and a harder level
(more notes per second, grace notes) brings more mistakes back. Each (level, chunk) also has
fixed weak notes that stay wrong for the first `weak_reps` attempts. The same
(level, seed, chunk, rung, reps) always gives the same attempt, so two policies face
identical learners.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from examples.rondo_demo import PROFILES
from tutor.notes import Note

MISTAKE_TYPES = {"wrong_pitch": 0.45, "missed": 0.35, "timing": 0.20}
SPACING = 0.3   # seconds between two mistakes (target clock), so each one is clear


@dataclass(frozen=True)
class Skill:
    base: float        # mistake chance per note on an average passage, first attempt
    rate: float        # how fast it drops with practice
    weak_spots: int    # fixed weak notes per chunk
    weak_reps: int     # attempts before a weak spot is fixed


SKILLS = {
    "perfect": Skill(0.0, 1.0, 0, 0),
    "nearly_there": Skill(0.04, 0.35, 1, 2),
    "intermediate": Skill(0.12, 0.30, 1, 3),
    "struggling": Skill(0.30, 0.25, 2, 5),
}


def difficulty(notes: list[Note]) -> float:
    if len(notes) < 2:
        return 0.5
    span = max(n.onset for n in notes) - min(n.onset for n in notes) or 1.0
    nps = len(notes) / span
    graces = any(n.offset - n.onset < 0.1 for n in notes)
    return min(max(nps / 2.5, 0.4), 2.5) * (1.4 if graces else 1.0)


@dataclass
class SimLearner:
    level: str
    seed: int = 0

    @property
    def skill(self) -> Skill:
        return SKILLS[self.level]

    def weak_times(self, tgt) -> list[float]:
        """Piece-time onsets of this learner's weak notes in the chunk (fixed per level + chunk)."""
        rng = random.Random(f"weak:{self.level}:{tgt.chunk.index}")
        onsets = sorted({round(n.onset, 3) for n in tgt.chunk.notes if n.offset - n.onset >= 0.1})
        return rng.sample(onsets, min(self.skill.weak_spots, len(onsets)))

    def mistake_chance(self, tgt, reps: int) -> float:
        return self.skill.base * difficulty(tgt.notes) * math.exp(-self.skill.rate * reps)

    def play(self, tgt, reps: int) -> list[Note]:
        rng = random.Random(f"{self.level}:{self.seed}:{tgt.chunk.index}:{tgt.rung}:{reps}")
        notes = [Note(n.pitch, n.onset, n.offset, n.velocity) for n in tgt.notes]
        p = self.mistake_chance(tgt, reps)
        weak = set()
        if reps < self.skill.weak_reps:
            for wt in self.weak_times(tgt):   # nearest target note to each weak spot
                j = min(range(len(notes)), key=lambda i: abs(tgt.to_piece_time(notes[i].onset) - wt))
                weak.add(j)
        drop, taken = set(), []
        for i, n in enumerate(notes):
            if n.offset - n.onset < 0.1:
                if rng.random() < p * 1.5:     # grace notes get skipped easily
                    drop.add(i)
                continue
            if i not in weak and rng.random() >= p:
                continue
            if any(abs(n.onset - t) < SPACING for t in taken):
                continue
            kind = rng.choices(list(MISTAKE_TYPES), list(MISTAKE_TYPES.values()))[0]
            near = {m.pitch for m in tgt.notes if abs(m.onset - n.onset) < 0.6}
            if kind == "timing" and any(m.pitch == n.pitch and 0 < abs(m.onset - n.onset) < 0.6 for m in tgt.notes):
                kind = "wrong_pitch"           # a same-pitch neighbour would make "late" ambiguous
            if kind == "wrong_pitch":
                opts = [n.pitch + d for d in (-5, -3, -2, 2, 3, 5) if n.pitch + d not in near]
                if not opts:
                    continue
                n.pitch = rng.choice(opts)
            elif kind == "missed":
                drop.add(i)
            else:
                d = 0.25
                n.onset += d
                n.offset += d
            taken.append(n.onset)
        out = [n for i, n in enumerate(notes) if i not in drop]
        prof = PROFILES[self.level]
        tempo = prof.tempo * (1 + 0.15 * math.exp(-0.5 * reps))   # a bit slower while still unsure
        jit = random.Random(f"jitter:{self.level}:{self.seed}:{tgt.chunk.index}:{tgt.rung}:{reps}")
        for n in out:
            dj = jit.uniform(-prof.jitter_ms, prof.jitter_ms) / 1000
            n.onset = max(0.0, (n.onset + dj) * tempo)
            n.offset = (n.offset + dj) * tempo
        return sorted(out, key=lambda n: n.onset)
