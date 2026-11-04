"""Note-level alignment of a performance against a reference, and error candidates.

v0 algorithm (good enough for synthetic perturbations; improve for real playing):
  1. Fit a global linear time map perf_t = a * ref_t + b from a coarse
     same-pitch match, so a constant offset or steady tempo change is absorbed.
  2. Greedily match same-pitch notes whose warped onsets are within `match_tol`.
  3. Unmatched ref notes -> "missed"; unmatched perf notes -> "extra".
  4. A missed + extra pair close in time -> "wrong_pitch".
  5. Matched notes whose residual onset error > `timing_tol` -> "timing".
Real performances have local tempo variation; replacing step 1 with DTW
(e.g. over chroma or onset sequences) is the obvious next step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tutor.notes import Note


@dataclass
class Alignment:
    pairs: list[tuple[int, int]]      # (ref_idx, perf_idx)
    missed: list[int]                 # ref indices
    extra: list[int]                  # perf indices
    a: float                          # time-map scale
    b: float                          # time-map offset


def _greedy_match(ref, perf, a, b, tol):
    cands = []
    for i, r in enumerate(ref):
        t = a * r.onset + b
        for j, p in enumerate(perf):
            if p.pitch == r.pitch and abs(p.onset - t) <= tol:
                cands.append((abs(p.onset - t), i, j))
    cands.sort()
    used_r, used_p, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_r and j not in used_p:
            used_r.add(i); used_p.add(j); pairs.append((i, j))
    return pairs


def align(ref: list[Note], perf: list[Note], match_tol: float = 0.3) -> Alignment:
    pairs = _greedy_match(ref, perf, 1.0, 0.0, match_tol * 2)
    a, b = 1.0, 0.0
    if len(pairs) >= 4:
        x = np.array([ref[i].onset for i, _ in pairs])
        y = np.array([perf[j].onset for _, j in pairs])
        a, b = np.polyfit(x, y, 1)
        if not 0.5 < a < 2.0:  # implausible fit; keep identity
            a, b = 1.0, float(np.median(y - x))
    pairs = _greedy_match(ref, perf, a, b, match_tol)
    mr = {i for i, _ in pairs}
    mp = {j for _, j in pairs}
    return Alignment(
        pairs=sorted(pairs),
        missed=[i for i in range(len(ref)) if i not in mr],
        extra=[j for j in range(len(perf)) if j not in mp],
        a=float(a), b=float(b),
    )


def find_errors(
    ref: list[Note],
    perf: list[Note],
    match_tol: float = 0.3,
    timing_tol: float = 0.04,
    pair_tol: float = 0.08,
) -> list[dict]:
    """Return error candidates in a common schema (times are on the REFERENCE clock)."""
    al = align(ref, perf, match_tol)
    to_ref = lambda t: (t - al.b) / al.a
    errors: list[dict] = []

    extra = list(al.extra)
    for i in al.missed:
        r = ref[i]
        # A missed note with an extra note at about the same time = wrong pitch.
        best = None
        for j in extra:
            dt = abs(to_ref(perf[j].onset) - r.onset)
            if dt <= pair_tol and (best is None or dt < best[0]):
                best = (dt, j)
        if best:
            extra.remove(best[1])
            errors.append({"type": "wrong_pitch", "time": round(r.onset, 3), "pitch": r.pitch,
                           "played_pitch": perf[best[1]].pitch})
        else:
            errors.append({"type": "missed", "time": round(r.onset, 3), "pitch": r.pitch})
    for j in extra:
        p = perf[j]
        errors.append({"type": "extra", "time": round(to_ref(p.onset), 3), "pitch": p.pitch})
    for i, j in al.pairs:
        dt = to_ref(perf[j].onset) - ref[i].onset
        if abs(dt) > timing_tol:
            errors.append({"type": "timing", "time": round(ref[i].onset, 3), "pitch": ref[i].pitch,
                           "offset_ms": round(dt * 1000)})
    errors.sort(key=lambda e: e["time"])
    return errors
