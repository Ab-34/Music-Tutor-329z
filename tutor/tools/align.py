"""Note-level alignment of a performance against a reference, and error candidates.

v0 algorithm (good enough for synthetic perturbations; improve for real playing):
  1. Fit a global linear time map perf_t = a * ref_t + b: start from the ratio of the
     two spans, then refit from a coarse and a fine same-pitch match, so a constant
     offset or a steadily slower/faster player is absorbed.
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
            d = abs(p.onset - t) / a  # residual on the reference clock, like timing_tol
            if p.pitch == r.pitch and d <= tol:
                cands.append((d, i, j))
    cands.sort()
    used_r, used_p, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_r and j not in used_p:
            used_r.add(i); used_p.add(j); pairs.append((i, j))
    return pairs


def _initial_map(ref: list[Note], perf: list[Note]) -> tuple[float, float]:
    """First guess at the time map from the overall spans, so a uniformly slower or
    faster player doesn't drift out of the coarse matching window."""
    if len(ref) < 2 or len(perf) < 2:
        return 1.0, 0.0
    r0, r1 = min(n.onset for n in ref), max(n.onset for n in ref)
    p0, p1 = min(n.onset for n in perf), max(n.onset for n in perf)
    a = (p1 - p0) / (r1 - r0) if r1 > r0 else 1.0
    if not 0.5 < a < 2.0:
        return 1.0, 0.0
    return a, p0 - a * r0


def _fit(ref, perf, pairs, fallback):
    if len(pairs) < 4:
        return fallback
    x = np.array([ref[i].onset for i, _ in pairs])
    y = np.array([perf[j].onset for _, j in pairs])
    a, b = np.polyfit(x, y, 1)
    if not 0.5 < a < 2.0:  # implausible fit; keep the fallback scale
        a = fallback[0]
        b = float(np.median(y - a * x))
    return float(a), float(b)


def align(ref: list[Note], perf: list[Note], match_tol: float = 0.3) -> Alignment:
    a, b = _initial_map(ref, perf)
    for tol in (match_tol * 2, match_tol):  # coarse match -> fit, then refine
        a, b = _fit(ref, perf, _greedy_match(ref, perf, a, b, tol), (a, b))
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
