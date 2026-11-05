"""Demo learners for Mozart's Rondo alla Turca (K. 331, 3rd mvt), with mistakes you can hear.

Ground truth is the public-domain Mutopia score MIDI (examples/rondo/source/). We take the
pickup and the first 8 bars, both hands, play them at 90 bpm, and generate takes by learners
at four levels. Each learner has a few fixed *weak spots* they get wrong in every take (so
the agent's memory has something to notice) plus random slips that differ per take.

    python -m examples.rondo_demo        # writes examples/rondo/{ref.mid, takes/, manifest.jsonl}

Mistakes are chosen to be obvious to a listener: wrong melody notes off by a whole step or
more, skipped melody notes, clear hesitations (~250 ms late), and loud out-of-key extra notes.
"""
from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from tutor.notes import Note, pitch_name, save_midi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "rondo"
SOURCE = OUT / "source" / "KV331_3_RondoAllaTurca.mid"

SOURCE_SEC_PER_BEAT = 1.0      # the Mutopia MIDI is quarter = 60 bpm
BPM = 90
SEC_PER_BEAT = 60 / BPM
PICKUP_BEATS = 1               # B4 A4 G#4 A4 before bar 1
BEATS_PER_BAR = 2              # 2/4
N_BARS = 8
SPACING = 0.35                 # min seconds between two mistakes (reference clock)
OUT_OF_KEY = [1, 3, 6, 10]     # C#, D#, F#, A# pitch classes: clash in A minor


@dataclass(frozen=True)
class Profile:
    title: str
    blurb: str
    n_weak: int                 # weak spots repeated in every take
    slips: tuple[int, int]      # random slips per take (min, max)
    tempo: float                # >1 = slower than the score
    jitter_ms: float            # natural unevenness, below the 40 ms timing threshold
    weights: dict               # error type -> relative probability
    takes: int = 3


PROFILES: dict[str, Profile] = {
    "perfect": Profile("Perfect", "Plays exactly what's written. Checks that the system doesn't invent mistakes.",
                       0, (0, 0), 1.0, 8, {}, takes=1),
    "nearly_there": Profile("Nearly there", "Almost perfect: one or two clear slips.",
                            1, (0, 1), 1.0, 12, {"wrong_pitch": 5, "timing": 3, "missed": 2}),
    "intermediate": Profile("Intermediate", "Knows the tune but trips up a few times, a little slower.",
                            2, (2, 4), 1.1, 15, {"wrong_pitch": 4, "missed": 3, "timing": 2, "extra": 1}),
    "struggling": Profile("Struggling", "Lots of wrong and skipped notes, hesitations and stray notes, much slower.",
                          4, (8, 11), 1.3, 15, {"wrong_pitch": 4, "missed": 3, "timing": 2, "extra": 2}),
}


# ---- ground truth -----------------------------------------------------------

def load_excerpt() -> tuple[list[Note], list[Note]]:
    """(right hand, left hand) for the pickup + first N_BARS bars, at BPM, sorted."""
    with warnings.catch_warnings():  # the Mutopia file keeps its tempo/meter on track 1; harmless here
        warnings.simplefilter("ignore", RuntimeWarning)
        pm = pretty_midi.PrettyMIDI(str(SOURCE))
    end = (PICKUP_BEATS + N_BARS * BEATS_PER_BAR) * SOURCE_SEC_PER_BEAT
    scale = SEC_PER_BEAT / SOURCE_SEC_PER_BEAT
    hands = []
    for inst in pm.instruments[:2]:
        hands.append(sorted((Note(n.pitch, round(n.start * scale, 4), round(min(n.end, end) * scale, 4), n.velocity)
                             for n in inst.notes if n.start < end - 1e-6), key=lambda n: (n.onset, n.pitch)))
    return hands[0], hands[1]


def reference() -> list[Note]:
    rh, lh = load_excerpt()
    return sorted(rh + lh, key=lambda n: (n.onset, n.pitch))


def bar_beat(t: float) -> tuple[int, int]:
    """Reference time (s) -> (bar, beat). Bar 0 is the pickup."""
    b = t / SEC_PER_BEAT - PICKUP_BEATS + 1e-3  # onsets are rounded to 0.1 ms
    if b < 0:
        return 0, 1
    return int(b // BEATS_PER_BAR) + 1, int(b % BEATS_PER_BAR) + 1


def where(t: float) -> str:
    bar, beat = bar_beat(t)
    return "Pickup" if bar == 0 else f"Bar {bar}, beat {beat}"


def bar_beat_str(t: float) -> str:
    bar, beat = bar_beat(t)
    return "pickup" if bar == 0 else f"{bar}.{beat}"


def describe(e: dict) -> str:
    loc, p = where(e["time"]), pitch_name(e["pitch"])
    if e["type"] == "wrong_pitch":
        return f"{loc}: played {pitch_name(e['played_pitch'])} instead of {p}"
    if e["type"] == "missed":
        return f"{loc}: skipped the {p}"
    if e["type"] == "timing":
        return f"{loc}: hesitated, so the {p} came {e['offset_ms']} ms late"
    return f"{loc}: an extra {p} that isn't in the score"


# ---- mistakes ---------------------------------------------------------------

def _melody_idx(ref: list[Note], rh: list[Note]) -> list[int]:
    """Indices into `ref` of right-hand top-voice notes long enough to matter (no grace notes)."""
    rh_keys = {(n.pitch, n.onset) for n in rh}
    out = []
    for i, n in enumerate(ref):
        if (n.pitch, n.onset) not in rh_keys or n.offset - n.onset < 0.08:
            continue
        if any(m.onset == n.onset and m.pitch > n.pitch and (m.pitch, m.onset) in rh_keys for m in ref):
            continue  # not the top note of a right-hand dyad
        out.append(i)
    return out


def _near_pitches(ref: list[Note], t: float, win: float = SPACING) -> set[int]:
    return {n.pitch for n in ref if abs(n.onset - t) < win}


def _make_error(etype: str, ref: list[Note], melody: list[int], taken: list[float], rng: random.Random) -> dict | None:
    """One obvious mistake of `etype`, at least SPACING away from the times in `taken`. None if impossible."""
    free = lambda t: all(abs(t - u) >= SPACING for u in taken)
    if etype == "extra":
        gaps = [(ref[i].onset, ref[j].onset) for i, j in zip(melody, melody[1:]) if ref[j].onset - ref[i].onset >= 0.3]
        rng.shuffle(gaps)
        for a, b in gaps:
            t = round((a + b) / 2, 4)
            if not free(t):
                continue
            base = ref[next(i for i in melody if ref[i].onset == a)].pitch
            near = _near_pitches(ref, t)
            opts = [p for p in range(base - 5, base + 6) if p % 12 in OUT_OF_KEY and p not in near]
            if opts:
                return {"type": "extra", "time": t, "pitch": rng.choice(opts)}
        return None
    cands = melody[:]
    rng.shuffle(cands)
    for i in cands:
        n = ref[i]
        if not free(n.onset):
            continue
        if etype == "missed":
            return {"type": "missed", "time": n.onset, "pitch": n.pitch, "_i": i}
        if etype == "timing":
            if any(m.pitch == n.pitch and 0 < abs(m.onset - n.onset) < 0.6 for m in ref):
                continue  # a same-pitch neighbour would make the late note ambiguous
            ms = rng.choice([230, 250, 270])
            return {"type": "timing", "time": n.onset, "pitch": n.pitch, "offset_ms": ms, "_i": i}
        if etype == "wrong_pitch":
            near = _near_pitches(ref, n.onset)
            deltas = [d for d in (-7, -5, -4, -3, -2, 2, 3, 4, 5, 7, 12, -12)
                      if 21 <= n.pitch + d <= 108 and n.pitch + d not in near]
            if deltas:
                d = rng.choice(deltas)
                return {"type": "wrong_pitch", "time": n.onset, "pitch": n.pitch, "played_pitch": n.pitch + d,
                        "_i": i}
    return None


def _pick(profile: Profile, k: int, ref, melody, taken, rng) -> list[dict]:
    types, weights = zip(*profile.weights.items()) if profile.weights else ((), ())
    out = []
    for _ in range(k * 4):  # a few retries if a type has no room left
        if len(out) == k:
            break
        e = _make_error(rng.choices(types, weights)[0], ref, melody, taken, rng)
        if e:
            out.append(e)
            taken.append(e["time"])
    return out


def take(level: str, k: int) -> tuple[list[Note], list[Note], list[dict]]:
    """(reference, performance, labelled mistakes) for take `k` (1-based) of a learner at `level`."""
    profile = PROFILES[level]
    rh, _ = load_excerpt()
    ref = reference()
    melody = _melody_idx(ref, rh)
    taken: list[float] = []
    weak = _pick(profile, profile.n_weak, ref, melody, taken, random.Random(f"weak:{level}"))
    for e in weak:
        e["weak_spot"] = True
    rng = random.Random(f"take:{level}:{k}")
    slips = _pick(profile, rng.randint(*profile.slips), ref, melody, taken, rng)
    errors = sorted(weak + slips, key=lambda e: e["time"])

    perf = [Note(n.pitch, n.onset, n.offset, n.velocity) for n in ref]
    drop = set()
    for e in errors:
        i = e.pop("_i", None)
        if e["type"] == "missed":
            drop.add(i)
        elif e["type"] == "wrong_pitch":
            perf[i].pitch = e["played_pitch"]
        elif e["type"] == "timing":
            d = e["offset_ms"] / 1000
            perf[i].onset += d
            perf[i].offset += d
        elif e["type"] == "extra":
            perf.append(Note(e["pitch"], e["time"], e["time"] + 0.25, 100))
        e["description"] = describe(e) + ("  (weak spot: same mistake every take)" if e.get("weak_spot") else "")
    perf = [n for j, n in enumerate(perf) if j not in drop]
    jit = random.Random(f"jitter:{level}:{k}")
    for n in perf:
        dj = jit.uniform(-profile.jitter_ms, profile.jitter_ms) / 1000
        n.onset = max(0.0, (n.onset + dj) * profile.tempo)
        n.offset = (n.offset + dj) * profile.tempo
    perf.sort(key=lambda n: (n.onset, n.pitch))
    for e in errors:
        e["time"] = round(e["time"], 3)
    return ref, perf, errors


def case_id(level: str, k: int) -> str:
    return f"rondo_{level}_take{k}"


def write_all(out: Path = OUT) -> Path:
    """Write ref.mid, takes/<case>.mid + .json and manifest.jsonl (eval-manifest format) under `out`."""
    ref = reference()
    save_midi(ref, out / "ref.mid")
    rel = lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    rows = []
    for level, profile in PROFILES.items():
        for k in range(1, profile.takes + 1):
            _, perf, errors = take(level, k)
            cid = case_id(level, k)
            save_midi(perf, out / "takes" / f"{cid}.mid")
            (out / "takes" / f"{cid}.json").write_text(json.dumps(errors, indent=2) + "\n")
            rows.append({"case_id": cid, "source": "rondo_alla_turca", "start": 0, "severity": len(errors),
                         "level": level, "take": k, "learner": f"rondo_{level}", "ref": rel(out / "ref.mid"),
                         "perf_midi": rel(out / "takes" / f"{cid}.mid"), "errors": errors})
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def main() -> None:
    m = write_all()
    for r in map(json.loads, m.read_text().splitlines()):
        print(f"{r['case_id']:<28} {len(r['errors']):>2} mistakes")
        for e in r["errors"]:
            print("    " + e["description"])
    print(f"wrote {m}")


if __name__ == "__main__":
    main()
