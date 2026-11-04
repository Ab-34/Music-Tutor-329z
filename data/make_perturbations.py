"""Build the perturbation-detection eval set.

For each reference MIDI excerpt, inject a known number of errors and save:
    data/cases/<case_id>/ref.mid
    data/cases/<case_id>/perf.mid      (ground-truth perturbed performance)
    data/cases/<case_id>/perf.wav      (only with --audio; synthesized)
and one manifest line per case with the ground-truth error labels.

Severity = number of injected errors. Severity 0 cases are clean; they measure
the false-positive rate (the "noise floor" of transcription + alignment).

Usage:
    python -m data.make_perturbations --midi-dir data/raw/maestro --n-excerpts 30 \
        --severities 0 1 3 6 --per-severity 3 --audio
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pretty_midi

from tutor.notes import Note, load_midi, save_midi

ERROR_TYPES = ["wrong_pitch", "missed", "extra", "timing"]
PITCH_DELTAS = [-12, -2, -1, 1, 2, 12]       # semitones (±1 is subtle, ±12 is easy)
TIMING_DELTAS_MS = [50, 100, 200]


def perturb(ref: list[Note], k: int, rng: random.Random) -> tuple[list[Note], list[dict]]:
    perf = [Note(n.pitch, n.onset, n.offset, n.velocity) for n in ref]
    labels: list[dict] = []
    touched: set[int] = set()
    end_t = max(n.offset for n in ref)
    for _ in range(k):
        etype = rng.choice(ERROR_TYPES)
        if etype == "extra":
            t = rng.uniform(0.2, end_t - 0.2)
            nearby = [n.pitch for n in ref if abs(n.onset - t) < 0.5] or [60]
            pitch = min(108, max(21, rng.choice(nearby) + rng.choice([-3, -1, 2, 4])))
            perf.append(Note(pitch, t, t + 0.2, 70))
            labels.append({"type": "extra", "time": round(t, 3), "pitch": pitch})
            continue
        free = [i for i in range(len(ref)) if i not in touched]
        if not free:
            break
        i = rng.choice(free)
        touched.add(i)
        n = perf[i]
        if etype == "wrong_pitch":
            d = rng.choice(PITCH_DELTAS)
            new = min(108, max(21, n.pitch + d))
            labels.append({"type": "wrong_pitch", "time": round(n.onset, 3), "pitch": n.pitch,
                           "played_pitch": new, "delta": d})
            n.pitch = new
        elif etype == "missed":
            labels.append({"type": "missed", "time": round(n.onset, 3), "pitch": n.pitch})
            n.velocity = -1  # mark for deletion
        elif etype == "timing":
            d = rng.choice(TIMING_DELTAS_MS) * rng.choice([-1, 1]) / 1000
            labels.append({"type": "timing", "time": round(n.onset, 3), "pitch": n.pitch,
                           "offset_ms": int(d * 1000)})
            n.onset = max(0.0, n.onset + d)
            n.offset = n.offset + d
    perf = [n for n in perf if n.velocity >= 0]
    perf.sort(key=lambda n: (n.onset, n.pitch))
    labels.sort(key=lambda e: e["time"])
    return perf, labels


def excerpts(midi_dir: Path, n: int, length: float, rng: random.Random):
    files = sorted(list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi")))
    rng.shuffle(files)
    for f in files[:n]:
        total = pretty_midi.PrettyMIDI(str(f)).get_end_time()
        start = rng.uniform(0, max(0.0, total - length))
        notes = load_midi(f, start, start + length)
        if len(notes) >= 10:
            yield f.stem, round(start, 2), notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-dir", type=Path, default=Path("examples"))
    ap.add_argument("--out", type=Path, default=Path("data/cases"))
    ap.add_argument("--n-excerpts", type=int, default=30)
    ap.add_argument("--length", type=float, default=20.0)
    ap.add_argument("--severities", type=int, nargs="+", default=[0, 1, 3, 6])
    ap.add_argument("--per-severity", type=int, default=3)
    ap.add_argument("--audio", action="store_true", help="also synthesize perf.wav (needs audio extra)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    count = 0
    with manifest.open("w") as mf:
        for stem, start, ref in excerpts(args.midi_dir, args.n_excerpts, args.length, rng):
            for sev in args.severities:
                for rep in range(args.per_severity if sev > 0 else 1):
                    cid = f"{stem[:40]}_{int(start)}_s{sev}_{rep}"
                    d = args.out / cid
                    perf, labels = perturb(ref, sev, rng)
                    save_midi(ref, d / "ref.mid")
                    save_midi(perf, d / "perf.mid")
                    row = {"case_id": cid, "source": stem, "start": start, "severity": sev,
                           "ref": str(d / "ref.mid"), "perf_midi": str(d / "perf.mid"), "errors": labels}
                    if args.audio:
                        from tutor.tools.synth import synthesize, write_wav
                        write_wav(synthesize(perf), d / "perf.wav")
                        row["perf_audio"] = str(d / "perf.wav")
                    mf.write(json.dumps(row) + "\n")
                    count += 1
    print(f"wrote {count} cases -> {manifest}")


if __name__ == "__main__":
    main()
