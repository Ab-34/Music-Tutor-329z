"""Note-level transcription F1 (mir_eval, onset tolerance 50 ms, pitch-only).

Usage:
    python -m eval.grade_transcription --manifest data/cases/manifest.jsonl \
        --backend basic_pitch --limit 20
Uses perf_audio as input and perf_midi as ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np

from tutor.notes import load_midi, notes_to_arrays
from tutor.tools.transcribe import transcribe


def note_f1(ref_notes, est_notes) -> dict:
    ri, rp = notes_to_arrays(ref_notes)
    ei, ep = notes_to_arrays(est_notes)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    return {"precision": p, "recall": r, "f1": f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--backend", default="basic_pitch")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines()]
    rows = [r for r in rows if r.get("perf_audio")][: args.limit]
    if not rows:
        raise SystemExit("no cases with perf_audio; rebuild the dataset with --audio")
    scores = []
    for r in rows:
        s = note_f1(load_midi(r["perf_midi"]), transcribe(r["perf_audio"], backend=args.backend))
        scores.append(s)
        print(r["case_id"], {k: round(v, 3) for k, v in s.items()})
    print("MEAN", {k: round(float(np.mean([s[k] for s in scores])), 3) for k in scores[0]})


if __name__ == "__main__":
    main()
