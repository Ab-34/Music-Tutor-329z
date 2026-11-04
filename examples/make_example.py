"""Generate the bundled quick-start example (no dataset download needed).

    examples/ref.mid   a short two-hand piano exercise (melody + left-hand chords)
    examples/perf.mid  the same piece with 3 known mistakes (see examples/perf_errors.json)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from data.make_perturbations import perturb
from tutor.notes import Note, save_midi

HERE = Path(__file__).parent
BEAT = 0.5  # 120 bpm


def piece() -> list[Note]:
    melody = [64, 62, 60, 62, 64, 64, 64, 62, 62, 62, 64, 67, 67, 64, 62, 60, 62, 64, 64, 64, 64, 62, 62, 64, 62, 60]
    chords = [[48, 52, 55], [43, 47, 50], [48, 52, 55], [41, 45, 48], [48, 52, 55], [43, 47, 50], [48, 52, 55]]
    notes = [Note(p, i * BEAT, (i + 0.9) * BEAT, 80) for i, p in enumerate(melody)]
    for b, ch in enumerate(chords):
        notes += [Note(p, b * 4 * BEAT, (b * 4 + 3.8) * BEAT, 60) for p in ch]
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def main() -> None:
    ref = piece()
    perf, labels = perturb(ref, 3, random.Random(7))
    save_midi(ref, HERE / "ref.mid")
    save_midi(perf, HERE / "perf.mid")
    (HERE / "perf_errors.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"wrote {HERE}/ref.mid, perf.mid; true errors: {labels}")


if __name__ == "__main__":
    main()
