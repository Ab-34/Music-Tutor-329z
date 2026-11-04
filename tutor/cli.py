"""Command-line entry points.

    python -m tutor.cli assess   --ref examples/ref.mid --perf examples/perf.mid [--learner you] [--no-llm]
    python -m tutor.cli simplify --ref examples/ref.mid --out runs/simplified
    python -m tutor.cli memory   --learner you
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tutor.notes import load_midi, pitch_name, save_midi


def _print_errors(errors: list[dict]) -> None:
    if not errors:
        print("  no errors found")
    for e in errors:
        extra = ""
        if e["type"] == "wrong_pitch" and "played_pitch" in e:
            extra = f" (played {pitch_name(e['played_pitch'])})"
        if e["type"] == "timing" and "offset_ms" in e:
            extra = f" ({e['offset_ms']:+d} ms)"
        print(f"  {e['time']:6.2f}s  {e['type']:<11} {pitch_name(e['pitch'])}{extra}  {e.get('note', '')}")


def cmd_assess(a) -> None:
    if a.no_llm:
        from tutor.baselines.algorithmic import run
        out = run(a.ref, a.perf, backend=a.backend)
    else:
        from tutor.agent import run_agent
        out = run_agent(a.ref, a.perf, learner=a.learner, piece=a.piece)
    print("Errors:")
    _print_errors(out["errors"])
    if out.get("feedback"):
        print("\nFeedback:\n" + out["feedback"])
    if out.get("trace"):
        print("\nTool calls: " + " -> ".join(t["tool"] for t in out["trace"] if "tool" in t))
    if out.get("error"):
        print("\nERROR:", out["error"])


def cmd_simplify(a) -> None:
    from tutor.tools.simplify import STEPS, difficulty

    notes = load_midi(a.ref)
    out = Path(a.out)
    print(f"{'original':<18} {difficulty(notes)}")
    ladder = ["drop_inner_voices", "bass_roots", "melody_only"]
    cur = notes
    for i, step in enumerate(ladder, 1):
        cur = STEPS[step](cur)
        save_midi(cur, out / f"{i}_{step}.mid")
        print(f"{step:<18} {difficulty(cur)}")
    print(f"wrote ladder to {out}/")


def cmd_memory(a) -> None:
    from tutor.tools import memory
    print(json.dumps(memory.read(a.learner), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(prog="tutor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("assess")
    p.add_argument("--ref", required=True)
    p.add_argument("--perf", required=True, help=".mid (symbolic) or .wav (audio)")
    p.add_argument("--learner", default="default")
    p.add_argument("--piece", default=None)
    p.add_argument("--backend", default="basic_pitch")
    p.add_argument("--no-llm", action="store_true", help="run the algorithmic baseline only (no API key needed)")
    p.set_defaults(fn=cmd_assess)
    p = sub.add_parser("simplify")
    p.add_argument("--ref", required=True)
    p.add_argument("--out", default="runs/simplified")
    p.set_defaults(fn=cmd_simplify)
    p = sub.add_parser("memory")
    p.add_argument("--learner", default="default")
    p.set_defaults(fn=cmd_memory)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
