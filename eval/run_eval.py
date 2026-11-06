"""Run one system over the eval manifest, save predictions, and grade them.

Usage:
    python -m eval.run_eval --system algorithmic --input midi
    python -m eval.run_eval --system agent --input audio --backend basic_pitch --limit 20

--input midi  : the performance is the perturbed MIDI (oracle transcription).
                Isolates assessment from transcription errors -> an upper bound.
--input audio : the performance is the synthesized perf.wav (realistic setting).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.grade_detection import grade
from tutor.llm import BudgetExceeded, total_spend

SYSTEMS = ["algorithmic", "text_llm", "audio_llm", "agent"]


def get_runner(name: str):
    if name == "algorithmic":
        from tutor.baselines.algorithmic import run
    elif name == "text_llm":
        from tutor.baselines.text_llm import run
    elif name == "audio_llm":
        from tutor.baselines.audio_llm import run
    elif name == "agent":
        from tutor.agent import run_agent

        def run(ref_path, perf_path, **_):
            # Each eval case is a fresh learner so memory never leaks between cases.
            return run_agent(ref_path, perf_path, learner=f"eval_{Path(ref_path).parent.name}", write_memory=False)
    else:
        raise ValueError(name)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/cases/manifest.jsonl"))
    ap.add_argument("--input", choices=["midi", "audio"], default="midi")
    ap.add_argument("--backend", default="basic_pitch")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text().splitlines()][: args.limit]
    key = "perf_midi" if args.input == "midi" else "perf_audio"
    backend = "oracle" if args.input == "midi" else args.backend
    out_dir = args.out or Path("runs") / f"{args.system}_{args.input}{'' if args.input == 'midi' else '_' + backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    run = get_runner(args.system)

    with pred_path.open("w") as f:
        for i, r in enumerate(rows):
            if key not in r:
                raise SystemExit(f"case {r['case_id']} has no {key}; rebuild dataset with --audio")
            t0 = time.time()
            try:
                out = run(r["ref"], r[key], backend=backend)
            except BudgetExceeded as e:  # stop the whole run; grade what we have
                print(f"STOPPING: {e}")
                rows = rows[:i]
                break
            except Exception as e:
                out = {"errors": [], "error": f"{type(e).__name__}: {e}"}
            out["case_id"] = r["case_id"]
            out["seconds"] = round(time.time() - t0, 2)
            f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(rows)}] {r['case_id']}: {len(out.get('errors', []))} predicted, "
                  f"{len(r['errors'])} true{'  ERROR ' + out['error'] if out.get('error') else ''}")

    sliced = out_dir / "manifest_slice.jsonl"
    sliced.write_text("".join(json.dumps(r) + "\n" for r in rows))
    scores = grade(sliced, pred_path)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))
    print(f"Total LLM spend so far (all runs): ${total_spend():.3f}  -- details: python -m tutor.llm")


if __name__ == "__main__":
    main()
