"""Simulated Rondo lessons: the tutor agent vs. the fixed-rules baseline, by learner level.

    python -m eval.run_lessons --policy rules --seeds 5                     # free
    python -m eval.run_lessons --policy agent --levels struggling --seeds 1  # one tutor-agent turn per attempt

A simulated learner (examples/rondo_learner.py) plays whatever it is assigned and improves with
practice. `rules` drives tutor/lesson.py's Lesson controller; `agent` sends each simulated take to
the same tutor agent the GUI uses (tutor/tutor_agent.py) and follows its assignments.
Writes runs/lessons/summary_<policy>.json (+ one .jsonl per lesson).
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from tutor import llm
from tutor.lesson import LESSON_DIR, Lesson, RulesPolicy, chunks, rondo_piece

LEVELS = ["nearly_there", "intermediate", "struggling"]
MAX_ATTEMPTS = 32


def run_agent_lesson(level: str, seed: int, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Drive the tutor agent with a simulated learner until it says the piece is learned (or the cap)."""
    from examples.rondo_learner import SimLearner
    from tutor.notes import save_midi
    from tutor.tools import memory
    from tutor.tutor_agent import assignment_target, current_assignment, run_tutor_turn

    piece, sim = rondo_piece(), SimLearner(level, seed)
    learner = f"sim_{level}_s{seed}_{time.strftime('%H%M%S')}"
    memory.reset(learner, piece.name)
    log = LESSON_DIR / f"{level}_agent_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows, stopped, reps = [], None, {}
    with tempfile.TemporaryDirectory() as d:
        for k in range(max_attempts):
            a = current_assignment(learner, piece)
            tgt = assignment_target(piece, a)
            key = str(a["scope"])
            path = Path(d) / f"take{k}.mid"
            save_midi(sim.play(tgt, reps.get(key, 0)), path)
            reps[key] = reps.get(key, 0) + 1
            try:
                res = run_tutor_turn(learner, str(path), piece)
            except llm.BudgetExceeded as e:
                stopped = str(e)
                break
            row = {"attempt": k + 1, "assignment": a, "score": (res.get("rubric") or {}).get("total"),
                   "decision": res.get("decision"), "next": res.get("next_assignment"), "reason": res.get("reason"),
                   "feedback": res.get("feedback"), "error": res.get("error"),
                   "cost_usd": res["usage"]["cost_usd"]}
            rows.append(row)
            with log.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if res.get("decision") == "piece_learned":
                break
    learned = bool(rows) and rows[-1]["decision"] == "piece_learned"
    summary = {"lesson_id": log.stem, "policy": "agent", "level": level, "seed": seed, "finished": learned,
               "stopped": stopped, "attempts": len(rows), "all_passed": learned,
               "chunks_passed": None, "n_chunks": len(chunks(piece)),
               "mean_final_full_score": next((r["score"] for r in reversed(rows)
                                              if r["assignment"] == {"scope": "whole", "level": "full"}), None),
               "wasted_repeats": sum(1 for a, b in zip(rows, rows[1:]) if a["decision"] == "repeat"
                                     and (b["score"] or 0) <= (a["score"] or 0)),
               "coach_overrides": 0, "fallbacks": sum(1 for r in rows if r["error"]),
               "cost_usd": round(sum(r["cost_usd"] for r in rows), 4)}
    with log.open("a") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
    for f in (memory._path(learner), memory._progress_path(learner)):  # simulated learners leave no trace
        f.unlink(missing_ok=True)
    return summary


def aggregate(summaries: list[dict]) -> dict:
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else None
    out = {}
    for level in sorted({s["level"] for s in summaries}, key=lambda lv: LEVELS.index(lv) if lv in LEVELS else 99):
        ss = [s for s in summaries if s["level"] == level]
        out[level] = {
            "lessons": len(ss),
            "learned_whole_piece": round(sum(s["all_passed"] for s in ss) / len(ss), 2),
            "attempts": mean([s["attempts"] for s in ss]),
            "final_whole_piece_score": mean([s["mean_final_full_score"] for s in ss
                                             if s["mean_final_full_score"] is not None]),
            "wasted_repeats": mean([s["wasted_repeats"] for s in ss]),
            "failed_turns": sum(s["fallbacks"] for s in ss),
            "cost_usd": round(sum(s["cost_usd"] for s in ss), 4),
        }
    return out


def main() -> None:
    from examples.rondo_learner import SimLearner

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["rules", "agent"], required=True)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    summaries = []
    for level in args.levels:
        for seed in range(args.seeds):
            if args.policy == "rules":
                s = Lesson(rondo_piece(), SimLearner(level, seed), RulesPolicy()).run()
            else:
                s = run_agent_lesson(level, seed)
            summaries.append(s)
            print(f"{level:<13} seed {seed}: {s['attempts']:>2} attempts, finished={s['finished']}, "
                  f"${s['cost_usd']:.3f}" + (f"  STOPPED: {s['stopped']}" if s["stopped"] else ""))
    agg = aggregate(summaries)
    LESSON_DIR.mkdir(parents=True, exist_ok=True)
    path = LESSON_DIR / f"summary_{args.policy}.json"
    path.write_text(json.dumps({"policy": args.policy, "by_level": agg, "lessons": summaries}, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"wrote {path}  (total LLM spend so far: ${llm.total_spend():.3f})")


if __name__ == "__main__":
    main()
