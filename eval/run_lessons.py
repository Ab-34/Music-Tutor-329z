"""Run simulated Rondo lessons under one policy and summarise them by learner level.

    python -m eval.run_lessons --policy rules --seeds 5
    python -m eval.run_lessons --policy llm --levels struggling --seeds 2

Writes runs/lessons/<lesson_id>.jsonl per lesson and runs/lessons/summary_<policy>.json.
The rules policy is free; the llm policy spends budget (TUTOR_BUDGET_USD still applies).
"""
from __future__ import annotations

import argparse
import json
import statistics

from tutor import llm
from tutor.lesson import LESSON_DIR, CoachPolicy, Lesson, RulesPolicy, rondo_piece

LEVELS = ["nearly_there", "intermediate", "struggling"]


def aggregate(summaries: list[dict]) -> dict:
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else None
    out = {}
    for level in sorted({s["level"] for s in summaries}, key=lambda lv: LEVELS.index(lv) if lv in LEVELS else 99):
        ss = [s for s in summaries if s["level"] == level]
        out[level] = {
            "lessons": len(ss),
            "all_chunks_passed": round(sum(s["all_passed"] for s in ss) / len(ss), 2),
            "chunks_passed": mean([s["chunks_passed"] for s in ss]),
            "attempts": mean([s["attempts"] for s in ss]),
            "mean_final_full_score": mean([s["mean_final_full_score"] for s in ss if s["mean_final_full_score"] is not None]),
            "wasted_repeats": mean([s["wasted_repeats"] for s in ss]),
            "coach_overrides": mean([s["coach_overrides"] for s in ss]),
            "fallbacks": sum(s["fallbacks"] for s in ss),
            "cost_usd": round(sum(s["cost_usd"] for s in ss), 4),
        }
    return out


def main() -> None:
    from examples.rondo_learner import SimLearner

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["rules", "llm"], required=True)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default=None, help="OpenRouter model for the coach (default: TUTOR_MODEL)")
    args = ap.parse_args()

    piece = rondo_piece()
    summaries = []
    for level in args.levels:
        for seed in range(args.seeds):
            policy = RulesPolicy() if args.policy == "rules" else CoachPolicy(args.model)
            lesson = Lesson(piece, SimLearner(level, seed), policy)
            s = lesson.run()
            summaries.append(s)
            print(f"{level:<13} seed {seed}: {s['attempts']:>2} attempts, {s['chunks_passed']}/{s['n_chunks']} chunks "
                  f"passed, {s['wasted_repeats']} wasted, ${s['cost_usd']:.3f}"
                  + (f"  STOPPED: {s['stopped']}" if s["stopped"] else ""))
            if s["stopped"] and "Spent" in s["stopped"]:
                break
    agg = aggregate(summaries)
    LESSON_DIR.mkdir(parents=True, exist_ok=True)
    path = LESSON_DIR / f"summary_{args.policy}.json"
    path.write_text(json.dumps({"policy": args.policy, "by_level": agg, "lessons": summaries}, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"wrote {path}  (total LLM spend so far: ${llm.total_spend():.3f})")


if __name__ == "__main__":
    main()
