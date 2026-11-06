"""Rebuild full tutor transcripts for a session recorded before transcripts were saved, and export it.

    python -m eval.rebuild_transcripts --learner Chinmay --since "2026-11-06 00:00" \
        --out experiments/2026-11-06_chinmay_rondo

How: the learner's saved history (memory_store/<learner>.progress.json) lists every attempt and
its recording. Every model reply from that session is still in the LLM response cache
(.llm_cache/), and the tools are deterministic. So we replay the cached replies, in order,
through the real tools on a scratch copy of memory. That re-creates every message, tool call and
tool result. Each rebuilt turn is checked against the decision and next assignment the history
recorded. No new LLM calls are made.

Output folder:
    README.md         what's here and how to load it
    index.json        one row per attempt (assignment, score, decision, next, cost, files)
    turns/turn_NN.json full transcript per attempt (messages, tool trace, transcription, result)
    recordings/turn_NN.wav  the learner's recordings
    state/            the learner's progress and mistake log at the end of the session
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

from tutor import llm
from tutor import tutor_agent as T
from tutor.lesson import rondo_piece
from tutor.tools import memory

ROOT = Path(__file__).resolve().parent.parent


def cached_replies(since: float, until: float) -> list[Path]:
    return sorted((p for p in llm.CACHE_DIR.glob("*/*.json") if since <= p.stat().st_mtime <= until),
                  key=lambda p: p.stat().st_mtime)


def ledger_rows(tag: str, since: float, until: float) -> list[dict]:
    return sorted((r for r in llm.read_ledger() if r.get("tag") == tag and since <= r["ts"] <= until),
                  key=lambda r: r["ts"])


def rebuild(learner: str, since: float, out: Path, piece_name: str = "Rondo alla Turca") -> dict:
    piece = rondo_piece()
    history = memory.read_progress(learner, piece_name)["history"]
    until = max(Path(h["recording"]).stat().st_mtime for h in history) + 600
    replies, costs = cached_replies(since, until), ledger_rows("tutor", since, until)
    if len(replies) != len(costs):
        raise SystemExit(f"{len(replies)} cached replies but {len(costs)} ledger rows; can't pair them")
    queue = list(zip(replies, costs))
    used: set[Path] = set()
    cache_hits = []

    def replay_chat(messages, tools=None, response_format=None, model=None, max_tokens=16000, **kw):
        # A request identical to an earlier one was answered from the cache in the live run
        # (no new cache file, no ledger row), so answer it the same way here.
        hit = llm.cache_path(llm.build_request(messages, tools=tools, response_format=response_format,
                                               model=model, max_tokens=max_tokens))
        if hit in used:
            cache_hits.append(hit.name)
            return dict(json.loads(hit.read_text()), cost_usd=0.0, cached=True)
        path, row = queue.pop(0)
        used.add(path)
        return dict(json.loads(path.read_text()), cost_usd=row["cost_usd"], cached=False)

    (out / "turns").mkdir(parents=True, exist_ok=True)
    (out / "recordings").mkdir(parents=True, exist_ok=True)
    rows, mismatches = [], []
    real_chat, real_root, real_tr = llm.chat, memory.ROOT, T.TRANSCRIPTS
    with tempfile.TemporaryDirectory() as scratch:
        llm.chat, memory.ROOT, T.TRANSCRIPTS = replay_chat, Path(scratch) / "memory", Path(scratch) / "tr"
        try:
            for i, h in enumerate(history, 1):
                n_before = len(queue)
                res = T.run_tutor_turn(learner, h["recording"], piece)
                tr = json.loads(Path(res["transcript"]).read_text())
                rec = Path(h["recording"])
                tr.update(attempt=i, recorded_at=dt.datetime.fromtimestamp(rec.stat().st_mtime).isoformat(),
                          recording=f"recordings/turn_{i:02d}{rec.suffix}", original_recording=str(rec),
                          rebuilt_from_cache=True)
                (out / "turns" / f"turn_{i:02d}.json").write_text(json.dumps(tr, indent=1, default=str))
                shutil.copy2(rec, out / "recordings" / f"turn_{i:02d}{rec.suffix}")
                same = (res.get("decision") == h["decision"] and res.get("next_assignment") == h["next_assignment"]
                        and (res.get("rubric") or {}).get("total") == h["score"])
                if not same:
                    mismatches.append(i)
                rows.append({"attempt": i, "recorded_at": tr["recorded_at"], "assignment": h["assignment"],
                             "assignment_text": T.describe_assignment(piece, h["assignment"]),
                             "score": h["score"], "band": h["band"], "decision": h["decision"],
                             "next_assignment": h["next_assignment"], "reason": h["reason"],
                             "feedback": h["feedback"], "llm_calls": n_before - len(queue),
                             "cost_usd": round(res["usage"]["cost_usd"], 4),
                             "cache_hits": res["usage"]["cached_calls"],
                             "tools": [t["tool"] for t in res["trace"] if "tool" in t],
                             "matches_history": same, "transcript": f"turns/turn_{i:02d}.json",
                             "recording": tr["recording"]})
        finally:
            llm.chat, memory.ROOT, T.TRANSCRIPTS = real_chat, real_root, real_tr
    if queue:
        mismatches.append(f"{len(queue)} cached replies left over")
    (out / "state").mkdir(exist_ok=True)
    for f in (memory._progress_path(learner), memory._path(learner)):
        if f.exists():
            shutil.copy2(f, out / "state" / f.name)
    index = {"learner": learner, "piece": piece_name, "attempts": len(rows), "llm_calls": len(replies),
             "cost_usd": round(sum(r["cost_usd"] for r in costs), 4), "cache_hits": len(cache_hits),
             "rebuilt_ok": not mismatches,
             "mismatches": mismatches, "turns": rows}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True)
    ap.add_argument("--since", required=True, help='local time, e.g. "2026-11-06 00:00"')
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    idx = rebuild(args.learner, dt.datetime.fromisoformat(args.since).timestamp(), args.out)
    print(f"{idx['attempts']} attempts, {idx['llm_calls']} paid LLM calls + {idx['cache_hits']} cache hits, "
          f"${idx['cost_usd']:.3f}, "
          f"rebuilt_ok={idx['rebuilt_ok']} {idx['mismatches'] or ''}")


if __name__ == "__main__":
    main()
