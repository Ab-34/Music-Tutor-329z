"""Persistent mistake log (JSON file per learner).

Entry: {piece, time, pitch, type, count, first_seen, last_seen, note}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path("memory_store")


def _path(learner: str) -> Path:
    safe = "".join(c for c in learner if c.isalnum() or c in "-_") or "default"
    return ROOT / f"{safe}.json"


def read(learner: str, piece: str | None = None) -> list[dict]:
    p = _path(learner)
    if not p.exists():
        return []
    log = json.loads(p.read_text())
    return [e for e in log if piece is None or e["piece"] == piece]


def write(learner: str, piece: str, mistakes: list[dict]) -> dict:
    """Merge mistakes into the log; repeated (piece, type, pitch, ~time) bump the count."""
    p = _path(learner)
    p.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(p.read_text()) if p.exists() else []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    added = updated = 0
    for m in mistakes:
        hit = next((e for e in log if e["piece"] == piece and e["type"] == m["type"]
                    and e.get("pitch") == m.get("pitch") and abs(e["time"] - m["time"]) < 0.5), None)
        if hit:
            hit["count"] += 1
            hit["last_seen"] = now
            updated += 1
        else:
            log.append({"piece": piece, "time": m["time"], "pitch": m.get("pitch"), "type": m["type"],
                        "count": 1, "first_seen": now, "last_seen": now, "note": m.get("note", "")})
            added += 1
    p.write_text(json.dumps(log, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    return {"added": added, "updated": updated, "total": len(log)}


# ---- lesson progress: where the learner is in a piece ------------------------
# memory_store/<learner>.progress.json = {piece: {"assignment": {...}, "history": [...]}}

def _progress_path(learner: str) -> Path:
    return _path(learner).with_suffix(".progress.json")


def read_progress(learner: str, piece: str) -> dict:
    p = _progress_path(learner)
    data = json.loads(p.read_text()) if p.exists() else {}
    return data.get(piece, {"assignment": None, "history": []})


def write_progress(learner: str, piece: str, progress: dict) -> None:
    p = _progress_path(learner)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[piece] = progress
    p.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


def reset(learner: str, piece: str) -> None:
    """Forget this learner's progress and mistakes on one piece."""
    p = _progress_path(learner)
    if p.exists():
        data = json.loads(p.read_text())
        data.pop(piece, None)
        p.write_text(json.dumps(data, indent=2))
    q = _path(learner)
    if q.exists():
        q.write_text(json.dumps([e for e in json.loads(q.read_text()) if e["piece"] != piece], indent=2))
