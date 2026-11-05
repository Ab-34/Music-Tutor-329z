"""Background eval runs: launch `python -m eval.run_eval` as a subprocess, track it, stop it.

Each run lives in runs/<name>/ with job.json (pid, args, start), job.log, predictions.jsonl
and, when it finishes, scores.json. Only one job may run at a time.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from eval.grade_detection import grade
from gui.sources import MANIFEST, ROOT, read_manifest

RUNS = ROOT / "runs"
_procs: dict[int, subprocess.Popen] = {}  # children started by this server process


def _alive(pid: int) -> bool:
    proc = _procs.get(pid)
    if proc is not None:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return rows


def active_job(runs: Path = RUNS) -> Path | None:
    for jf in runs.glob("*/job.json"):
        job = json.loads(jf.read_text())
        if _alive(job["pid"]) and not (jf.parent / "scores.json").exists():
            return jf.parent
    return None


def launch(name: str, system: str, input_kind: str = "midi", backend: str = "basic_pitch",
           limit: int | None = None, manifest: Path = MANIFEST, runs: Path = RUNS) -> Path:
    from tutor.llm import total_spend

    if (running := active_job(runs)) is not None:
        raise RuntimeError(f"another eval job is still running: {running.name}")
    out = runs / name
    if (out / "predictions.jsonl").exists():
        raise RuntimeError(f"run {name!r} already exists; pick another name")
    out.mkdir(parents=True, exist_ok=True)
    n_total = len(read_manifest(manifest))
    n_total = min(n_total, limit) if limit else n_total
    cmd = [sys.executable, "-m", "eval.run_eval", "--system", system, "--input", input_kind,
           "--backend", backend, "--manifest", str(manifest), "--out", str(out)]
    if limit:
        cmd += ["--limit", str(limit)]
    log = open(out / "job.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    _procs[proc.pid] = proc
    (out / "job.json").write_text(json.dumps({
        "pid": proc.pid, "system": system, "input": input_kind, "backend": backend, "limit": limit,
        "manifest": str(manifest), "n_total": n_total, "started": time.time(), "spend_at_start": total_spend(),
        "cmd": cmd}, indent=2))
    return out


def status(run_dir: Path) -> dict:
    from tutor.llm import total_spend

    job_path = run_dir / "job.json"
    job = json.loads(job_path.read_text()) if job_path.exists() else {}
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    if (run_dir / "scores.json").exists():
        state = "finished"
    elif job and _alive(job["pid"]):
        state = "running"
    else:
        state = "stopped"
    log = (run_dir / "job.log").read_text()[-3000:] if (run_dir / "job.log").exists() else ""
    return {
        "state": state, "done": len(preds), "total": job.get("n_total"), "job": job,
        "spend": (total_spend() - job["spend_at_start"]) if job and state == "running" else None,
        "last": preds[-1] if preds else None, "log_tail": log,
        "elapsed": time.time() - job["started"] if job and state == "running" else None,
    }


def stop(run_dir: Path) -> None:
    job = json.loads((run_dir / "job.json").read_text())
    if _alive(job["pid"]):
        os.kill(job["pid"], signal.SIGTERM)


def list_runs(runs: Path = RUNS) -> list[Path]:
    dirs = [p.parent for p in runs.glob("*/predictions.jsonl")]
    return sorted(dirs, key=lambda d: (d / "predictions.jsonl").stat().st_mtime, reverse=True)


def run_system(run_dir: Path) -> str:
    job_path = run_dir / "job.json"
    if job_path.exists():
        return json.loads(job_path.read_text())["system"]
    for s in ("algorithmic", "text_llm", "audio_llm", "agent"):  # CLI runs: runs/<system>_<input>...
        if run_dir.name.startswith(s + "_"):
            return s
    return "?"


def load_run(run_dir: Path, manifest: Path = MANIFEST) -> dict:
    """Scores (graded now if the run was stopped early), the manifest rows, and the predictions."""
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    sliced = run_dir / "manifest_slice.jsonl"
    rows = _read_jsonl(sliced) if sliced.exists() else []
    if not rows:
        job_path = run_dir / "job.json"
        mf = Path(json.loads(job_path.read_text())["manifest"]) if job_path.exists() else manifest
        done = {p["case_id"] for p in preds}
        rows = [r for r in read_manifest(mf) if r["case_id"] in done]
    if (run_dir / "scores.json").exists():
        scores = json.loads((run_dir / "scores.json").read_text())
    elif rows and preds:
        tmp = run_dir / "_partial_manifest.jsonl"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        scores = grade(tmp, run_dir / "predictions.jsonl")
        scores["partial"] = True
    else:
        scores = None
    return {"scores": scores, "rows": rows, "preds": preds, "system": run_system(run_dir)}


def case_cost(pred: dict) -> float:
    return float((pred.get("usage") or {}).get("cost_usd") or 0.0)


def estimate_cost(system: str, n_cases: int, runs: Path = RUNS) -> float | None:
    """Average cost per case over past runs of this system (cache hits excluded); None if no history."""
    if system == "algorithmic":
        return 0.0
    costs = [case_cost(p) for d in list_runs(runs) if run_system(d) == system
             for p in _read_jsonl(d / "predictions.jsonl") if case_cost(p) > 0]
    return (sum(costs) / len(costs)) * n_cases if costs else None
