"""Where inputs come from: the bundled example, generated eval cases, and uploads."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "cases" / "manifest.jsonl"
UPLOADS = ROOT / "runs" / "gui_uploads"


@dataclass
class Source:
    label: str
    ref: str                       # reference .mid
    perf_midi: str | None = None   # performance as MIDI (exact)
    perf_audio: str | None = None  # performance as audio (.wav)
    gt: list[dict] | None = None   # known errors, if any
    case_id: str | None = None
    severity: int | None = None


def _abs(p: str | None) -> str | None:
    if p is None:
        return None
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


def bundled() -> Source:
    ex = ROOT / "examples"
    gt_path = ex / "perf_errors.json"
    return Source("Bundled example (3 known mistakes)", str(ex / "ref.mid"), perf_midi=str(ex / "perf.mid"),
                  gt=json.loads(gt_path.read_text()) if gt_path.exists() else None)


def read_manifest(manifest: Path = MANIFEST) -> list[dict]:
    if not manifest.exists():
        return []
    return [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]


def eval_cases(manifest: Path = MANIFEST) -> list[Source]:
    out = []
    for r in read_manifest(manifest):
        out.append(Source(f"{r['case_id']}  (severity {r['severity']}, {len(r['errors'])} errors)",
                          _abs(r["ref"]), perf_midi=_abs(r.get("perf_midi")), perf_audio=_abs(r.get("perf_audio")),
                          gt=r["errors"], case_id=r["case_id"], severity=r["severity"]))
    return out


def case_by_id(case_id: str, manifest: Path = MANIFEST) -> Source | None:
    return next((s for s in eval_cases(manifest) if s.case_id == case_id), None)


def save_upload(uploaded, folder: Path | None = None) -> str:
    """Save a Streamlit UploadedFile under runs/gui_uploads/<timestamp>/ and return its path."""
    folder = folder or UPLOADS / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
    return str(path)
