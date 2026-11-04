"""Grade error-detection predictions against ground-truth perturbation labels.

A prediction matches a ground-truth error when:
  - same type, and
  - |time difference| <= TIME_TOL (100 ms), and
  - same pitch (reference pitch; for "extra" the played pitch).
Matching is one-to-one, greedy by time difference.

Also reports localization: fraction of ground-truth errors that have ANY
prediction (any type) within LOC_TOL (500 ms) -- "did it point at the right spot".

Usage:
    python -m eval.grade_detection --manifest data/cases/manifest.jsonl \
        --pred runs/agent_oracle/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TIME_TOL = 0.10
LOC_TOL = 0.50
TYPES = ["wrong_pitch", "missed", "extra", "timing"]


def match(gt: list[dict], pred: list[dict]) -> list[tuple[int, int]]:
    cands = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if p.get("type") != g["type"]:
                continue
            if p.get("pitch") is not None and p.get("pitch") != g["pitch"]:
                continue
            dt = abs(float(p.get("time", 1e9)) - g["time"])
            if dt <= TIME_TOL:
                cands.append((dt, i, j))
    cands.sort()
    ug, up, out = set(), set(), []
    for _, i, j in cands:
        if i not in ug and j not in up:
            ug.add(i); up.add(j); out.append((i, j))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def grade(manifest: Path, pred_path: Path) -> dict:
    gt_rows = {r["case_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    preds = {r["case_id"]: r for r in map(json.loads, pred_path.read_text().splitlines())}
    by_type = defaultdict(lambda: [0, 0, 0])
    by_sev = defaultdict(lambda: [0, 0, 0])
    loc_hit = loc_total = 0
    clean_fp, clean_cases, failed = 0, 0, 0
    for cid, row in gt_rows.items():
        if cid not in preds:
            continue
        if preds[cid].get("error"):
            failed += 1
        gt, pr = row["errors"], preds[cid].get("errors", [])
        m = match(gt, pr)
        mg, mp = {i for i, _ in m}, {j for _, j in m}
        for i, g in enumerate(gt):
            by_type[g["type"]][0 if i in mg else 2] += 1
            by_sev[row["severity"]][0 if i in mg else 2] += 1
            loc_total += 1
            loc_hit += any(abs(float(p.get("time", 1e9)) - g["time"]) <= LOC_TOL for p in pr)
        for j, p in enumerate(pr):
            if j not in mp:
                by_type[p.get("type", "?")][1] += 1
                by_sev[row["severity"]][1] += 1
        if row["severity"] == 0:
            clean_cases += 1
            clean_fp += len(pr)
    tot = [sum(v[k] for v in by_type.values()) for k in range(3)]
    return {
        "n_cases": len(preds),
        "failed_runs": failed,
        "overall": prf(*tot),
        "by_type": {t: prf(*by_type[t]) for t in sorted(by_type)},
        "by_severity": {str(s): prf(*by_sev[s]) for s in sorted(by_sev)},
        "localization_acc": round(loc_hit / loc_total, 3) if loc_total else None,
        "false_positives_per_clean_case": round(clean_fp / clean_cases, 2) if clean_cases else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(grade(args.manifest, args.pred), indent=2))


if __name__ == "__main__":
    main()
