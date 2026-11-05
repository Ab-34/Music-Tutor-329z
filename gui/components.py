"""Shared, Streamlit-free helpers: scoring, piano roll, audio, tables. (Pages do the st.* calls.)"""
from __future__ import annotations

import importlib.util
import io
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from eval.grade_detection import match, prf
from tutor.notes import Note, pitch_name, save_midi

# B1 is still a stub (raises NotImplementedError); remove it from here once it's implemented.
HIDDEN_SYSTEMS = {"audio_llm"}
LLM_SYSTEMS = {"agent", "text_llm", "audio_llm"}
SYSTEM_LABELS = {
    "agent": "Agent (LLM + tools)",
    "algorithmic": "B0 algorithmic (no LLM)",
    "text_llm": "B2 text LLM (one call)",
    "audio_llm": "B1 audio LLM",
}
TYPE_COLORS = {"wrong_pitch": "#d62728", "missed": "#ff7f0e", "extra": "#9467bd", "timing": "#1f77b4"}


def visible_systems() -> list[str]:
    from eval.run_eval import SYSTEMS
    return [s for s in SYSTEMS if s not in HIDDEN_SYSTEMS]


def has_key() -> bool:
    import tutor.llm  # noqa: F401  (loads .env)
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def audio_extras_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("basic_pitch", "librosa"))


def is_audio(path: str | None) -> bool:
    return bool(path) and Path(path).suffix.lower() not in {".mid", ".midi"}


# ---- scoring --------------------------------------------------------------

def score(gt: list[dict] | None, pred: list[dict]) -> dict:
    """Same matching as eval/grade_detection.py. Returns P/R/F1 (or None without ground truth),
    plus a status per prediction ("hit"/"false_positive") and per known error ("found"/"missed")."""
    if gt is None:
        return {"metrics": None, "pred_status": [None] * len(pred), "gt_status": []}
    pairs = match(gt, pred)
    mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
    return {
        "metrics": prf(len(pairs), len(pred) - len(pairs), len(gt) - len(pairs)),
        "pred_status": ["hit" if j in mp else "false_positive" for j in range(len(pred))],
        "gt_status": ["found" if i in mg else "missed" for i in range(len(gt))],
    }


def error_rows(errors: list[dict], status: list[str | None]) -> list[dict]:
    rows = []
    for e, st in zip(errors, status):
        row = {"time (s)": round(float(e.get("time", 0)), 2), "type": e.get("type"),
               "note": pitch_name(int(e["pitch"])) if e.get("pitch") is not None else ""}
        if e.get("played_pitch") is not None:
            row["played"] = pitch_name(int(e["played_pitch"]))
        if e.get("offset_ms") is not None:
            row["offset (ms)"] = e["offset_ms"]
        if st:
            row["vs. known"] = "✅ hit" if st == "hit" else "❌ false positive"
        if e.get("note"):
            row["explanation"] = e["note"]
        rows.append(row)
    return rows


# ---- piano roll -----------------------------------------------------------

def _segments(notes: list[Note]):
    xs, ys = [], []
    for n in notes:
        xs += [n.onset, max(n.offset, n.onset + 0.05), None]
        ys += [n.pitch, n.pitch, None]
    return xs, ys


def piano_roll(ref: list[Note], perf: list[Note] | None = None, pred: list[dict] | None = None,
               gt: list[dict] | None = None, pred_status: list | None = None, gt_status: list | None = None,
               title: str = "", height: int = 420) -> go.Figure:
    fig = go.Figure()
    x, y = _segments(ref)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="reference", hoverinfo="skip",
                             line=dict(color="rgba(150,150,150,0.55)", width=9)))
    if perf:
        x, y = _segments(perf)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="performance", hoverinfo="skip",
                                 line=dict(color="#2ca02c", width=3)))
    if gt:
        gs = gt_status or [None] * len(gt)
        fig.add_trace(go.Scatter(
            x=[g["time"] for g in gt], y=[g["pitch"] for g in gt], mode="markers", name="known mistake",
            marker=dict(symbol="circle-open", size=20, line=dict(width=3),
                        color=["#2ca02c" if s == "found" else "#d62728" if s == "missed" else "#555" for s in gs]),
            hovertext=[f"KNOWN {g['type']} {pitch_name(g['pitch'])} @ {g['time']:.2f}s"
                       + (f" — {s}" if s else "") for g, s in zip(gt, gs)], hoverinfo="text"))
    if pred:
        ps = pred_status or [None] * len(pred)
        pts = [(p, s) for p, s in zip(pred, ps) if p.get("pitch") is not None]
        fig.add_trace(go.Scatter(
            x=[p["time"] for p, _ in pts], y=[p["pitch"] for p, _ in pts], mode="markers", name="predicted error",
            marker=dict(size=11, symbol=["x" if s == "false_positive" else "diamond" for _, s in pts],
                        color=[TYPE_COLORS.get(p["type"], "#333") for p, _ in pts],
                        line=dict(width=1, color="white")),
            hovertext=[f"{p['type']} {pitch_name(int(p['pitch']))} @ {float(p['time']):.2f}s"
                       + (f" — {s.replace('_', ' ')}" if s else "") + (f"<br>{p['note']}" if p.get("note") else "")
                       for p, s in pts], hoverinfo="text"))
    pitches = [n.pitch for n in ref + (perf or [])] or [60]
    lo, hi = min(pitches) - 2, max(pitches) + 2
    cs = [p for p in range(lo, hi + 1) if p % 12 == 0]
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      xaxis_title="time (s, reference clock)", legend=dict(orientation="h", y=1.08),
                      yaxis=dict(range=[lo, hi], tickvals=cs, ticktext=[pitch_name(p) for p in cs], title="pitch"),
                      plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ---- audio / midi bytes -----------------------------------------------------

def wav_bytes(notes: list[Note], sr: int = 22050) -> bytes:
    from tutor.tools.synth import synthesize
    audio = synthesize(notes, sr=sr) if notes else np.zeros(sr // 2)
    peak = float(np.max(np.abs(audio))) or 1.0
    pcm = (audio / peak * 0.9 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return buf.getvalue()


def midi_bytes(notes: list[Note]) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.mid"
        save_midi(notes, path)
        return path.read_bytes()
