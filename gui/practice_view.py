"""Render an assignment (any passage, any level) as the practice-sheet tiles, for the Tutor page."""
from __future__ import annotations

from examples import rondo_practice as P
from tutor.lesson import Target


def assignment_html(tgt: Target, title: str, intro: str) -> str:
    piece = tgt.piece
    v = {"id": "assignment", "title": title, "intro": intro, "bpm": round(piece.bpm / tgt.factor),
         "notes": tgt.notes,
         # beats from bar 1 of the piece, so chunk tiles keep their real bar numbers
         "beat_of": lambda t: tgt.to_piece_time(t) / piece.sec_per_beat - piece.pickup_beats}
    used = {n.pitch for n in piece.melody}
    lo, hi = min(min(used) // 12 * 12, 60), max(used) // 12 * 12 + (0 if max(used) % 12 == 0 else 12)
    start = min(tgt.notes, key=lambda n: n.onset).pitch
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<style>{P.CSS} .wrap{{padding:0 4px 8px}} .version{{margin-top:14px}} header{{display:none}}</style></head>
<body><div class="wrap"><div class="keys">{P.keyboard(lo, hi, used, start)}</div>{P.section(v)}</div>
<script>{P.JS}</script></body></html>"""


def height_for(tgt: Target) -> int:
    n_bars = len(P.bars(tgt.notes, tgt.piece.sec_per_beat * tgt.factor,
                        lambda t: tgt.to_piece_time(t) / tgt.piece.sec_per_beat - tgt.piece.pickup_beats))
    return 330 + 120 * ((n_bars + 2) // 3)
