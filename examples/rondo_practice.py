"""Practice sheet for playing the Rondo alla Turca one note at a time (no chords).

    python -m examples.rondo_practice     # writes examples/rondo/practice.html (+ practice/*.mid)

Version 1 is the full right-hand melody, every note as written (a two-note interval keeps its top note).
Version 2 is that melody after the simplification module:
drop_ornaments -> thin_to_grid (to eighth notes) -> slow_down (90 -> 60 bpm).
Both are saved as MIDI too, so a take of either can be assessed against the right reference.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from examples.rondo_demo import BEATS_PER_BAR, BPM, OUT, PICKUP_BEATS, SEC_PER_BEAT, load_excerpt
from tutor.notes import Note, save_midi
from tutor.tools.simplify import difficulty, drop_ornaments, slow_down, thin_to_grid

PAGE = OUT / "practice.html"
SLOW_BPM = 60
NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
BLACK = {1, 3, 6, 8, 10}


def top_line(notes: list[Note], tol: float = 0.005) -> list[Note]:
    """Top note of each group of notes struck together. Unlike melody_only (which groups
    within 50 ms), this keeps quick grace notes as separate notes."""
    groups: list[list[Note]] = []
    for n in sorted(notes, key=lambda n: (n.onset, -n.pitch)):
        if groups and n.onset - groups[-1][0].onset <= tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    return [max(g, key=lambda n: n.pitch) for g in groups]


def full_melody() -> list[Note]:
    rh, _ = load_excerpt()
    return top_line(rh)  # top note of each right-hand interval; one key at a time


def simplified_melody() -> list[Note]:
    line = drop_ornaments(full_melody())
    line = thin_to_grid(line, grid=SEC_PER_BEAT / 2)   # runs of four sixteenths -> two eighths
    return slow_down(line, BPM / SLOW_BPM)             # 90 -> 60 bpm


def versions() -> list[dict]:
    return [
        {"id": "full", "title": "The full melody", "bpm": BPM, "notes": full_melody(),
         "intro": "Every note of the right hand, as Mozart wrote it. The quick runs come four to a beat."},
        {"id": "simple", "title": "The simplified melody", "bpm": SLOW_BPM, "notes": simplified_melody(),
         "intro": "The same tune with the quick runs thinned out and the tempo slowed down.",
         "changes": ["Dropped the two quick grace notes before the B in bars 5, 6 and 7.",
                     "Kept two notes out of every quick run of four, so the fastest notes are now half a beat.",
                     f"Slowed down from {BPM} to {SLOW_BPM} beats a minute."]},
    ]


# ---- layout -----------------------------------------------------------------

def _q(x: float, step: float = 0.25) -> float:
    return round(x / step) * step


def bars(notes: list[Note], spb: float) -> list[dict]:
    """Group a single-note line into bars of tiles. Units are sixteenth notes."""
    out: dict[int, dict] = {}
    graces: list[Note] = []
    for i, n in enumerate(sorted(notes, key=lambda n: n.onset)):
        beat = n.onset / spb - PICKUP_BEATS
        dur = (n.offset - n.onset) / spb
        if dur < 0.15:          # grace note: drawn small, above the note it leads into
            graces.append(n)
            continue
        bar = 0 if beat < -1e-3 else int(beat / BEATS_PER_BAR + 1e-3) + 1  # onsets are rounded to 0.1 ms
        start = -PICKUP_BEATS if bar == 0 else (bar - 1) * BEATS_PER_BAR
        b = out.setdefault(bar, {"bar": bar, "items": [], "cursor": float(start)})
        rest = _q(beat - b["cursor"])
        if rest >= 0.25:
            b["items"].append({"rest": int(rest * 4)})
        units = max(1, int(_q(dur) * 4))
        b["items"].append({"note": n, "units": units, "graces": graces})
        graces = []
        b["cursor"] = _q(beat) + units / 4
    return [out[k] for k in sorted(out)]


def name(p: int) -> tuple[str, int]:
    return NAMES[p % 12], p // 12 - 1


def tile(n: Note, units: int, graces: list[Note], idx: dict) -> str:
    letter, octave = name(n.pitch)
    kind = "ebony" if n.pitch % 12 in BLACK else "ivory"
    g = ""
    if graces:
        g = '<span class="grace" title="Quick grace notes: tap them just before this note">' + "".join(
            f'<button type="button" class="gn" data-i="{idx[id(x)]}" data-m="{x.pitch}">{name(x.pitch)[0]}</button>'
            for x in graces) + "</span>"
    label = f"{letter}{octave}, " + ("hold " + {1: "a quarter beat", 2: "half a beat", 4: "one beat",
                                               8: "two beats"}.get(units, f"{units / 4:g} beats"))
    return (f'<span class="slot" style="--u:{units}">{g}<button type="button" class="n {kind}" '
            f'data-i="{idx[id(n)]}" data-m="{n.pitch}" aria-label="{html.escape(label)}">'
            f'<b>{letter}</b><small>{octave}</small></button></span>')


def section(v: dict) -> str:
    spb = 60 / v["bpm"]
    notes = sorted(v["notes"], key=lambda n: n.onset)
    idx = {id(n): i for i, n in enumerate(notes)}
    rows = []
    for b in bars(notes, spb):
        cells = "".join(f'<span class="slot rest" style="--u:{it["rest"]}" aria-label="rest"></span>' if "rest" in it
                        else tile(it["note"], it["units"], it["graces"], idx) for it in b["items"])
        label = "Pickup" if b["bar"] == 0 else f"Bar {b['bar']}"
        rows.append(f'<div class="bar"><span class="bn">{label}</span><div class="tiles">{cells}</div></div>')
    d = difficulty(notes)
    data = json.dumps([[n.pitch, round(n.onset, 4), round(n.offset - n.onset, 4)] for n in notes])
    changes = ""
    if v.get("changes"):
        changes = ('<div class="changes"><h3>What the simplifier changed</h3><ul>'
                   + "".join(f"<li>{html.escape(c)}</li>" for c in v["changes"]) + "</ul></div>")
    return f'''
<section class="version" id="{v["id"]}" data-notes='{data}'>
  <div class="vhead">
    <h2>{v["title"]}</h2>
    <p class="facts">{len(notes)} notes, {v["bpm"]} beats a minute, about {d["notes_per_sec"]} notes a second</p>
    <p>{html.escape(v["intro"])}</p>
    <div class="controls">
      <button type="button" class="play" data-rate="1">Play</button>
      <button type="button" class="play" data-rate="0.5">Play at half speed</button>
      <button type="button" class="stop" hidden>Stop</button>
    </div>
  </div>
  <div class="bars">{"".join(rows)}</div>
  {changes}
</section>'''


def keyboard(lo: int, hi: int, used: set[int], start: int) -> str:
    whites = [p for p in range(lo, hi + 1) if p % 12 not in BLACK]
    w = 100 / len(whites)
    keys, labels = [], []
    for k, p in enumerate(whites):
        cls = "wk" + (" used" if p in used else "") + (" start" if p == start else "")
        letter, octave = name(p)
        keys.append(f'<span class="{cls}" data-m="{p}" style="left:{k * w:.4f}%;width:{w:.4f}%">'
                    f'<span>{letter}<small>{octave}</small></span></span>')
        if p == 60:
            labels.append(f'<span class="kl" style="left:{k * w:.4f}%;width:{w:.4f}%">middle C</span>')
        if p == start:
            labels.append(f'<span class="kl startl" style="left:{k * w:.4f}%;width:{w:.4f}%">start here</span>')
    for p in range(lo, hi + 1):
        if p % 12 in BLACK:
            left_white = sum(1 for q in whites if q < p)
            cls = "bk" + (" used" if p in used else "")
            lbl = f"<span>{name(p)[0]}</span>" if p in used else ""
            keys.append(f'<span class="{cls}" data-m="{p}" style="left:{left_white * w - w * 0.3:.4f}%;'
                        f'width:{w * 0.6:.4f}%">{lbl}</span>')
    return f'<div class="kb" aria-hidden="true">{"".join(keys)}</div><div class="kls">{"".join(labels)}</div>'


CSS = """
:root{--paper:#F6F7F9;--ink:#17212B;--muted:#5A6472;--ivory:#FFFFFF;--line:#C7CDD6;--ebony:#1F242C;
--accent:#0E7C86;--accent-ink:#FFFFFF;--unit:40px}
@media (prefers-color-scheme:dark){:root{--paper:#14181D;--ink:#E8ECF1;--muted:#9AA4B1;--line:#59616C}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.55 "Atkinson Hyperlegible",system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px 64px}
header{padding:28px 0 8px}
h1{font-size:2.1rem;line-height:1.15;margin:0 0 6px;font-weight:700;letter-spacing:-.01em}
h1+p{margin:0;color:var(--muted);max-width:62ch}
.keys{position:sticky;top:0;z-index:5;background:var(--paper);padding:12px 0 10px;border-bottom:1px solid var(--line)}
.kb{position:relative;height:118px;max-width:760px}
.wk{position:absolute;top:0;height:118px;background:var(--ivory);border:1px solid var(--line);border-radius:0 0 6px 6px;
display:flex;align-items:flex-end;justify-content:center;padding-bottom:6px;color:#8A93A0;font-size:.8rem}
.wk.used{color:#17212B;font-weight:700}
.wk.used>span::after{content:"";display:block;width:6px;height:6px;border-radius:50%;background:#17212B;margin:3px auto 0}
.bk{position:absolute;top:0;height:72px;background:var(--ebony);border-radius:0 0 4px 4px;z-index:2;
display:flex;align-items:flex-end;justify-content:center;padding-bottom:5px;color:#fff;font-size:.7rem;font-weight:700}
.wk.start{box-shadow:inset 0 -5px 0 var(--accent)}
.wk.on,.bk.on{background:var(--accent);color:var(--accent-ink)}
.kls{position:relative;height:20px;max-width:760px;font-size:.78rem;color:var(--muted)}
.kl{position:absolute;top:2px;text-align:center;white-space:nowrap}
.startl{color:var(--accent);font-weight:700}
.howto{display:flex;flex-wrap:wrap;gap:6px 22px;margin:18px 0 0;padding:0;list-style:none;color:var(--muted);font-size:.95rem}
.howto i{display:inline-block;width:18px;height:18px;vertical-align:-3px;margin-right:6px;border:1px solid var(--line);border-radius:3px}
.howto .i1{background:var(--ivory)}.howto .i2{background:var(--ebony);border-color:var(--ebony)}
.version{margin-top:44px}
h2{font-size:1.5rem;margin:0 0 2px}
.facts{margin:0 0 6px;color:var(--muted);font-size:.95rem}
.vhead p{max-width:64ch}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 18px}
.controls button{font:inherit;font-weight:700;padding:9px 18px;border-radius:999px;border:2px solid var(--ink);
background:transparent;color:var(--ink);cursor:pointer}
.controls .play[data-rate="1"]{background:var(--ink);color:var(--paper)}
button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.bars{display:flex;flex-wrap:wrap;gap:22px 28px}
.bar{display:flex;flex-direction:column;gap:6px}
.bn{font-size:.85rem;color:var(--muted)}
.tiles{display:flex;align-items:flex-end;gap:4px;min-height:96px;padding-left:10px;border-left:2px solid var(--line)}
.slot{display:flex;flex-direction:column;align-items:stretch;width:calc(var(--u) * var(--unit) - 4px)}
.slot.rest{align-self:center;height:2px;background:repeating-linear-gradient(90deg,var(--line) 0 6px,transparent 6px 10px)}
.n{font:inherit;height:64px;border-radius:6px;cursor:pointer;display:flex;align-items:baseline;justify-content:center;
padding-top:16px;gap:1px;border:1.5px solid var(--line);background:var(--ivory);color:#17212B}
.n b{font-size:1.35rem;line-height:1}
.n small{font-size:.72rem;opacity:.7}
.n.ebony{background:var(--ebony);border-color:var(--ebony);color:#fff}
.n.on,.gn.on{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.grace{display:flex;gap:2px;margin-bottom:4px}
.gn{font:inherit;font-size:.72rem;font-weight:700;width:22px;height:22px;padding:0;border-radius:4px;cursor:pointer;
border:1.5px dashed var(--muted);background:var(--ivory);color:#17212B}
.changes{margin-top:22px;max-width:64ch}
.changes h3{font-size:1.05rem;margin:0 0 4px}
.changes ul{margin:0;padding-left:20px}
footer{margin-top:56px;color:var(--muted);font-size:.85rem;max-width:72ch}
@media (max-width:600px){:root{--unit:34px}h1{font-size:1.7rem}.kb,.wk{height:92px}.bk{height:56px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
const ctx = () => (window._ac ||= new (window.AudioContext || window.webkitAudioContext)());
const hz = m => 440 * Math.pow(2, (m - 69) / 12);
function tone(m, t, d) {           // reedy, harmonium-like voice
  const ac = ctx(), g = ac.createGain(), f = ac.createBiquadFilter();
  f.type = "lowpass"; f.frequency.value = 2200; g.connect(ac.destination); f.connect(g);
  [0, 4].forEach(c => { const o = ac.createOscillator(); o.type = "sawtooth"; o.frequency.value = hz(m);
    o.detune.value = c; o.connect(f); o.start(t); o.stop(t + d + 0.1); });
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.12, t + 0.03);
  g.gain.setValueAtTime(0.12, t + Math.max(0.04, d - 0.04)); g.gain.linearRampToValueAtTime(0, t + d + 0.08);
}
function light(sec, i, m, on) {
  sec.querySelectorAll(`[data-i="${i}"]`).forEach(e => e.classList.toggle("on", on));
  document.querySelectorAll(`.kb [data-m="${m}"]`).forEach(e => e.classList.toggle("on", on));
}
let timers = [];
function stopAll() {
  timers.forEach(clearTimeout); timers = [];
  document.querySelectorAll(".on").forEach(e => e.classList.remove("on"));
  document.querySelectorAll(".stop").forEach(b => b.hidden = true);
  if (window._ac) { window._ac.close(); window._ac = null; }
}
document.querySelectorAll(".version").forEach(sec => {
  const notes = JSON.parse(sec.dataset.notes);
  sec.querySelectorAll(".play").forEach(btn => btn.addEventListener("click", () => {
    stopAll();
    const rate = +btn.dataset.rate, ac = ctx(), t0 = ac.currentTime + 0.15;
    notes.forEach(([m, t, d], i) => {
      tone(m, t0 + t / rate, d / rate);
      timers.push(setTimeout(() => light(sec, i, m, true), 150 + 1000 * t / rate));
      timers.push(setTimeout(() => light(sec, i, m, false), 150 + 1000 * (t + d) / rate));
    });
    const last = notes[notes.length - 1];
    timers.push(setTimeout(stopAll, 400 + 1000 * (last[1] + last[2]) / rate));
    sec.querySelector(".stop").hidden = false;
  }));
  sec.querySelector(".stop").addEventListener("click", stopAll);
  sec.querySelectorAll("[data-i]").forEach(el => el.addEventListener("click", () => {
    const [m, , d] = notes[+el.dataset.i], ac = ctx();
    tone(m, ac.currentTime + 0.01, Math.min(d, 0.6));
    light(sec, el.dataset.i, m, true); setTimeout(() => light(sec, el.dataset.i, m, false), 450);
  }));
});
"""


def render() -> str:
    vs = versions()
    used = {n.pitch for v in vs for n in v["notes"]}
    lo, hi = (min(used) // 12) * 12, (max(used) // 12) * 12 + (0 if max(used) % 12 == 0 else 12)
    lo = min(lo, 60)
    start = min(vs[0]["notes"], key=lambda n: n.onset).pitch
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rondo alla Turca practice</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>Rondo alla Turca, one note at a time</h1>
  <p>Mozart, Piano Sonata No. 11, last movement: the opening theme, for the right hand only.
     Press one key at a time, like on a harmonium. Play the full melody first, then the simplified one.</p>
</header>
<div class="keys">
  {keyboard(lo, hi, used, start)}
</div>
<ul class="howto">
  <li><i class="i1"></i>White tile: white key</li>
  <li><i class="i2"></i>Dark tile: black key (the ♯ one just right of the letter)</li>
  <li>Wider tile: hold it longer. A dashed line is a short silence.</li>
  <li>Tap any tile to hear that note. The key lights up above.</li>
</ul>
{"".join(section(v) for v in vs)}
<footer>Notes from the public-domain Mutopia Project score of K. 331 (examples/rondo/source).
Generated by <code>python -m examples.rondo_practice</code>. Two-note intervals in bars 5 to 8 keep their top note.</footer>
</div><script>{JS}</script></body></html>
'''


def main() -> None:
    PAGE.write_text(render())
    for v in versions():
        save_midi(v["notes"], OUT / "practice" / f"{v['id']}_melody.mid")
        print(f"{v['title']:<24} {len(v['notes']):>3} notes  {difficulty(v['notes'])}")
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
