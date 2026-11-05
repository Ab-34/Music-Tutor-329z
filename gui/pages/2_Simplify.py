import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from gui import components as C  # noqa: E402
from gui import sources  # noqa: E402
from tutor.notes import load_midi  # noqa: E402
from tutor.tools.simplify import STEPS, difficulty  # noqa: E402

common.page("Simplify", "🪜")
st.title("🪜 Simplification ladder")
st.caption("Each rung applies one more step to the rung above it. The difficulty scores should go down as you go.")

kind = st.radio("Piece", ["Bundled example", "Eval case reference", "Upload"], horizontal=True)
ref_path = None
if kind == "Bundled example":
    ref_path = sources.bundled().ref
elif kind == "Eval case reference":
    cases = sources.eval_cases()
    if cases:
        ref_path = st.selectbox("Case", cases, format_func=lambda c: c.label).ref
    else:
        st.info("No eval set yet. Build one on the **Eval** page.")
else:
    up = st.file_uploader("Score (.mid)", type=["mid", "midi"])
    if up:
        ref_path = sources.save_upload(up)

steps = st.multiselect("Steps, in order", list(STEPS), default=["drop_inner_voices", "bass_roots", "melody_only"],
                       help="The order you pick them in is the order they're applied.")

if ref_path:
    notes = load_midi(ref_path)
    ladder = [("original", notes)]
    for step in steps:
        ladder.append((step, STEPS[step](ladder[-1][1])))

    diffs = [{"rung": f"{i}. {name}", **difficulty(n), "notes": len(n)} for i, (name, n) in enumerate(ladder)]
    st.dataframe(diffs, width="stretch", hide_index=True)
    chart = {k: [d[k] for d in diffs] for k in ("notes_per_sec", "mean_polyphony", "pitch_range", "max_leap")}
    st.line_chart(chart, height=220)

    for i, (name, n) in enumerate(ladder):
        with st.expander(f"Rung {i}: {name}  ({len(n)} notes)", expanded=i in (0, len(ladder) - 1)):
            st.plotly_chart(C.piano_roll(n, height=300), width="stretch", key=f"roll_{i}")
            a, b = st.columns([3, 1])
            a.audio(C.wav_bytes(n), format="audio/wav")
            b.download_button("Download .mid", C.midi_bytes(n), file_name=f"{i}_{name}.mid", mime="audio/midi",
                              key=f"dl_{i}")
