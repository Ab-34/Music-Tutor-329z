"""Music Tutor console. Run from the repo root:  streamlit run gui/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from gui.sources import read_manifest  # noqa: E402

common.page("Home")

st.title("🎹 Agentic Music Tutor: console")
st.markdown(
    "Run every tool the tutor has so far and see how it's doing. "
    "Start with **Assess** on the bundled example: it has 3 known mistakes, so you can check the result right away."
)

cols = st.columns(6)
pages = [
    ("pages/1_Assess.py", "🔎 Assess", "Run the agent or a baseline on a performance. Live tool trace, piano roll, "
                                     "score against known mistakes, feedback, audio."),
    ("pages/2_Simplify.py", "🪜 Simplify", "Build the ladder of easier arrangements. Hear each rung, see its difficulty."),
    ("pages/3_Memory.py", "🧠 Memory", "The mistake log the agent keeps for each learner."),
    ("pages/4_Eval.py", "📊 Eval", "Launch and watch eval runs over the perturbation set, then compare systems."),
    ("pages/5_Spending.py", "💸 Spending", "Every paid LLM call, by tag and model, against the budget."),
    ("pages/6_Lesson.py", "🧑‍🏫 Lesson", "A practice loop for learners who are far off: chunks, simplified first, "
                                         "repeat or move on after each attempt."),
]
for col, (path, label, desc) in zip(cols, pages):
    with col:
        st.page_link(path, label=f"**{label}**")
        st.caption(desc)

st.divider()
n_cases = len(read_manifest())
st.markdown(
    f"**Eval set:** {n_cases} cases in `data/cases/manifest.jsonl`." if n_cases else
    "**Eval set:** none built yet. The Eval page can build a small one from the bundled example, or see the README "
    "for the MAESTRO set."
)
