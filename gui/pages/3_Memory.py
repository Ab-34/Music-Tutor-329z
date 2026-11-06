import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from tutor.notes import pitch_name  # noqa: E402
from tutor.tools import memory  # noqa: E402

common.page("Memory", "🧠")
st.title("🧠 Mistake log")
st.caption("The persistent memory the agent reads before giving feedback and writes after each assessment.")

learners = sorted(p.stem for p in memory.ROOT.glob("*.json"))
if not learners:
    st.info("No mistake logs yet. Run the agent on the **Assess** page with *Write to memory* on.")
    st.stop()

learner = st.selectbox("Learner", learners)
log = memory.read(learner)
pieces = sorted({e["piece"] for e in log})
c = st.columns(3)
c[0].metric("Entries", len(log))
c[1].metric("Pieces", len(pieces))
c[2].metric("Repeated mistakes (count > 1)", sum(e["count"] > 1 for e in log))

for piece in pieces:
    st.subheader(piece)
    rows = [{"time (s)": e["time"], "type": e["type"],
             "note": pitch_name(e["pitch"]) if e.get("pitch") is not None else "", "count": e["count"],
             "first seen": e["first_seen"], "last seen": e["last_seen"], "explanation": e.get("note", "")}
            for e in sorted(log, key=lambda e: (-e["count"], e["time"])) if e["piece"] == piece]
    st.dataframe(rows, width="stretch", hide_index=True)

st.divider()
with st.expander("Clear this learner's log"):
    confirm = st.checkbox(f"Yes, delete every entry for **{learner}**")
    if st.button("Delete log", disabled=not confirm, type="secondary"):
        memory._path(learner).unlink(missing_ok=True)
        st.success(f"Deleted the log for {learner}.")
        st.rerun()
