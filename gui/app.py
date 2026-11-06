"""Music Tutor. Run from the repo root:  streamlit run gui/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

nav = st.navigation({
    "Tutor": [st.Page("views/tutor.py", title="Learn the Rondo", icon="🎹", default=True)],
    "Developer": [
        st.Page("views/assess.py", title="Assess (one system)", icon="🔎"),
        st.Page("views/eval.py", title="Eval runs", icon="📊"),
        st.Page("views/spending.py", title="Spending", icon="💸"),
    ],
})
nav.run()
