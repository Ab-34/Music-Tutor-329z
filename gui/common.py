"""Page setup shared by every page: repo-root cwd, page config, and the status sidebar."""
from __future__ import annotations

import os

import streamlit as st

from gui import components as C
from gui.sources import ROOT


def page(title: str, icon: str = "🎹") -> None:
    os.chdir(ROOT)  # memory_store/, runs/ and manifest paths are relative to the repo root
    st.set_page_config(page_title=f"{title} · Music Tutor", page_icon=icon, layout="wide")
    sidebar()


def sidebar() -> None:
    from tutor import llm

    with st.sidebar:
        st.markdown("### Status")
        st.caption(f"Model: `{llm.MODEL}` · effort `{llm.EFFORT}`")
        spent, cap = llm.total_spend(), llm.budget()
        if cap:
            st.progress(min(spent / cap, 1.0), text=f"Spent ${spent:.2f} of ${cap:.2f} budget")
        else:
            st.caption(f"Spent ${spent:.2f} (no TUTOR_BUDGET_USD set)")
        st.caption(("✅" if C.has_key() else "⚠️") + " OpenRouter key" +
                   ("" if C.has_key() else " missing: LLM systems disabled"))
        st.caption(("✅" if C.audio_extras_installed() else "➖") + " Audio extras (.wav input)" +
                   ("" if C.audio_extras_installed() else ' not installed: `uv pip install -e ".[audio]"`'))


def budget_banner(e: Exception) -> None:
    st.error(f"**Budget reached.** {e}  \nRaise `TUTOR_BUDGET_USD` in `.env` to continue, then restart the GUI.")
