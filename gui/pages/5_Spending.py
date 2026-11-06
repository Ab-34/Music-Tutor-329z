import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from tutor import llm  # noqa: E402

common.page("Spending", "💸")
st.title("💸 LLM spending")
st.caption(f"Every paid call is logged in `{llm.LEDGER.name}`. Cache hits cost nothing and aren't logged.")

rows = llm.read_ledger()
cap = llm.budget()
spent = sum(r.get("cost_usd", 0.0) for r in rows)
c = st.columns(4)
c[0].metric("Total spent", f"${spent:.3f}")
c[1].metric("Budget", f"${cap:.2f}" if cap else "none")
c[2].metric("Left", f"${cap - spent:.2f}" if cap else "—")
c[3].metric("Paid calls", len(rows))
if not rows:
    st.info("No paid LLM calls yet.")
    st.stop()

xs, ys, run = [], [], 0.0
for r in sorted(rows, key=lambda r: r.get("ts", 0)):
    run += r.get("cost_usd", 0.0)
    xs.append(datetime.fromtimestamp(r.get("ts", 0)))
    ys.append(run)
fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line_shape="hv", name="cumulative spend"))
if cap:
    fig.add_hline(y=cap, line_dash="dash", line_color="#d62728", annotation_text=f"budget ${cap:.2f}")
fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="USD", plot_bgcolor="white")
st.plotly_chart(fig, width="stretch")

agg = defaultdict(lambda: {"calls": 0, "input tokens": 0, "output tokens": 0, "USD": 0.0, "seconds": 0.0})
for r in rows:
    a = agg[(r.get("tag") or "(untagged)", r.get("model", "?"))]
    a["calls"] += 1
    a["input tokens"] += r.get("input_tokens", 0)
    a["output tokens"] += r.get("output_tokens", 0)
    a["USD"] += r.get("cost_usd", 0.0)
    a["seconds"] += r.get("seconds", 0.0)
table = [{"tag": t, "model": m, **{k: round(v, 4) if isinstance(v, float) else v for k, v in a.items()}}
         for (t, m), a in sorted(agg.items(), key=lambda kv: -kv[1]["USD"])]
st.dataframe(table, hide_index=True, width="stretch")

with st.expander("Last 50 calls"):
    st.dataframe(list(reversed(rows[-50:])), hide_index=True, width="stretch")
