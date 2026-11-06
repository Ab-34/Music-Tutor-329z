import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from gui import components as C  # noqa: E402
from tutor import llm, rubric  # noqa: E402
from tutor.lesson import LESSON_DIR, RUNGS, CoachPolicy, Lesson, RulesPolicy, rondo_piece  # noqa: E402

common.page("Lesson", "🧑‍🏫")
st.title("🧑‍🏫 Practice lesson")
st.caption("For a learner who is far off: the tutor splits the Rondo into 2-bar chunks. Each chunk has three levels: "
           "simplified, all notes slowly, and all notes at full tempo. After every attempt the tutor scores it and "
           "decides what comes next. The learner here is simulated and improves with practice.")

from examples.rondo_demo import PROFILES  # noqa: E402
from examples.rondo_learner import SKILLS, SimLearner  # noqa: E402

levels = [lv for lv in SKILLS if lv != "perfect"]
default_level = st.session_state.pop("lesson_level", None)
if default_level in levels:
    st.session_state["lesson_level_pick"] = default_level
st.session_state.setdefault("lesson_level_pick", "struggling")

c = st.columns([1.2, 1.4, 0.8, 1, 1, 1])
level = c[0].selectbox("Learner level", levels, format_func=lambda lv: PROFILES[lv].title, key="lesson_level_pick")
policy_name = c[1].radio("Who decides", ["rules", "llm"], horizontal=True, key="lesson_policy",
                         format_func=lambda p: "Rules (baseline)" if p == "rules" else "LLM coach",
                         help="Rules: pass at 85, harder; two fails below 50, easier; else repeat. "
                              "LLM coach: sees the rubric and history, may override the rules if it says why.")
seed = c[2].number_input("Seed", min_value=0, max_value=999, value=0, key="lesson_seed")
if policy_name == "llm" and not C.has_key():
    st.warning("Set OPENROUTER_API_KEY in .env to use the LLM coach.")

cfg = (level, policy_name, int(seed))
if st.session_state.get("lesson_cfg") != cfg or "lesson" not in st.session_state:
    if c[3].button("Start lesson", type="primary", key="lesson_new",
                   disabled=policy_name == "llm" and not C.has_key()):
        policy = RulesPolicy() if policy_name == "rules" else CoachPolicy()
        st.session_state["lesson"] = Lesson(rondo_piece(), SimLearner(level, int(seed)), policy)
        st.session_state["lesson_cfg"] = cfg
        st.rerun()
lesson: Lesson | None = st.session_state.get("lesson") if st.session_state.get("lesson_cfg") == cfg else None


def advance(n: int) -> None:
    try:
        for _ in range(n):
            if lesson.done:
                break
            lesson.step()
    except llm.BudgetExceeded as e:
        common.budget_banner(e)


if lesson is not None:
    if c[3].button("Step", disabled=lesson.done, key="lesson_step"):
        advance(1)
    if c[4].button("Run to end", disabled=lesson.done, key="lesson_run"):
        with st.spinner("Running the lesson…"):
            advance(200)
    if c[5].button("Reset", key="lesson_reset"):
        del st.session_state["lesson"]
        st.rerun()

if lesson is None:
    st.info("Choose a learner and who decides, then press **Start lesson**.")
else:
    s = lesson.summary()
    state = lesson.state
    m = st.columns(5)
    m[0].metric("Attempts", s["attempts"])
    m[1].metric("Chunks passed", f"{s['chunks_passed']}/{s['n_chunks']}")
    m[2].metric("Now on", "Done" if lesson.done else f"Chunk {state.chunk + 1}, {RUNGS[state.rung][0].replace('_', ' ')}")
    m[3].metric("Wasted repeats", s["wasted_repeats"], help="Repeats that didn't raise the score")
    m[4].metric("Coach cost", f"${s['cost_usd']:.3f}")
    if lesson.stopped:
        st.warning(f"Lesson stopped: {lesson.stopped}")
    elif lesson.done:
        st.success("Lesson finished." + (" Every chunk passed at full tempo." if s["all_passed"] else ""))

    if lesson.rows:
        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
        symbols = ["circle-open", "circle", "star"]
        for ch in lesson.chunks:
            rows = [r for r in lesson.rows if r["chunk"] == ch.index]
            if rows:
                fig.add_trace(go.Scatter(
                    x=[r["attempt"] for r in rows], y=[r["rubric"]["total"] for r in rows], mode="lines+markers",
                    name=ch.label, line=dict(color=colors[ch.index % 5]),
                    marker=dict(size=11, symbol=[symbols[r["rung"]] for r in rows]),
                    hovertext=[f"{r['rung_name']}: {r['rubric']['total']} → {r['action']}" for r in rows],
                    hoverinfo="text"))
        fig.add_hline(y=rubric.POLISH_AT, line_dash="dash", line_color="#2ca02c", annotation_text="pass 85")
        fig.add_hline(y=rubric.CHUNK_BELOW, line_dash="dot", line_color="#d62728", annotation_text="50")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[-5, 105], title="score"),
                          xaxis_title="attempt  (○ simplified · ● all notes slowly · ★ full tempo)",
                          legend=dict(orientation="h", y=1.12), plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Attempts")
        for i in range(len(lesson.rows) - 1, -1, -1):
            r = lesson.rows[i]
            rub = r["rubric"]
            override = r["chosen"] != r["suggested"]
            head = (f"{r['attempt']}. {r['chunk_label']} · {r['rung_name'].replace('_', ' ')} · "
                    f"**{rub['total']}** → {r['action'].replace('_', ' ')}" + (" ✱" if override else ""))
            with st.expander(head, expanded=i == len(lesson.rows) - 1):
                pc = st.columns(4)
                for col, (k, v) in zip(pc, rub["parts"].items()):
                    col.metric(rubric.PART_LABELS[k], v)
                st.markdown(f"**Feedback:** {r['feedback']}")
                st.markdown(f"**Why {r['action'].replace('_', ' ')}:** {r['reason']}"
                            + (f"  _(adjusted: {r['guard']})_" if r["guard"] else ""))
                if override:
                    st.caption(f"✱ The coach chose **{r['chosen'].replace('_', ' ')}**; the rules would have chosen "
                               f"**{r['suggested'].replace('_', ' ')}**."
                               + (f" It became **{r['action'].replace('_', ' ')}** ({r['guard']})." if r["guard"] else ""))
                if r["fallback"]:
                    st.caption("The coach's answer was unusable, so the rules decided.")
                if r["mistakes"]:
                    st.markdown("**Mistakes:** " + "; ".join(r["mistakes"]))
                tgt, att = lesson.audio[i]
                a1, a2 = st.columns(2)
                a1.caption("Target")
                a1.audio(C.wav_bytes(tgt), format="audio/wav")
                a2.caption("This attempt")
                a2.audio(C.wav_bytes(att), format="audio/wav")

with st.expander("Past lessons"):
    logs = sorted(LESSON_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
    rows = []
    for p in logs:
        last = p.read_text().splitlines()[-1:] or ["{}"]
        summ = json.loads(last[0]).get("summary")
        if summ:
            rows.append({k: summ[k] for k in ("lesson_id", "policy", "level", "seed", "attempts", "chunks_passed",
                                               "all_passed", "wasted_repeats", "coach_overrides", "cost_usd")})
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No finished lessons yet.")
    for pol in ("rules", "llm"):
        f = LESSON_DIR / f"summary_{pol}.json"
        if f.exists():
            st.markdown(f"**`eval.run_lessons` summary: {pol}**")
            st.dataframe([{"level": lv, **v} for lv, v in json.loads(f.read_text())["by_level"].items()],
                         hide_index=True, width="stretch")
