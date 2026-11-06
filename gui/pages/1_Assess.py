import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402
import traceback  # noqa: E402

import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from gui import components as C  # noqa: E402
from gui import sources  # noqa: E402
from tutor import llm  # noqa: E402
from tutor.notes import load_midi  # noqa: E402

common.page("Assess", "🔎")
st.title("🔎 Assess a performance")


@st.cache_data(show_spinner=False)
def notes_of(path: str, mtime: float):
    return load_midi(path)


@st.cache_data(show_spinner=False)
def wav_of(path: str, mtime: float) -> bytes:
    return C.wav_bytes(load_midi(path))


def _mtime(p: str) -> float:
    return Path(p).stat().st_mtime


# ---- input ------------------------------------------------------------------

def pick_rondo() -> dict:
    from examples.rondo_demo import PROFILES

    takes = sources.rondo_takes()
    c1, c2 = st.columns([1, 2])
    level = c1.selectbox("Learner level", list(PROFILES), format_func=lambda lv: PROFILES[lv].title, key="rondo_level")
    mine = [t for t in takes if t.level == level]
    s = c2.selectbox("Take", mine, format_func=lambda t: t.label, key="rondo_take")
    st.caption(f"**{PROFILES[level].title}:** {PROFILES[level].blurb}  ·  Mozart, *Rondo alla Turca* (K. 331), "
               "pickup + first 8 bars at 90 bpm. True notes are from the public-domain Mutopia score.")
    a1, a2 = st.columns(2)
    a1.markdown("**The score, played correctly**")
    a1.audio(wav_of(s.ref, _mtime(s.ref)), format="audio/wav")
    a2.markdown("**This learner's take**")
    a2.audio(wav_of(s.perf_midi, _mtime(s.perf_midi)), format="audio/wav")
    with st.expander(f"🙈 What's wrong in this take? Listen first, then click to reveal ({len(s.gt)} mistakes)"):
        if not s.gt:
            st.write("Nothing. This take is played exactly as written.")
        for e in s.gt:
            st.markdown(f"- {e['description']}")
    return {"label": f"Rondo alla Turca · {s.label}", "ref": s.ref, "perf": s.perf_midi, "gt": s.gt,
            "case_id": s.case_id, "learner": s.learner, "piece": "rondo"}


def pick_source() -> dict | None:
    kind = st.radio("Input", ["Rondo alla Turca", "Bundled example", "Eval case", "Upload"], horizontal=True,
                    key="src_kind")
    if kind == "Rondo alla Turca":
        return pick_rondo()
    if kind == "Bundled example":
        s = sources.bundled()
        return {"label": s.label, "ref": s.ref, "perf": s.perf_midi, "gt": s.gt}
    if kind == "Eval case":
        cases = sources.eval_cases()
        if not cases:
            st.info("No eval set yet. Build one on the **Eval** page (or with `python -m data.make_perturbations`).")
            return None
        s = st.selectbox("Case", cases, format_func=lambda c: c.label, key="src_case")
        perf = s.perf_midi
        if s.perf_audio:
            if st.radio("Performance as", ["MIDI (exact)", "Audio (.wav)"], horizontal=True, key="src_as") != "MIDI (exact)":
                perf = s.perf_audio
        return {"label": s.label, "ref": s.ref, "perf": perf, "gt": s.gt, "case_id": s.case_id, "piece": s.piece}
    c1, c2, c3 = st.columns(3)
    ref = c1.file_uploader("Reference score (.mid)", type=["mid", "midi"], key="up_ref")
    perf = c2.file_uploader("Performance (.mid or .wav)", type=["mid", "midi", "wav"], key="up_perf")
    gt = c3.file_uploader("Known mistakes (optional .json)", type=["json"], key="up_gt")
    if not (ref and perf):
        st.caption("Upload a reference and a performance to continue.")
        return None
    folder = sources.UPLOADS / f"{ref.file_id[:8]}-{perf.file_id[:8]}"
    out = {"label": f"Upload: {perf.name}", "ref": sources.save_upload(ref, folder),
           "perf": sources.save_upload(perf, folder), "gt": None}
    if gt:
        try:
            out["gt"] = json.loads(gt.getvalue())
        except json.JSONDecodeError:
            st.warning("Couldn't parse the known-mistakes JSON; ignoring it.")
    return out


# ---- running ----------------------------------------------------------------

def run(system: str, src: dict, opts: dict) -> dict:
    if system == "algorithmic":
        from tutor.baselines.algorithmic import run as b0
        return b0(src["ref"], src["perf"], backend=opts["backend"])
    old = llm.MODEL, llm.EFFORT
    llm.MODEL, llm.EFFORT = opts["model"], opts["effort"]
    try:
        if system == "text_llm":
            from tutor.baselines.text_llm import run as b2
            return b2(src["ref"], src["perf"], backend=opts["backend"])
        from tutor.agent import run_agent

        with st.status("Agent is working…", expanded=True) as status:
            def on_event(ev):
                if ev["kind"] == "llm":
                    tag = " (cached)" if ev["cached"] else ""
                    status.update(label=f"Agent is working… turn {ev['turn']} · ${ev['total_cost_usd']:.3f}{tag}")
                    if ev.get("text"):
                        st.caption(f"💭 {ev['text'][:300]}")
                else:
                    icon = "✅" if ev["ok"] else "❌"
                    args = json.dumps(ev["input"]) if not isinstance(ev["input"], str) else ev["input"]
                    st.markdown(f"{icon} **`{ev['name']}`** `{args[:160]}`")
                    if not ev["ok"]:
                        st.caption(f"error: {ev['error']}")

            out = run_agent(src["ref"], src["perf"], learner=opts["learner"], write_memory=opts["write_memory"],
                            on_event=on_event)
            status.update(label="Agent finished" if not out.get("error") else f"Agent stopped: {out['error']}",
                          state="complete" if not out.get("error") else "error", expanded=False)
        return out
    finally:
        llm.MODEL, llm.EFFORT = old


# ---- results ----------------------------------------------------------------

def show_result(res: dict) -> None:
    out, src = res["out"], res["src"]
    errors = out.get("errors", [])
    bb = None
    if src.get("piece") == "rondo":
        from examples.rondo_demo import bar_beat_str as bb
    sc = C.score(src.get("gt"), errors)
    ref = notes_of(src["ref"], _mtime(src["ref"]))
    perf = None if C.is_audio(src["perf"]) else notes_of(src["perf"], _mtime(src["perf"]))

    if out.get("error"):
        st.error(f"Run error: {out['error']}")
    usage = out.get("usage") or {}
    m = st.columns(6)
    m[0].metric("Predicted errors", len(errors))
    m[1].metric("Known mistakes", len(src["gt"]) if src.get("gt") is not None else "—")
    if sc["metrics"]:
        m[2].metric("Precision", f"{sc['metrics']['precision']:.2f}")
        m[3].metric("Recall", f"{sc['metrics']['recall']:.2f}")
        m[4].metric("F1", f"{sc['metrics']['f1']:.2f}")
    m[5].metric("Cost", f"${usage.get('cost_usd', 0):.3f}" if usage else "$0")

    st.plotly_chart(C.piano_roll(ref, perf, errors, src.get("gt"), sc["pred_status"], sc["gt_status"],
                                 title=f"{C.SYSTEM_LABELS.get(res['system'], res['system'])} · {src['label']}"),
                    width="stretch")
    if perf is None:
        st.caption("The performance is audio, so the roll shows the reference with the predicted errors on top.")

    tabs = st.tabs(["Errors", "Feedback", "Trace", "Audio", "Raw"])
    with tabs[0]:
        rows = C.error_rows(errors, sc["pred_status"], bb)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.write("No errors predicted.")
        if src.get("gt") is not None:
            missed = [g for g, s in zip(src["gt"], sc["gt_status"]) if s == "missed"]
            st.markdown(f"**Known mistakes not found:** {len(missed)}")
            if missed:
                st.dataframe(C.error_rows(missed, [None] * len(missed), bb), width="stretch", hide_index=True)
    with tabs[1]:
        st.markdown(out.get("feedback") or "_This system doesn't write feedback._")
        if out.get("memory"):
            mem = out["memory"]
            st.caption(f"Mistake log: {mem['added']} added, {mem['updated']} repeats, {mem['total']} total entries.")
    with tabs[2]:
        trace = [t for t in out.get("trace", []) if "tool" in t]
        if not trace:
            st.write("No tool trace (baselines don't call tools).")
        for i, t in enumerate(trace, 1):
            with st.expander(f"{i}. {'✅' if t['ok'] else '❌'} {t['tool']}"):
                st.json(t.get("input") or {})
                if not t["ok"]:
                    st.error(t.get("error"))
        if usage:
            st.caption(f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out · "
                       f"cached calls: {usage.get('cached_calls', usage.get('cached', 0))}")
    with tabs[3]:
        a1, a2 = st.columns(2)
        a1.markdown("**Reference**")
        a1.audio(wav_of(src["ref"], _mtime(src["ref"])), format="audio/wav")
        a2.markdown("**Performance**")
        if C.is_audio(src["perf"]):
            a2.audio(src["perf"])
        else:
            a2.audio(wav_of(src["perf"], _mtime(src["perf"])), format="audio/wav")
    with tabs[4]:
        st.json({k: v for k, v in out.items() if k != "trace"})


# ---- page -------------------------------------------------------------------

preload = st.session_state.get("assess_preload")
if preload:
    st.info(f"Showing the stored prediction for **{preload['src']['label']}** from run `{preload['run']}`. "
            "No LLM call was made.")
    if st.button("Back to running new assessments"):
        del st.session_state["assess_preload"]
        st.rerun()
    show_result(preload)
    st.stop()

src = pick_source()
st.session_state.setdefault("learner", "demo")
if src and src.get("learner") and st.session_state.get("_learner_from") != src["learner"]:
    st.session_state["learner"] = st.session_state["_learner_from"] = src["learner"]  # one memory per level
c1, c2, c3 = st.columns([2, 2, 1])
systems = C.visible_systems()
system = c1.selectbox("System", systems, index=systems.index("agent"), format_func=lambda s: C.SYSTEM_LABELS[s],
                      key="system")
learner = c2.text_input("Learner", key="learner",
                        help="The agent reads and writes this learner's mistake log.")
write_memory = c3.checkbox("Write to memory", value=True, key="write_memory", disabled=system != "agent",
                           help="Run the same piece twice and the agent should mention the repeated mistakes.")
with st.expander("Model settings"):
    m1, m2, m3 = st.columns(3)
    model = m1.text_input("OpenRouter model", value=llm.MODEL, key="model")
    effort = m2.selectbox("Reasoning effort", ["low", "medium", "high"],
                          index=["low", "medium", "high"].index(llm.EFFORT) if llm.EFFORT in ("low", "medium", "high") else 1,
                          key="effort")
    backend = m3.selectbox("Transcription backend (.wav)", ["basic_pitch"], key="backend")

blocked = None
if src is None:
    blocked = "Choose an input."
elif system in C.LLM_SYSTEMS and not C.has_key():
    blocked = "Set OPENROUTER_API_KEY in .env to use LLM systems."
elif C.is_audio(src["perf"]) and not C.audio_extras_installed():
    blocked = 'Audio input needs the audio extras: `uv pip install -e ".[audio]"` (and fluidsynth for self-check).'
if blocked:
    st.warning(blocked)

if st.button("▶ Run assessment", type="primary", disabled=bool(blocked), key="run"):
    try:
        out = run(system, src, {"model": model, "effort": effort, "learner": learner,
                                "write_memory": write_memory and system == "agent", "backend": backend})
        st.session_state["assess_result"] = {"out": out, "src": src, "system": system}
    except llm.BudgetExceeded as e:
        common.budget_banner(e)
    except Exception as e:  # keep the page usable
        st.error(f"{type(e).__name__}: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

res = st.session_state.get("assess_result")
if res:
    st.divider()
    show_result(res)
