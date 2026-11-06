import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from gui import common  # noqa: E402
from gui import components as C  # noqa: E402
from gui.practice_view import assignment_html, height_for  # noqa: E402
from gui.sources import ROOT  # noqa: E402
from tutor import llm, rubric  # noqa: E402
from tutor.lesson import rondo_piece  # noqa: E402
from tutor.tools import memory  # noqa: E402
from tutor.tutor_agent import assignment_target, current_assignment, describe_assignment, run_tutor_turn  # noqa: E402

common.page("Learn the Rondo", "🎹")
RECORDINGS = ROOT / "runs" / "tutor_recordings"
HEADLINES = {
    "polish": "Nearly there. Polish these spots",
    "easier_version": "Let's slow it down first",
    "beginner_lesson": "Let's learn it in small pieces",
    "repeat": "Once more",
    "easier": "Let's make this one easier",
    "harder": "Good. On to the next level",
    "next_chunk": "Got it. On to the next chunk",
    "piece_learned": "You've learned it!",
}


@st.cache_resource
def piece():
    return rondo_piece()


P = piece()
st.title("Learn the Rondo alla Turca")
st.caption("Mozart, Piano Sonata No. 11 (K. 331), the opening theme. Right hand only, one note at a time. Play your "
           "assignment and send the tutor a recording. It listens, tells you what to fix, and decides what you "
           "practise next.")

name = st.text_input("Your name", value="me", key="tutor_name",
                     help="Your progress and mistakes are saved under this name.")
learner = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "me"
progress = memory.read_progress(learner, P.name)
history = progress.get("history", [])
assignment = current_assignment(learner, P)
tgt = assignment_target(P, assignment)


# ---- the tutor's last reply -------------------------------------------------

def show_reply(res: dict) -> None:
    if res.get("error"):
        st.error(f"The tutor couldn't finish: {res['error']}")
        return
    with st.container(border=True):
        st.subheader(HEADLINES.get(res["decision"], res["decision"]))
        st.markdown(res["feedback"])
        r = res.get("rubric")
        if r:
            cols = st.columns([1.3, 1, 1, 1, 1])
            cols[0].metric("Score", f"{r['total']}/100")
            for col, (k, v) in zip(cols[1:], r["parts"].items()):
                col.metric(rubric.PART_LABELS[k], v)
        if res["mistakes"]:
            st.markdown("**Mistakes:** " + "; ".join(m["description"] for m in res["mistakes"]))
        st.caption(f"Next: {res['next_assignment_text']}. Why: {res['reason']}")
        with st.expander("What the tutor did"):
            for i, t in enumerate([t for t in res["trace"] if "tool" in t], 1):
                st.markdown(f"{i}. {'✅' if t['ok'] else '❌'} `{t['tool']}`"
                            + (f"  `{json.dumps(t['input'])[:140]}`" if t.get("input") else ""))
            if res.get("target"):
                st.plotly_chart(C.piano_roll(res["target"], pred=res["mistakes"], height=300,
                                             title=f"What you were asked to play ({res['assignment_text']})"),
                                width="stretch")
            u = res.get("usage") or {}
            st.caption(f"{u.get('input_tokens', 0)} tokens in, {u.get('output_tokens', 0)} out, "
                       f"${u.get('cost_usd', 0):.3f}")


last = st.session_state.get("tutor_last", {}).get(learner)
if last:
    show_reply(last)

# ---- assignment -------------------------------------------------------------
st.subheader("Your assignment" if history else "Your first assignment")
st.markdown(f"**Play {describe_assignment(P, assignment)}.**"
            + ("" if history else " The tutor starts everyone on the full melody, then adjusts to how it goes."))
components.html(assignment_html(tgt, "Play this", "Listen first if you like, then play it on your piano or keyboard."),
                height=height_for(tgt), scrolling=True)

# ---- record / upload --------------------------------------------------------
st.subheader("Play it and send it to the tutor")
n = len(history)
tab_rec, tab_up, tab_sim = st.tabs(["🎙️ Record", "📁 Upload a file", "🤖 No piano? Simulate a take"])
with tab_rec:
    rec_round = st.session_state.setdefault("rec_round", 0)   # bumped by "Discard and re-record"
    rec = st.audio_input("Record yourself playing the assignment", key=f"rec_{learner}_{n}_{rec_round}")
with tab_up:
    up = st.file_uploader("A recording (.wav) or a MIDI file from a digital piano (.mid)",
                          type=["wav", "mid", "midi"], key=f"up_{learner}_{n}")
with tab_sim:
    st.caption("For trying the tutor without an instrument: a simulated learner plays your current assignment "
               "(as MIDI) and gets a little better each time they practise the same passage.")
    from examples.rondo_demo import PROFILES
    from examples.rondo_learner import SKILLS
    sim_level = st.selectbox("Simulated learner", [lv for lv in SKILLS if lv != "perfect"],
                             format_func=lambda lv: PROFILES[lv].title, index=2, key="sim_level")
    use_sim = st.checkbox("Use a simulated take", key=f"sim_{learner}_{n}")


def save_input() -> str | None:
    folder = RECORDINGS / learner
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if use_sim:
        from examples.rondo_learner import SimLearner
        from tutor.notes import save_midi
        reps = sum(1 for h in history if h["assignment"]["scope"] == assignment["scope"])
        path = folder / f"{stamp}_simulated_{sim_level}.mid"
        save_midi(SimLearner(sim_level, seed=n).play(tgt, reps), path)
        return str(path)
    src = rec or up
    if src is None:
        return None
    ext = Path(getattr(src, "name", "") or "take.wav").suffix.lower() or ".wav"
    path = folder / f"{stamp}{ext if ext in ('.wav', '.mid', '.midi') else '.wav'}"
    path.write_bytes(src.getvalue())
    return str(path)


have_input = bool(use_sim or rec or up)
is_audio = not use_sim and C.is_audio(getattr(rec or up, "name", "take.wav") if (rec or up) else "x.mid")
blocked = None
if not C.has_key():
    blocked = "Set OPENROUTER_API_KEY in .env so the tutor can listen."
elif have_input and is_audio and not C.audio_extras_installed():
    blocked = 'Recordings need the audio extras: `uv pip install -e ".[audio]"`. MIDI files work without them.'
if blocked:
    st.warning(blocked)

b_send, b_redo, _ = st.columns([1.3, 1.6, 5])
if rec is not None and b_redo.button("Discard and re-record", key="rerecord",
                                     help="Throw this recording away and record again"):
    st.session_state["rec_round"] += 1   # a new key gives an empty recorder
    st.rerun()
if b_send.button("Send to the tutor", type="primary", disabled=not have_input or bool(blocked), key="send"):
    path = save_input()
    try:
        with st.status("The tutor is listening…", expanded=True) as status:
            def on_event(ev):
                if ev["kind"] == "tool":
                    label = {"transcribe_recording": "Transcribing your recording",
                             "self_check": "Checking the transcription",
                             "compare_to_assignment": "Comparing it with your assignment",
                             "score_performance": "Scoring it",
                             "get_learner_history": "Looking at how earlier attempts went",
                             "preview_version": "Trying out a different version for you",
                             "get_assignment": "Reading your assignment",
                             "finish_turn": "Writing your feedback"}.get(ev["name"], ev["name"])
                    st.write(("✅ " if ev["ok"] else "❌ ") + label)
            res = run_tutor_turn(learner, path, P, on_event=on_event)
            status.update(label="The tutor has replied", state="complete" if not res.get("error") else "error",
                          expanded=False)
        st.session_state.setdefault("tutor_last", {})[learner] = res
        st.rerun()
    except llm.BudgetExceeded as e:
        common.budget_banner(e)
    except Exception as e:  # keep the page usable
        st.error(f"{type(e).__name__}: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

# ---- progress ---------------------------------------------------------------
if history:
    with st.expander(f"Your progress ({len(history)} attempt{'s' * (len(history) != 1)})"):
        scores = [h["score"] for h in history if h.get("score") is not None]
        if scores:
            st.line_chart({"score": scores}, height=180)
        st.dataframe([{"#": i, "played": describe_assignment(P, h["assignment"]), "score": h.get("score"),
                       "tutor's call": HEADLINES.get(h["decision"], h["decision"])}
                      for i, h in enumerate(history, 1)], hide_index=True, width="stretch")
        recurring = sorted(memory.read(learner, P.name), key=lambda e: -e["count"])[:5]
        if recurring:
            st.markdown("**Mistakes that keep coming back:** " + "; ".join(
                f"{e.get('note') or e['type']} ({e['count']}×)" for e in recurring))
        if st.button("Start over", key="reset"):
            memory.reset(learner, P.name)
            st.session_state.get("tutor_last", {}).pop(learner, None)
            st.rerun()
