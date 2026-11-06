import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import streamlit as st  # noqa: E402

from gui import common  # noqa: E402
from gui import components as C  # noqa: E402
from gui import jobs, sources  # noqa: E402
from tutor import llm  # noqa: E402

common.page("Eval", "📊")
st.title("📊 Evaluation runs")

sources.rondo_takes()  # builds examples/rondo/ if it's missing
sets = sources.eval_sets()
set_name = st.selectbox("Eval set", list(sets), key="eval_set",
                        help="Rondo alla Turca learners: 10 takes at 4 levels with known, audible mistakes. "
                             "Perturbation set: random errors injected into MAESTRO (or bundled) excerpts.")
manifest = sets[set_name]
rows = sources.read_manifest(manifest)
if not sources.MANIFEST.exists():
    with st.expander("No perturbation set at `data/cases/manifest.jsonl` yet: build one"):
        st.markdown("For the real set, download MAESTRO and run `python -m data.make_perturbations` (see the README). "
                    "Or build a **small offline set** from the bundled example (MIDI only, about a second).")
        if st.button("Build small perturbation set from the bundled example"):
            r = subprocess.run([sys.executable, "-m", "data.make_perturbations", "--midi-dir", "examples",
                                "--n-excerpts", "2", "--severities", "0", "1", "3", "--per-severity", "2"],
                               cwd=sources.ROOT, capture_output=True, text=True)
            (st.success if r.returncode == 0 else st.error)(r.stdout + r.stderr)
            st.rerun()

has_audio = any("perf_audio" in r for r in rows)
st.caption(f"Eval set: {len(rows)} cases"
           f"{' (with audio)' if has_audio else ' (MIDI only, so build with --audio for the audio setting)'}.")

# ---- live job ---------------------------------------------------------------


@st.fragment(run_every="3s")
def live_panel():
    run_dir = st.session_state.get("watch_run") or jobs.active_job()
    if run_dir is None:
        return
    s = jobs.status(Path(run_dir))
    st.subheader(f"Run `{Path(run_dir).name}`: {s['state']}")
    total = s["total"] or max(s["done"], 1)
    st.progress(min(s["done"] / total, 1.0), text=f"{s['done']} / {total} cases")
    c = st.columns(3)
    if s["spend"] is not None:
        c[0].metric("Spent this run", f"${s['spend']:.3f}")
    if s["elapsed"] is not None:
        c[1].metric("Elapsed", f"{s['elapsed']:.0f} s")
    if s["last"]:
        last = s["last"]
        c[2].metric("Last case", f"{len(last.get('errors', []))} predicted",
                    help=last["case_id"] + (f" · ERROR {last['error']}" if last.get("error") else ""))
    if s["state"] == "running":
        if st.button("⏹ Stop run", type="secondary"):
            jobs.stop(Path(run_dir))
            st.toast("Stop signal sent. Finished cases are kept.")
    else:
        if st.button("Dismiss"):
            st.session_state.pop("watch_run", None)
            st.rerun(scope="app")
    with st.expander("Log"):
        st.code(s["log_tail"] or "(empty)")


live_panel()

# ---- launch -----------------------------------------------------------------
st.subheader("Launch a run")
running = jobs.active_job()
with st.form("launch"):
    c = st.columns(4)
    systems = C.visible_systems()
    system = c[0].selectbox("System", systems, format_func=lambda s: C.SYSTEM_LABELS[s])
    input_kind = c[1].selectbox("Input", ["midi", "audio"] if has_audio else ["midi"],
                                help="midi = exact transcription (upper bound); audio = realistic")
    limit = c[2].number_input("Cases (limit)", min_value=1, max_value=len(rows), value=len(rows))
    name = c[3].text_input("Run name", value=f"gui_{time.strftime('%m%d-%H%M%S')}")
    est = jobs.estimate_cost(system, int(limit))
    spent, cap = llm.total_spend(), llm.budget()
    st.caption(
        ("Estimated cost: " + (f"**${est:.2f}**" if est is not None else "**unknown** (no earlier runs of this system)"))
        + (f" · Budget left: **${cap - spent:.2f}** of ${cap:.2f}" if cap else "")
        + "  \n_The estimate updates when you change the system or limit and press Launch. Launching starts the run._")
    submitted = st.form_submit_button("🚀 Launch", type="primary", disabled=running is not None)
if running is not None:
    st.caption(f"A run is in progress (`{running.name}`), and only one runs at a time.")
if submitted:
    if system in C.LLM_SYSTEMS and not C.has_key():
        st.error("Set OPENROUTER_API_KEY in .env to run LLM systems.")
    elif input_kind == "audio" and not C.audio_extras_installed():
        st.error('Audio input needs the audio extras: `uv pip install -e ".[audio]"`.')
    else:
        try:
            run_dir = jobs.launch(name, system, input_kind, limit=int(limit), manifest=manifest)
            st.session_state["watch_run"] = str(run_dir)
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))

# ---- results ----------------------------------------------------------------
st.divider()
st.subheader("Results")
runs = jobs.list_runs()
if not runs:
    st.caption("No runs yet.")
    st.stop()

tab_one, tab_cmp = st.tabs(["One run", "Compare runs"])
with tab_one:
    run_dir = st.selectbox("Run", runs, format_func=lambda d: f"{d.name}  [{jobs.run_system(d)}]")
    data = jobs.load_run(run_dir)
    sc = data["scores"]
    if not sc:
        st.caption("Nothing to grade yet.")
    else:
        if sc.get("partial"):
            st.warning("This run didn't finish. The scores cover only the cases it completed.")
        o = sc["overall"]
        c = st.columns(6)
        c[0].metric("Cases", sc["n_cases"])
        c[1].metric("Precision", f"{o['precision']:.2f}")
        c[2].metric("Recall", f"{o['recall']:.2f}")
        c[3].metric("F1", f"{o['f1']:.2f}")
        c[4].metric("Localization", f"{sc['localization_acc']:.2f}" if sc["localization_acc"] is not None else "—")
        c[5].metric("FP / clean case", sc["false_positives_per_clean_case"]
                    if sc["false_positives_per_clean_case"] is not None else "—")
        a, b = st.columns(2)
        a.markdown("**By error type**")
        a.dataframe([{"type": t, **v} for t, v in sc["by_type"].items()], hide_index=True, width="stretch")
        b.markdown("**By severity (number of injected errors)**")
        b.dataframe([{"severity": s, **v} for s, v in sc["by_severity"].items()], hide_index=True,
                    width="stretch")
        if sc["failed_runs"]:
            st.caption(f"{sc['failed_runs']} case(s) failed with an error.")

    by_id = {r["case_id"]: r for r in data["rows"]}
    table = []
    for p in data["preds"]:
        r = by_id.get(p["case_id"])
        if not r:
            continue
        m = C.score(r["errors"], p.get("errors", []))["metrics"]
        table.append({"case": p["case_id"], "severity": r["severity"], "known": len(r["errors"]),
                      "predicted": len(p.get("errors", [])), "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                      "cost": round(jobs.case_cost(p), 4), "sec": p.get("seconds"), "error": p.get("error", "")})
    levels = {}
    for t in table:
        lv = by_id[t["case"]].get("level")
        if lv:
            agg = levels.setdefault(lv, [0, 0, 0])
            agg[0] += t["tp"]; agg[1] += t["fp"]; agg[2] += t["fn"]
    if levels:
        from eval.grade_detection import prf
        from examples.rondo_demo import PROFILES
        st.markdown("**By learner level**")
        st.dataframe([{"level": PROFILES[lv].title if lv in PROFILES else lv, **prf(*levels[lv])}
                      for lv in PROFILES if lv in levels], hide_index=True, width="stretch")
    if table:
        st.markdown("**Cases**")
        st.dataframe(table, hide_index=True, width="stretch")
        pick = st.selectbox("Open a case in Assess (uses the stored prediction, so no new LLM call)",
                            [t["case"] for t in table])
        if st.button("Open in Assess"):
            r, p = by_id[pick], next(p for p in data["preds"] if p["case_id"] == pick)
            job = json.loads((run_dir / "job.json").read_text()) if (run_dir / "job.json").exists() else {}
            audio = job.get("input") == "audio" or "_audio" in run_dir.name
            src = sources.case_by_id(pick, Path(job["manifest"]) if job.get("manifest") else sources.MANIFEST)
            perf = (src.perf_audio if audio else src.perf_midi) if src else None
            st.session_state["assess_preload"] = {
                "out": p, "system": data["system"], "run": run_dir.name,
                "src": {"label": src.label if src else pick, "ref": src.ref if src else r["ref"],
                        "perf": perf or r["perf_midi"], "gt": r["errors"], "piece": src.piece if src else None}}
            st.switch_page("views/assess.py")

with tab_cmp:
    chosen = st.multiselect("Runs", runs, default=runs[: min(3, len(runs))],
                            format_func=lambda d: f"{d.name}  [{jobs.run_system(d)}]")
    cmp_rows = []
    for d in chosen:
        sc = jobs.load_run(d)["scores"]
        if not sc:
            continue
        row = {"run": d.name, "system": jobs.run_system(d), "cases": sc["n_cases"],
               **{k: sc["overall"][k] for k in ("precision", "recall", "f1")},
               "loc": sc["localization_acc"], "fp/clean": sc["false_positives_per_clean_case"]}
        row.update({f"F1 {t}": v["f1"] for t, v in sc["by_type"].items() if t in C.TYPE_COLORS})
        cmp_rows.append(row)
    if cmp_rows:
        st.dataframe(cmp_rows, hide_index=True, width="stretch")
        st.bar_chart({r["run"]: r["f1"] for r in cmp_rows}, height=240)
