# Tutor Console GUI: Design Spec (2026-11-05)

Status: design approved in brainstorming. Next step: an implementation plan.

## Goal
A local GUI that runs every tool the repo has today and shows how the system is doing. It has two jobs:

1. **Dev console.** We use it to run assessments, look at agent traces, compare predictions with
   known mistakes, launch eval runs and track spending.
2. **Demo-ready.** It should be clean enough to screen-record for the paper video (Nov 13) and to show at the
   December live demo.

**Success:** from one `streamlit run gui/app.py`, one of us can:
- assess the bundled example with B0, B2 or the agent, and watch the agent's tool calls live
- see the errors on a piano roll next to the known mistakes, with P/R/F1
- hear the reference and the performance
- build and hear the simplification ladder
- inspect and clear the mistake log
- launch, watch, stop and compare eval runs with a cost estimate shown first
- see spending against the budget

## Scope decisions
- Users: the two of us, run locally. Not hosted, no auth.
- Input: bundled examples, generated eval cases (`data/cases/manifest.jsonl`), or uploaded `.mid`/`.wav`.
  No recording in the browser and no MIDI-keyboard capture.
- Eval runs can be launched from the GUI (not view-only), and the budget caps still apply.
- Framework: **Streamlit + Plotly**, Python only. We rejected Gradio (weaker for multi-page apps and background
  jobs) and FastAPI with a custom frontend (2–3× the code, and a JS frontend to maintain).
- The GUI calls the existing functions. The one change to the core code is an optional `on_event`
  callback in `run_agent`.

## Layout
```
gui/
  app.py            entry point; sidebar: model, total spend vs TUTOR_BUDGET_USD, env/key check
  pages/
    1_Assess.py
    2_Simplify.py
    3_Memory.py
    4_Eval.py
    5_Spending.py
  components.py     piano roll, audio player, error table, trace timeline, result scoring
  sources.py        list inputs (bundled example, eval cases, uploads) and their ground truth
  jobs.py           launch, track and stop background eval subprocesses
```

## Pages

### Assess (centrepiece)
- **Input:** a reference and a performance, chosen from the bundled example or an eval case, or uploaded.
  Ground truth loads automatically when the source has it (`examples/perf_errors.json`, the manifest's `errors`).
- **System:** agent / B0 algorithmic / B2 text LLM. Model and effort can be overridden for this run.
  Learner name, plus a "write to memory" checkbox (on by default for the agent, off for baselines).
- **While running (agent):** an `st.status` timeline with one row per tool call. Each row expands to show the
  inputs and a short output. Failed tool calls are red. Turns and cost so far are shown.
- **Results:**
  - Piano roll: reference notes as grey bars, performance notes in colour. Predicted errors are marked, and
    known errors are drawn as rings, so hits, misses and false positives stand out.
  - Error table (time, type, pitch/note name, played pitch, explanation) and, when ground truth exists,
    P/R/F1 computed with `eval.grade_detection.match`.
  - The feedback text, and audio players for the reference and the performance.
  - Usage: tokens, cost, cached calls.
- **Opened from the Eval page:** a case can be opened with a stored prediction, so the page shows it without a new LLM call.

### Simplify
Pick a piece, then build the ladder. The step order can be changed (the default is the CLI ladder). Each rung shows a
piano roll, `difficulty()` scores, an audio player and a `.mid` download. A small chart shows how each difficulty score
changes from rung to rung.

### Memory
Pick a learner (from the files in `memory_store/`) to see the log as a table grouped by piece, with counts and
first/last seen. Clearing a learner's log needs a confirmation.

### Eval
- **Launch:** system, input (midi/audio), backend, `--limit` and a run name. Before starting, it shows a
  **cost estimate** (cases × average cost per case for that ledger tag, or "unknown" if there's no history) and
  the remaining budget.
- **Live:** progress (lines in `predictions.jsonl` / number of cases), the last case's result, spend since
  start, and a **Stop** button. Only one job runs at a time.
- **Results:** pick a past run from `runs/*/scores.json` to see overall P/R/F1, tables by type and by severity,
  localization accuracy and false positives per clean case. There's a table per case, and clicking a case opens Assess with
  that prediction.
- **Compare:** several runs side by side (overall F1 and F1 by type).

### Spending
Ledger tables grouped by tag and model, plus a cumulative-spend chart with the budget line.

## Data flow
- Pages call `run_agent`, `tutor.baselines.{algorithmic,text_llm}.run`, `find_errors`,
  `tutor.tools.simplify`, `tutor.tools.memory` and `tutor.llm.read_ledger`/`total_spend` directly.
- Audio is produced by `tutor.tools.synth.synthesize`, encoded to WAV bytes in memory and played with `st.audio`.
  With no soundfont it falls back to sine waves.
- Uploads are saved under `runs/gui_uploads/<timestamp>/` (git-ignored, since it's under `runs/`).
- **Agent hook:** `run_agent(..., on_event: Callable[[dict], None] | None = None)` emits
  `{"kind": "llm", "turn", "cost_usd", "cached"}` after each model call, and
  `{"kind": "tool", "name", "input", "ok", "output_preview" | "error"}` after each tool call. With no callback,
  behaviour and return value are unchanged.

## Background eval jobs (`gui/jobs.py`)
- **Launch:** `subprocess.Popen([sys.executable, "-m", "eval.run_eval", ..., "--out", "runs/<name>"])`, with stdout
  and stderr sent to `runs/<name>/job.log`. It writes `runs/<name>/job.json` with the PID, arguments, start time and the
  ledger total at start.
- **Status:**
  - *Running:* the process is alive. *Finished:* `scores.json` exists. *Stopped or failed:* the process is gone and
    there is no `scores.json`.
  - Progress is counted from `predictions.jsonl` lines. Spend is the current ledger total minus the total at start.
  - The page refreshes every ~3 s while a job runs.
- **Stop:** sends SIGTERM to the process. The cases already written stay, and the results view can grade them
  (using `grade_detection.grade` on the matching rows of the manifest).
- **Only one job at a time:** a new launch is refused while any `job.json` points to a live PID.

## Error handling
- **No `OPENROUTER_API_KEY`:** the agent and B2 are disabled with a hint. B0, Simplify, Memory and the result views still work.
- **`BudgetExceeded`:** a banner shows spent vs cap and where to raise the cap. When the agent hits its per-run cap, the
  partial trace and an error badge are shown.
- **Audio extras missing and a `.wav` is chosen:** an install hint (`uv pip install -e ".[audio]"`) is shown instead of a traceback.
- **Any other exception in a run:** an error box shows the message, with the traceback in an expander. The page stays usable.

## Testing
- **Unit tests (offline):**
  - piano-roll figure construction (trace counts, error markers)
  - result scoring
  - `sources` case listing from a temporary manifest
  - `jobs` status and progress parsing from a fake run directory
  - the `on_event` hook, using the fake `llm.chat` from `tests/test_llm.py`
- **Smoke tests with `streamlit.testing.v1.AppTest`:** every page loads without an exception, and a B0 assessment on the
  bundled example finds the 3 known errors. No network.
- **Live check before handoff:** one real agent run in the GUI on the bundled example (about $0.02).

## Dependencies
New optional extra `gui = ["streamlit>=1.40", "plotly>=5.20"]`. The README gets a "GUI" section.

## Out of scope
Recording in the browser, MIDI-keyboard input, hosting or auth, editing scores in the GUI, the lesson-loop
orchestrator, and the B1 audio-LLM baseline. B1 is still a stub, so the Assess and Eval pages hide `audio_llm` until it is
implemented. To show it, remove it from a `HIDDEN_SYSTEMS` set in `gui/components.py`.
