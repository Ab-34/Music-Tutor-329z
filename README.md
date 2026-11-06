# Agentic Music Tutor

CS329Z course project (Abhijnya Bhat, Chinmay Pimpalkhare).

An LLM orchestrator listens to a learner play a piece, finds and locates their mistakes
against a reference score, remembers recurring mistakes across sessions, and builds a
ladder of simpler arrangements. The headline evaluation is **perturbation detection**: we
inject known errors into reference performances and measure how well each system recovers them.

## Quick start (about 2 minutes, no dataset download)

Requires Python 3.10 or 3.11 and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone <this repo> music-tutor && cd music-tutor
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

**1. Offline check (no API key needed).** This runs the algorithmic baseline on the bundled example,
which has 3 known mistakes listed in `examples/perf_errors.json`:

```bash
python -m tutor.cli assess --ref examples/ref.mid --perf examples/perf.mid --no-llm
```

**2. Run the agent** (needs an [OpenRouter](https://openrouter.ai) API key):

```bash
cp .env.example .env        # then put your key in OPENROUTER_API_KEY
python -m tutor.cli assess --ref examples/ref.mid --perf examples/perf.mid --learner demo
python -m tutor.cli memory --learner demo      # the mistakes the agent logged
```

Run the `assess` command a second time and the feedback should mention the recurring mistakes.

**3. Simplification ladder:**

```bash
python -m tutor.cli simplify --ref examples/ref.mid --out runs/simplified
```

**4. Tests:** `pytest -q`

## Models and budget

All LLM calls go through `tutor/llm.py` to OpenRouter's OpenAI-compatible API. `TUTOR_MODEL` takes any
OpenRouter model id. The default is `anthropic/claude-sonnet-5.5`.

- **Total cap:** every paid call is logged to `llm_ledger.jsonl`. Once the total reaches `TUTOR_BUDGET_USD`,
  new calls raise `BudgetExceeded` and `eval.run_eval` stops and grades the cases it finished.
- **Cap per agent run:** `TUTOR_RUN_BUDGET_USD` (default $0.50) stops a single assessment that loops for too long.
- **Cache:** identical requests are answered from `.llm_cache/` for free. Set `TUTOR_CACHE=0` to turn it off.
- **Report:** `python -m tutor.llm` shows spending by tag and model.

For a hard limit on the provider side as well, set a credit limit on the key in the OpenRouter dashboard.

## GUI

```bash
uv pip install -e ".[gui,audio]"     # audio = Basic Pitch transcription, for recordings
streamlit run gui/app.py             # opens http://localhost:8501
```

**Learn the Rondo** is the app: one page where the tutor agent does everything. You see your current assignment as
piano-key tiles you can listen to. You play it, then **record it in the browser** or upload a `.wav` or `.mid`, and
send it to the tutor. The agent (`tutor/tutor_agent.py`):
1. transcribes the recording and checks the transcription
2. aligns it to the assignment and scores it with the rubric
3. reads your history: earlier attempts and mistakes that keep coming back
4. previews easier or harder versions if it wants to
5. decides what you do next: precise feedback if you're close, an easier version of the whole piece if you're partway
   there, or a beginner lesson in 2-bar chunks if you're far off

Inside a lesson it repeats a chunk, makes it easier or harder, or moves to the next chunk. Progress and mistakes are
saved per learner name (`memory_store/`), so the next recording, or the next session, carries on where you left off.
There's also a "simulate a take" option for trying it without an instrument.

The **Developer** section of the sidebar has: Assess (run one system on a take and compare it with the known
mistakes), Eval runs, and Spending.

## Demo learners: Rondo alla Turca

`examples/rondo/` holds the pickup and the first 8 bars of Mozart's *Rondo alla Turca* (K. 331), taken from the
public-domain [Mutopia Project](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=108) score MIDI (see
`examples/rondo/source/SOURCE.md`) and played at 90 bpm. There are 10 takes by learners at four levels, each mistake
labelled in plain English (for example "Bar 3, beat 1: played D5 instead of B5"):

| Level | Mistakes per take | Notes |
|---|---|---|
| Perfect | 0 | Checks that the system doesn't invent errors |
| Nearly there | 1 | One clear slip |
| Intermediate | 5–6 | Wrong and skipped notes, a hesitation, plays 10% slower |
| Struggling | 12–14 | Plays 30% slower, with many wrong and skipped notes, hesitations and stray notes |

Each level has fixed *weak spots* that repeat in every take, so the agent's memory has something to notice. The takes
are the GUI's default input, and you can listen before revealing what's wrong. They also form an eval set
(`examples/rondo/manifest.jsonl`). To regenerate them: `python -m examples.rondo_demo`.

## Rubric and practice lessons

The tutor scores every attempt with a rubric (`tutor/rubric.py`). It has four parts: right notes 35%, complete 30%,
rhythm 20%, clean 15%. Each part drops to 0 once 5% of the notes (at least 2) have that kind of mistake. The bands
guide the agent's decision:

| Score | Band | What the tutor usually does |
|---|---|---|
| ≥ 85 | polish | Precise note-by-note feedback |
| 50–84 | simplify | An easier version of the same passage |
| < 50 | chunk | A beginner lesson: 2-bar chunks, starting simplified |

Levels per passage (`tutor/lesson.py`):

| Level | Notes | Tempo |
|---|---|---|
| beginner | runs thinned, no grace notes | 60 bpm |
| intermediate | all notes | 60 bpm |
| full | all notes | 90 bpm |

To compare the agent with a fixed-rules baseline on simulated learners who improve with practice
(`examples/rondo_learner.py`):

```bash
python -m eval.run_lessons --policy rules --seeds 5                       # free
python -m eval.run_lessons --policy agent --levels struggling --seeds 1   # one agent turn per attempt, ~$0.04 each
```

## Audio path (transcription, synthesis, self-check)

```bash
brew install fluid-synth                      # macOS; apt install fluidsynth on Linux
uv pip install -e ".[audio]"
mkdir -p data/soundfonts                      # put a General MIDI .sf2 here, e.g. FluidR3_GM.sf2
```

Set `SOUNDFONT=` in `.env`. Without a soundfont, synthesis falls back to sine waves.
Then pass a `.wav` as `--perf` and the agent will transcribe it with Basic Pitch.

## Reproducing the evaluation

**Data.** Download the MAESTRO v3 **MIDI-only** package (small, unlike the full audio set) from
<https://magenta.withgoogle.com/datasets/maestro> and unzip it into `data/raw/maestro/`.

```bash
# 30 excerpts x 20 s; severity 0 (clean), 1, 3, 6 errors; 3 variants each -> 300 cases
python -m data.make_perturbations --midi-dir data/raw/maestro --n-excerpts 30 \
    --severities 0 1 3 6 --per-severity 3 --audio
```

**Systems** (`--input midi` uses exact transcription and is the upper bound; `--input audio` is the realistic setting):

| System | Command |
|---|---|
| B0 algorithmic (no LLM) | `python -m eval.run_eval --system algorithmic --input audio` |
| B1 one-shot audio LLM | `python -m eval.run_eval --system audio_llm --input audio` *(not implemented yet)* |
| B2 single LLM call on text | `python -m eval.run_eval --system text_llm --input audio` |
| Agent | `python -m eval.run_eval --system agent --input audio` |

Add `--limit N` for a quick run. Each run writes `runs/<system>_<input>[_<backend>]/` with
`predictions.jsonl` (including agent tool traces and token usage) and `scores.json`.

**Metrics** (`eval/grade_detection.py`):

- Precision, recall and F1 per error type and per severity. A prediction counts when the
  type and pitch match and the time is within 100 ms.
- Localization accuracy: a prediction lands within 500 ms of the true error.
- False positives per clean case: the noise floor.

Transcription quality on its own: `python -m eval.grade_transcription --manifest data/cases/manifest.jsonl`
(mir_eval note F1).

## Layout

```
tutor/notes.py            Note type, MIDI I/O
tutor/tools/              transcribe, synth, self_check, align (+error candidates), score_text, memory, simplify
tutor/agent.py            orchestrator: tool-use loop (OpenRouter), logs a trace per run
tutor/baselines/          algorithmic (B0), audio_llm (B1, stub), text_llm (B2)
gui/                      Streamlit app (app.py + views/: tutor, assess, eval, spending)
tutor/llm.py              OpenRouter client, budget ledger, cache (TUTOR_MODEL, TUTOR_BUDGET_USD)
data/make_perturbations.py  eval set builder
eval/                     run_eval, grade_detection, grade_transcription
examples/                 bundled quick-start example (+ make_example.py to regenerate it)
```

## Notes and known limitations

- MAESTRO is *performance* MIDI with no reliable bar lines, so we locate errors in seconds,
  not measures.
- Alignment is v0: a global linear time map (started from the ratio of the two durations, so a uniformly slower player is handled) plus greedy same-pitch matching. Real players vary
  their tempo locally; DTW-based alignment is the planned upgrade.
- With `--input midi`, the algorithmic baseline is already near-perfect on synthetic
  perturbations. The agent should add value on audio, where transcription errors look like
  player errors.
- Synthesized audio is cleaner than real recordings. We also evaluate on a small set of real
  recordings we made ourselves.
