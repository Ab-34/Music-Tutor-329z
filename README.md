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
tutor/llm.py              OpenRouter client, budget ledger, cache (TUTOR_MODEL, TUTOR_BUDGET_USD)
data/make_perturbations.py  eval set builder
eval/                     run_eval, grade_detection, grade_transcription
examples/                 bundled quick-start example (+ make_example.py to regenerate it)
```

## Notes and known limitations

- MAESTRO is *performance* MIDI with no reliable bar lines, so we locate errors in seconds,
  not measures.
- Alignment is v0: a global linear time map plus greedy same-pitch matching. Real players vary
  their tempo locally; DTW-based alignment is the planned upgrade.
- With `--input midi`, the algorithmic baseline is already near-perfect on synthetic
  perturbations. The agent should add value on audio, where transcription errors look like
  player errors.
- Synthesized audio is cleaner than real recordings. We also evaluate on a small set of real
  recordings we made ourselves.
