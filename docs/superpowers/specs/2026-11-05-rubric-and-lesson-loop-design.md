# Performance Rubric and Chunked Practice Loop: Design Spec (2026-11-05)

Status: design approved in brainstorming. Next step: an implementation plan.

## Goal
Turn the assessor into a tutor that acts on how good a performance was:

- **Close to right:** give precise, note-by-note feedback.
- **Far off:** give a simplified version of the same passage.
- **Very far off:** split the piece into small chunks, simplify the first one, and run a practice loop. After each
  attempt the tutor decides whether to repeat the chunk, make it easier, make it harder, or move on.

This is the proposal's closed loop (design spec 2026-10-09, data-flow steps 2–5). It is also the setup for E4 with
simulated learners: does an LLM coach get a learner to the full piece in fewer attempts than fixed rules?

**Success:**
- Each assessment gets a 0–100 rubric score, a band and the matching action.
- A simulated learner can be taken through a whole Rondo lesson under either policy, and every decision is logged.
- One command compares the two policies across learner levels.

## Scope decisions
- **Attempts:** come from **simulated learners** first. Uploading real attempts is future work, and the interfaces
  allow for it.
- **Bands:** three, with the cut-offs as settings. See *Rubric*.
- **Progression:** **climb, then advance.** Climb the current chunk's ladder up to the full version before starting the
  next chunk. Step down a rung when the learner keeps failing.
- **Approach:** a Python lesson controller plus **one LLM coach call per attempt** (`policy="llm"`), with a
  **rules-only baseline** (`policy="rules"`). We rejected running the whole lesson inside the existing tool-use agent:
  the conversation keeps growing, it costs more, and it is hard to control and evaluate.
- **Target:** the **one-note melody** (right hand, top note of each interval), as on the practice sheet. Two-hand
  lessons are out of scope.
- **Piece:** the Rondo alla Turca excerpt (pickup + bars 1–8). Chunking also has a time-window fallback for pieces with
  no meter.
- **Depends on:** the practice-sheet work (`drop_ornaments`, `thin_to_grid` in `tutor/tools/simplify.py`, and
  `examples/rondo_practice.py`), which should be committed first.

## Rubric (`tutor/rubric.py`)
`score(target: list[Note], errors: list[dict], perf: list[Note] | None = None, bar_beat=None) -> Rubric`

| Part | Formula (n = notes in target) | Weight |
|---|---|---|
| Right notes | 1 − wrong_pitch / n | 0.35 |
| Complete | 1 − missed / n | 0.30 |
| Rhythm | 1 − timing / n | 0.20 |
| Clean | 1 − min(extra, n) / n | 0.15 |

- Each part is clamped to [0, 1] and reported ×100. `total` is the weighted sum, rounded to an integer.
- `tempo_ratio` (performance duration ÷ target duration, only when `perf` is given) is reported but not scored.
- `worst_bars`: the bars with the most errors, for example `[{"bar": 3, "errors": 2, "types": {"wrong_pitch": 2}}]`,
  when `bar_beat` is given.
- `band(total)`: **polish** if ≥ `POLISH_AT` (85), **simplify** if ≥ `CHUNK_BELOW` (50), otherwise **chunk**. Both
  cut-offs are module constants.
- The rubric is computed **the same way every time** from the error list. It never calls an LLM.

**On the Assess page:** after any system runs, a rubric card shows the total, the four parts, the band and the action:
- **polish:** the system's feedback, plus the error table.
- **simplify:** the simplified version of the passage (from the practice-sheet pipeline) with audio, and a link to
  the practice sheet.
- **chunk:** a **Start a lesson** button that opens the Lesson page with this learner level selected.

## Chunks and rungs (`tutor/lesson.py`)
- **`Piece`:** the melody notes, plus `sec_per_beat`, `pickup_beats` and `beats_per_bar` (optional). The Rondo piece is
  built from `examples/rondo_practice.full_melody()` and the meter in `examples/rondo_demo`.
- **`chunks(piece, bars_per_chunk=2)`:** gives the Rondo 4 chunks (pickup + bars 1–2, bars 3–4, 5–6, 7–8). Every note
  belongs to exactly one chunk. With no meter it falls back to windows of about 4 s.
- **Rungs per chunk, easiest first:**
  1. `simplified`: drop_ornaments → thin_to_grid (eighths) → 60 bpm
  2. `full_slow`: every note, 60 bpm
  3. `full`: every note, 90 bpm (the goal)
- **Target times:** each target is rebased to start at 0, so an attempt only has to cover the chunk.

## Practice loop
```
start (chunk band): chunk 0, rung 0
loop:
  attempt = learner.play(target(chunk, rung))
  errors  = find_errors(target, attempt)             # oracle MIDI
  rub     = rubric.score(target, errors, bar_beat)
  action, feedback, reason = policy.decide(state, rub, history)
  apply guardrails -> next (chunk, rung) or finish
```
- **Actions:** `repeat`, `easier` (rung −1), `harder` (rung +1), `next_chunk` (next chunk at rung 0), `finish`.
- **Guardrails, for both policies:**
  - `harder` at the top rung becomes `next_chunk`. `next_chunk` on the last chunk becomes `finish`. `easier` at
    rung 0 becomes `repeat`.
  - At most `MAX_ATTEMPTS_PER_CHUNK = 8`. After that, force `next_chunk` (or `finish`) with the reason
    "attempt cap".
  - `llm.BudgetExceeded` stops the lesson. The log is kept and the lesson is marked unfinished.

### Policies
- **`rules` (baseline):**
  - score ≥ 85 → `harder` (which becomes `next_chunk` or `finish` at the top rung)
  - score < 50 on two attempts in a row at this rung → `easier`
  - otherwise → `repeat`

  Feedback is the rubric's worst bars, as a template.
- **`llm` (coach):** one `llm.chat` call per attempt, tag `coach`, JSON-schema output
  `{action: enum, feedback: str, reason: str}`. The prompt includes:
  - the chunk (k of N) and the rung (name and description)
  - the rubric parts and band
  - the errors in bar.beat form
  - the last up-to-5 scores on this chunk
  - the rules policy's suggestion

  The coach may override the suggestion but must say why. Invalid JSON or an invalid action → use the rules
  suggestion, logged as `fallback: true`.

## Simulated learner (`examples/rondo_learner.py`)
- Builds on the `examples/rondo_demo.PROFILES` levels and its obvious-mistake generator, applied to a chunk's target
  notes.
- **Chance of a mistake per note** = `base[level] × difficulty(rung) × exp(−rate[level] × reps_on_chunk)`.
  - `difficulty` grows with notes per second and with grace notes.
  - `reps_on_chunk` counts every attempt at this chunk, at any rung.
- **Weak spots:** each (level, chunk) has fixed weak notes that are wrong while `reps_on_chunk < weak_reps[level]`.
- **Seeds:** `random.Random(f"{level}:{seed}:{chunk}:{rung}:{reps}")`, so both policies get the same attempt for the
  same state. Tempo and jitter are as in `PROFILES`.

## Logging
- `runs/lessons/<lesson_id>.jsonl` gets one row per attempt:
  - `{attempt, chunk, rung, rubric, errors, action, suggested, fallback, feedback, reason, cost_usd}`
  - plus a final `{"summary": ...}` row.
- The lesson id is `<level>_<policy>_s<seed>_<timestamp>`.

## GUI: Lesson page (`gui/pages/6_Lesson.py`)
- **Setup:** learner level, policy (the LLM option is off without a key), seed. **Step** runs one attempt; **Run to end**
  runs them all. State lives in `st.session_state`.
- **Timeline:** a Plotly chart of score per attempt, coloured by chunk, with rung changes marked and lines at 85 and 50.
- **Attempt rows:** chunk and rung, total and parts, action (and the suggestion if it differs), reason, feedback, and
  audio for the target and the attempt.
- **Summary:** chunks finished, total attempts, attempts per chunk, final score on the `full` rung, cost.
- **Past lessons:** a list of earlier lessons from `runs/lessons/`.

## Evaluation (`eval/run_lessons.py`)
`python -m eval.run_lessons --policy rules|llm --levels nearly_there intermediate struggling --seeds 5`
- Writes `runs/lessons/summary_<policy>.json` with these metrics, by level:
  - attempts until all chunks pass the `full` rung
  - the share of lessons finished within the caps
  - the mean final `full`-rung score
  - wasted attempts (a repeat that didn't raise the score)
  - coach fallbacks
  - cost
- The `rules` policy runs offline for free. The `llm` policy respects `TUTOR_BUDGET_USD`.

## Testing (offline)
- **Rubric:**
  - the perfect take scores 100
  - the weights add up correctly on a hand-made error list
  - the band cut-offs
- **Chunks:** 4 chunks covering every note exactly once. `full` is harder than `simplified` (notes per second).
- **Simulated learner:**
  - the same seed and state give the same attempt
  - the mean score rises with `reps_on_chunk`
  - a stronger level scores higher
- **Rules policy:** finishes a lesson for `nearly_there` and `intermediate`. The guardrail conversions work.
- **LLM policy, with a fake `llm.chat`:**
  - a valid action is followed
  - invalid JSON or an invalid action falls back to rules
  - `harder` at the top rung → `next_chunk`
- **Lesson page:** an AppTest smoke test that steps through rules-policy attempts without an exception.
- **Live check:** one `llm` lesson for a struggling learner, about 15–25 coach calls, about $0.10–0.20.

## Out of scope
Real recorded or uploaded attempts, two-hand lessons, learning the cut-offs from data, lesson plans across sessions
stored in memory (lessons do not write the mistake log), and pieces other than the Rondo.
