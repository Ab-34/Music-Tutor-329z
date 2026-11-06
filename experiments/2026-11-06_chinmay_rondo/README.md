# Session: Chinmay learns the Rondo alla Turca (2026-11-06, 00:02–00:17)

Chinmay used the Tutor page (`streamlit run gui/app.py`) with real microphone recordings: **16 attempts in about 15
minutes**. That was 58 paid LLM calls plus 6 cache hits, costing **$0.62** (about $0.039 per attempt), on
`anthropic/claude-sonnet-5.5` with medium effort at commit `b04f01c`, plus the re-record button.

## What's in this folder

| Path | What it is |
|---|---|
| `index.json` | One row per attempt: assignment, score, band, decision, next assignment, reason, feedback, tools, LLM calls, cost, and file paths |
| `turns/turn_NN.json` | Full transcript of each attempt: every message (system prompt, tool calls, tool results), the tool trace, Basic Pitch's transcription of the recording, the rubric, the mistakes the agent kept, its feedback and the next assignment |
| `recordings/turn_NN.wav` | The 16 microphone recordings (copied from `runs/tutor_recordings/Chinmay/`) |
| `state/Chinmay.progress.json` | The learner's saved progress at the end (from `memory_store/`) |
| `state/Chinmay.json` | The learner's mistake log at the end (from `memory_store/`) |

**How it was made.** Full transcripts weren't saved yet during this session. They were rebuilt with
`python -m eval.rebuild_transcripts --learner Chinmay --since "2026-11-06 00:00" --out experiments/2026-11-06_chinmay_rondo`.
The script replays every model reply from the LLM response cache, in order, through the real (deterministic) tools.
All 16 rebuilt turns match the saved history: same decision, same next assignment, same score (`index.json`:
`rebuilt_ok: true`). From now on every turn is saved automatically to `runs/tutor_sessions/<learner>/<timestamp>.json`.

**Loading it**
```python
import json, pathlib
d = pathlib.Path("experiments/2026-11-06_chinmay_rondo")
index = json.loads((d / "index.json").read_text())
turn = json.loads((d / "turns/turn_06.json").read_text())   # turn["messages"], turn["trace"], turn["rubric"], ...
# later sessions: from tutor.tutor_agent import load_transcripts; load_transcripts("Chinmay")
```

## The attempts

"Candidates → kept" is the alignment's candidate mistakes, then how many the agent kept as real.

| # | Assignment | Notes played/target | Candidates → kept | Score | Decision | Next | Cost |
|---|---|---|---|---|---|---|---|
| 1 | whole excerpt, full | 50/46 | 80 → 8 | 55 | beginner_lesson | pickup + bars 1–2, beginner | $0.052 |
| 2 | pickup + bars 1–2, beginner | 14/8 | 9 → 0 | 100 | harder | pickup + bars 1–2, intermediate | $0.037 |
| 3 | pickup + bars 1–2, intermediate | 23/14 | 35 → 13 | 52 | repeat | pickup + bars 1–2, intermediate | $0.045 |
| 4 | pickup + bars 1–2, intermediate | 21/14 | 33 → 0 | 100 | harder | pickup + bars 1–2, full | $0.038 |
| 5 | pickup + bars 1–2, full | 19/14 | 29 → 0 | 100 | next_chunk | bars 3–4, beginner | $0.042 |
| 6 | bars 3–4, beginner | 7/7 | 4 → 4 | 100 | harder | bars 5–6, intermediate | $0.033 |
| 7 | bars 5–6, intermediate | 10/12 | 9 → 0 | 100 | harder | bars 5–6, full | $0.036 |
| 8 | bars 5–6, full | 11/12 | 5 → 0 | 100 | next_chunk | bars 7–8, beginner | $0.039 |
| 9 | bars 7–8, beginner | 8/7 | 8 → 2 | 68 | repeat | bars 7–8, beginner | $0.040 |
| 10 | bars 7–8, beginner | 12/7 | 10 → 1 | 85 | harder | bars 7–8, intermediate | $0.032 |
| 11 | bars 7–8, intermediate | 8/7 | 1 → 1 | 100 | harder | bars 7–8, full | $0.033 |
| 12 | bars 7–8, full | 13/7 | 9 → 3 | 80 | repeat | bars 7–8, full | $0.038 |
| 13 | bars 7–8, full | 11/7 | 8 → 2 | 80 | repeat | bars 7–8, full | $0.032 |
| 14 | bars 7–8, full | 15/7 | 14 → 2 | 78 | repeat | bars 7–8, full | $0.034 |
| 15 | bars 7–8, full | 9/7 | 4 → 0 | 100 | next_chunk | whole excerpt, full | $0.030 |
| 16 | whole excerpt, full | 55/46 | 89 → 12 | 35 | beginner_lesson | pickup + bars 1–2, beginner | $0.058 |

## Analysis

See the GitHub issue for this session for the findings and the follow-up work.
