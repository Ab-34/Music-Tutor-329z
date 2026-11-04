"""B1: one-shot audio-LLM baseline (promised in the proposal).

Claude does not take audio input, so this baseline needs an audio-capable model
from another provider (e.g. a Gemini or GPT-4o-audio model). Send the reference
(synthesized ref.wav) and the learner's perf.wav plus the same instructions as
text_llm.PROMPT, and parse the same JSON schema (text_llm.SCHEMA) so the grader
can score it unchanged.

Left unimplemented on purpose: pick the provider as a team, add its SDK to
pyproject.toml, and record the model/version in the report.
"""
from __future__ import annotations


def run(ref_path: str, perf_path: str, **_) -> dict:
    raise NotImplementedError("audio-LLM baseline: choose a provider and implement (see module docstring)")
