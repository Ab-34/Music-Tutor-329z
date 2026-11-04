"""Thin wrapper around the Anthropic client so every caller uses the same settings."""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("TUTOR_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("TUTOR_EFFORT", "medium")

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def create(**kwargs):
    """messages.create with project defaults.

    Uses the server-side refusal fallback (`fallbacks="default"`) so a rare
    safety-classifier decline is retried on another model instead of failing
    the eval case. Remove the `betas`/`fallbacks` lines to turn it off.
    """
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("max_tokens", 16000)
    kwargs.setdefault("output_config", {}).setdefault("effort", EFFORT)
    return client().beta.messages.create(
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **kwargs,
    )


def text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")
