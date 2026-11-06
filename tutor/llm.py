"""The one place the project calls an LLM: OpenRouter's OpenAI-compatible chat API.

    from tutor import llm
    resp = llm.chat(messages, tools=..., tag="agent")   # an OpenAI-style response dict

What this wrapper adds:

* Budget. Every paid call is appended to the ledger (``llm_ledger.jsonl`` at the
  repo root) with its tokens and OpenRouter's dollar cost. Once the ledger total
  reaches ``TUTOR_BUDGET_USD`` the wrapper raises ``BudgetExceeded`` instead of
  calling. ``python -m tutor.llm`` prints a spending report.
* Disk cache. Identical requests (same model, settings, messages, tools) are
  answered from ``.llm_cache/`` for free, so re-running an unchanged eval costs
  nothing. Set ``TUTOR_CACHE=0`` to turn it off.
* Settings from ``.env``: ``OPENROUTER_API_KEY``, ``TUTOR_MODEL`` (an OpenRouter
  model id, see https://openrouter.ai/models), ``TUTOR_EFFORT``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("TUTOR_MODEL", "anthropic/claude-sonnet-5.5")
EFFORT = os.environ.get("TUTOR_EFFORT", "medium")
LEDGER = Path(os.environ.get("TUTOR_LEDGER", ROOT / "llm_ledger.jsonl"))
CACHE_DIR = Path(os.environ.get("TUTOR_CACHE_DIR", ROOT / ".llm_cache"))


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call once the spending cap is reached."""


_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = OpenAI(base_url=OPENROUTER_URL, api_key=key, max_retries=4, timeout=180,
                         default_headers={"X-Title": "CS329Z Music Tutor"})
    return _client


# ---- budget ---------------------------------------------------------------

def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def total_spend() -> float:
    return sum(r.get("cost_usd", 0.0) for r in read_ledger())


def budget() -> float | None:
    cap = os.environ.get("TUTOR_BUDGET_USD")
    return float(cap) if cap else None


def check_budget() -> None:
    cap = budget()
    if cap is None:
        return
    spent = total_spend()
    if spent >= cap:
        raise BudgetExceeded(f"Spent ${spent:.2f} of the ${cap:.2f} cap (TUTOR_BUDGET_USD). "
                             "Raise it in .env if you mean to continue.")


def _record(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---- the call -------------------------------------------------------------

def chat(messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None,
         model: str | None = None, max_tokens: int = 16000, tag: str = "", cache: bool | None = None) -> dict:
    """One chat completion. Returns the response as a plain dict, plus
    ``resp["cost_usd"]`` (0 for a cache hit) and ``resp["cached"]``."""
    model = model or MODEL
    req = {"model": model, "messages": messages, "max_tokens": max_tokens,
           "extra_body": {"usage": {"include": True}}}
    if EFFORT:
        req["extra_body"]["reasoning"] = {"effort": EFFORT}
    if tools:
        req["tools"] = tools
    if response_format:
        req["response_format"] = response_format

    use_cache = os.environ.get("TUTOR_CACHE", "1") != "0" if cache is None else cache
    key = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if use_cache and path.exists():
        resp = json.loads(path.read_text())
        resp.update(cost_usd=0.0, cached=True)
        return resp

    check_budget()
    t0 = time.time()
    resp = client().chat.completions.create(**req).model_dump()
    usage = resp.get("usage") or {}
    cost = float(usage.get("cost") or 0.0)
    finish = (resp.get("choices") or [{}])[0].get("finish_reason")
    _record({"ts": round(t0, 3), "model": resp.get("model", model), "tag": tag or os.environ.get("TUTOR_TAG", ""),
             "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
             "cost_usd": cost, "seconds": round(time.time() - t0, 2), "finish_reason": finish})
    if use_cache and finish in {"stop", "tool_calls"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resp))
    resp.update(cost_usd=cost, cached=False)
    return resp


def message_of(resp: dict) -> dict:
    return resp["choices"][0]["message"]


def text_of(resp: dict) -> str:
    return message_of(resp).get("content") or ""


# ---- report ---------------------------------------------------------------

def spending_report() -> str:
    rows = read_ledger()
    if not rows:
        return "No LLM calls recorded yet."
    agg = defaultdict(lambda: [0, 0, 0, 0.0])
    for r in rows:
        a = agg[(r.get("tag") or "(untagged)", r.get("model", "?"))]
        a[0] += 1; a[1] += r.get("input_tokens", 0); a[2] += r.get("output_tokens", 0); a[3] += r.get("cost_usd", 0.0)
    lines = [f"{'tag':<28}{'model':<36}{'calls':>6}{'in tok':>11}{'out tok':>10}{'USD':>9}"]
    for (tag, m), (n, i, o, usd) in sorted(agg.items(), key=lambda kv: -kv[1][3]):
        lines.append(f"{tag[:27]:<28}{m[:35]:<36}{n:>6}{i:>11}{o:>10}{usd:>9.3f}")
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    cap = budget()
    lines.append(f"\nTotal: {len(rows)} calls, ${total:.3f}" + (f" of ${cap:.2f} budget" if cap else " (no budget set)"))
    return "\n".join(lines)


if __name__ == "__main__":  # python -m tutor.llm
    print(spending_report())
