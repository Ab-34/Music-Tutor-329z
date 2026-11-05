"""Offline tests for the LLM wrapper's budget stop and the agent loop (no network)."""
import json

import pytest

from tutor import agent, llm


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(llm, "LEDGER", path)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    return path


def test_budget_stop_refuses_calls_once_cap_is_reached(ledger, monkeypatch):
    ledger.write_text(json.dumps({"cost_usd": 3.0}) + "\n" + json.dumps({"cost_usd": 2.5}) + "\n")
    monkeypatch.setenv("TUTOR_BUDGET_USD", "5")
    monkeypatch.setattr(llm, "client", lambda: pytest.fail("must not reach the network"))
    with pytest.raises(llm.BudgetExceeded):
        llm.chat([{"role": "user", "content": "hi"}], cache=False)


def test_no_cap_means_no_stop(ledger, monkeypatch):
    monkeypatch.delenv("TUTOR_BUDGET_USD", raising=False)
    llm.check_budget()  # does not raise


def _tool_call(name, args, i):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_loop_runs_tools_and_submits(monkeypatch):
    script = [
        [_tool_call("transcribe_performance", {"backend": "oracle"}, 0)],
        [_tool_call("find_error_candidates", {}, 1)],
        [_tool_call("submit_report", {"errors": [{"type": "missed", "time": 1.0, "pitch": 60}],
                                      "feedback": "ok"}, 2)],
    ]
    seen = []

    def fake_chat(messages, **kw):
        seen.append(messages)
        calls = script[len(seen) - 1]
        return {"choices": [{"message": {"content": "", "tool_calls": calls}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "cost_usd": 0.01, "cached": False}

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = agent.run_agent("examples/ref.mid", "examples/perf.mid", write_memory=False)
    assert [t["tool"] for t in out["trace"]] == ["transcribe_performance", "find_error_candidates", "submit_report"]
    assert all(t["ok"] for t in out["trace"])
    assert out["errors"][0]["pitch"] == 60
    assert out["usage"]["cost_usd"] == pytest.approx(0.03)
    assert seen[-1][-1]["role"] == "tool"  # tool results go back as role=tool messages


def test_agent_stops_at_run_budget(monkeypatch):
    monkeypatch.setattr(agent, "RUN_BUDGET_USD", 0.05)
    monkeypatch.setattr(llm, "chat", lambda m, **kw: {
        "choices": [{"message": {"content": "", "tool_calls": [_tool_call("get_score_text", {}, len(m))]}}],
        "usage": {}, "cost_usd": 0.03, "cached": False})
    out = agent.run_agent("examples/ref.mid", "examples/perf.mid", write_memory=False)
    assert "run budget" in out["error"]
    assert out["usage"]["cost_usd"] == pytest.approx(0.06)


def test_agent_on_event_streams_llm_and_tool_events(monkeypatch):
    script = [[_tool_call("transcribe_performance", {"backend": "oracle"}, 0)],
              [_tool_call("submit_report", {"errors": [], "feedback": "fine"}, 1)]]
    n = {"i": 0}

    def fake_chat(messages, **kw):
        n["i"] += 1
        return {"choices": [{"message": {"content": "thinking", "tool_calls": script[n["i"] - 1]}}],
                "usage": {}, "cost_usd": 0.01, "cached": False}

    monkeypatch.setattr(llm, "chat", fake_chat)
    events = []
    out = agent.run_agent("examples/ref.mid", "examples/perf.mid", write_memory=False, on_event=events.append)
    assert [e["kind"] for e in events] == ["llm", "tool", "llm", "tool"]
    assert events[1]["name"] == "transcribe_performance" and events[1]["ok"]
    assert out["feedback"] == "fine"
