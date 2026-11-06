"""Offline tests for the GUI helpers and smoke tests for every page (no network)."""
import json

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from examples.rondo_demo import bar_beat_str  # noqa: E402
from gui import components as C  # noqa: E402
from gui import jobs, sources  # noqa: E402
from tutor.notes import Note  # noqa: E402

GT = [{"type": "missed", "time": 1.0, "pitch": 60}, {"type": "extra", "time": 2.0, "pitch": 64}]


def test_score_marks_hits_false_positives_and_misses():
    pred = [{"type": "missed", "time": 1.05, "pitch": 60}, {"type": "timing", "time": 3.0, "pitch": 62}]
    sc = C.score(GT, pred)
    assert sc["pred_status"] == ["hit", "false_positive"]
    assert sc["gt_status"] == ["found", "missed"]
    assert sc["metrics"]["tp"] == 1 and sc["metrics"]["fp"] == 1 and sc["metrics"]["fn"] == 1


def test_score_without_ground_truth():
    sc = C.score(None, [{"type": "missed", "time": 1.0, "pitch": 60}])
    assert sc["metrics"] is None and sc["pred_status"] == [None]


def test_piano_roll_has_one_trace_per_layer():
    ref = [Note(60, 0, 0.5), Note(64, 0.5, 1.0)]
    assert len(C.piano_roll(ref).data) == 1
    fig = C.piano_roll(ref, ref, pred=[GT[0]], gt=GT, pred_status=["hit"], gt_status=["found", "missed"])
    assert [t.name for t in fig.data] == ["reference", "performance", "known mistake", "predicted error"]


def test_wav_bytes_is_a_wav():
    assert C.wav_bytes([Note(60, 0, 0.3)])[:4] == b"RIFF"


def test_hidden_systems_not_offered():
    assert "audio_llm" not in C.visible_systems() and "agent" in C.visible_systems()


def _manifest(tmp_path, n=3):
    rows = [{"case_id": f"c{i}", "source": "x", "start": 0, "severity": i, "ref": "examples/ref.mid",
             "perf_midi": "examples/perf.mid", "errors": GT[:i]} for i in range(n)]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def test_eval_cases_resolve_paths_against_repo_root(tmp_path):
    path, _ = _manifest(tmp_path)
    cases = sources.eval_cases(path)
    assert [c.case_id for c in cases] == ["c0", "c1", "c2"]
    assert cases[1].ref == str(sources.ROOT / "examples/ref.mid") and cases[1].gt == GT[:1]


def test_stopped_run_is_graded_from_the_cases_it_finished(tmp_path):
    manifest, rows = _manifest(tmp_path)
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({"pid": 999999, "system": "algorithmic", "manifest": str(manifest),
                                              "n_total": 3, "started": 0, "spend_at_start": 0}))
    (run / "predictions.jsonl").write_text(json.dumps({"case_id": "c1", "errors": GT[:1]}) + "\n")
    st = jobs.status(run)
    assert st["state"] == "stopped" and st["done"] == 1 and st["total"] == 3
    data = jobs.load_run(run)
    assert data["scores"]["partial"] and data["scores"]["overall"]["f1"] == 1.0
    assert jobs.list_runs(tmp_path / "runs") == [run]
    assert jobs.active_job(tmp_path / "runs") is None


def test_estimate_cost_averages_past_cases(tmp_path):
    run = tmp_path / "agent_midi"
    run.mkdir()
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps({"case_id": f"c{i}", "usage": {"cost_usd": c}}) + "\n" for i, c in enumerate([0.02, 0.04])))
    assert jobs.estimate_cost("agent", 10, tmp_path) == pytest.approx(0.3)
    assert jobs.estimate_cost("text_llm", 10, tmp_path) is None
    assert jobs.estimate_cost("algorithmic", 10, tmp_path) == 0.0


# ---- page smoke tests -------------------------------------------------------

@pytest.mark.parametrize("page", ["gui/app.py", "gui/pages/1_Assess.py", "gui/pages/2_Simplify.py",
                                  "gui/pages/3_Memory.py", "gui/pages/4_Eval.py", "gui/pages/5_Spending.py"])
def test_page_loads(page):
    at = AppTest.from_file(str(sources.ROOT / page), default_timeout=30).run()
    assert not at.exception, at.exception


def test_assess_b0_finds_the_bundled_mistakes():
    at = AppTest.from_file(str(sources.ROOT / "gui/pages/1_Assess.py"), default_timeout=30).run()
    at.radio(key="src_kind").set_value("Bundled example").run()
    at.selectbox(key="system").select("algorithmic").run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    res = at.session_state["assess_result"]
    assert C.score(res["src"]["gt"], res["out"]["errors"])["metrics"]["recall"] == 1.0


def test_assess_defaults_to_rondo_and_scores_a_struggling_take():
    at = AppTest.from_file(str(sources.ROOT / "gui/pages/1_Assess.py"), default_timeout=30).run()
    assert at.radio(key="src_kind").value == "Rondo alla Turca"
    at.selectbox(key="rondo_level").select("struggling").run()
    assert at.text_input(key="learner").value == "rondo_struggling"  # one memory per learner level
    at.selectbox(key="system").select("algorithmic").run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    res = at.session_state["assess_result"]
    assert res["src"]["piece"] == "rondo" and len(res["src"]["gt"]) >= 10
    assert C.score(res["src"]["gt"], res["out"]["errors"])["metrics"]["f1"] == 1.0
    rows = C.error_rows(res["out"]["errors"], [None] * len(res["out"]["errors"]), bar_beat_str)
    assert all("bar.beat" in r for r in rows)


def test_rondo_takes_listed_for_every_level():
    takes = sources.rondo_takes()
    assert {t.level for t in takes} == {"perfect", "nearly_there", "intermediate", "struggling"}
    assert "Rondo alla Turca learners" in sources.eval_sets()
