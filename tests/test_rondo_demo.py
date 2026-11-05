"""The Rondo alla Turca demo learners: true notes, obvious mistakes, levels, weak spots."""
import json

import pytest

from eval.grade_detection import match
from examples import rondo_demo as R
from tutor.tools.align import find_errors


def test_excerpt_is_the_famous_opening():
    rh, lh = R.load_excerpt()
    # B4 A4 G#4 A4 | C5 . D5 C5 B4 C5 | E5 . F5 E5 D#5 E5
    assert [n.pitch for n in rh[:15]] == [71, 69, 68, 69, 72, 74, 72, 71, 72, 76, 77, 76, 75, 76, 83]
    assert lh[0].pitch == 57  # A3 under the first C5
    assert R.bar_beat(rh[0].onset) == (0, 1)   # pickup
    assert R.bar_beat(rh[4].onset) == (1, 1)   # first C5 = bar 1, beat 1
    assert R.bar_beat(rh[5].onset) == (1, 2)   # D5 = bar 1, beat 2
    assert max(n.offset for n in rh + lh) == pytest.approx((1 + 8 * 2) * R.SEC_PER_BEAT, abs=0.01)


def test_perfect_take_has_no_mistakes():
    ref, perf, errors = R.take("perfect", 1)
    assert errors == [] and find_errors(ref, perf) == []


@pytest.mark.parametrize("level", list(R.PROFILES))
def test_mistakes_are_obvious(level):
    rh, _ = R.load_excerpt()
    melody = {(n.pitch, round(n.onset, 3)) for n in rh}
    for k in range(1, R.PROFILES[level].takes + 1):
        _, _, errors = R.take(level, k)
        for e in errors:
            assert e["description"]
            if e["type"] == "wrong_pitch":
                assert abs(e["played_pitch"] - e["pitch"]) >= 2
            elif e["type"] == "timing":
                assert e["offset_ms"] >= 200
            elif e["type"] == "missed":
                assert (e["pitch"], e["time"]) in melody
            else:
                assert e["pitch"] % 12 in R.OUT_OF_KEY


@pytest.mark.parametrize("level", list(R.PROFILES))
def test_b0_recovers_every_mistake(level):
    for k in range(1, R.PROFILES[level].takes + 1):
        ref, perf, errors = R.take(level, k)
        pred = find_errors(ref, perf)
        assert len(match(errors, pred)) == len(errors) == len(pred), (level, k)


def test_levels_get_worse_and_weak_spots_repeat():
    def mean_errors(level):
        p = R.PROFILES[level]
        return sum(len(R.take(level, k)[2]) for k in range(1, p.takes + 1)) / p.takes

    means = [mean_errors(lv) for lv in ("perfect", "nearly_there", "intermediate", "struggling")]
    assert means == sorted(means) and means[-1] >= 10
    for level, p in R.PROFILES.items():
        weak = [{(e["type"], e["time"], e["pitch"]) for e in R.take(level, k)[2] if e.get("weak_spot")}
                for k in range(1, p.takes + 1)]
        assert all(w == weak[0] for w in weak) and len(weak[0]) == p.n_weak


def test_write_all_matches_generated_takes(tmp_path):
    manifest = R.write_all(tmp_path)
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    assert len(rows) == sum(p.takes for p in R.PROFILES.values())
    r = next(r for r in rows if r["case_id"] == "rondo_struggling_take2")
    assert r["errors"] == R.take("struggling", 2)[2] and r["learner"] == "rondo_struggling"
