"""Offline tests (no API key, no audio deps): perturb -> detect -> grade."""
import json
import random

from data.make_perturbations import perturb
from eval.grade_detection import match
from examples.make_example import piece
from tutor.tools.align import find_errors
from tutor.tools.simplify import STEPS, difficulty


def test_clean_performance_has_no_errors():
    ref = piece()
    assert find_errors(ref, ref) == []


def test_algorithmic_detector_recovers_most_perturbations():
    ref = piece()
    rng = random.Random(0)
    tp = total = 0
    for _ in range(50):
        perf, gt = perturb(ref, 3, rng)
        pred = find_errors(ref, perf)
        tp += len(match(gt, pred))
        total += len(gt)
    assert tp / total > 0.8, tp / total


def test_ladder_gets_easier():
    ref = piece()
    d0 = difficulty(ref)["mean_polyphony"]
    d1 = difficulty(STEPS["drop_inner_voices"](ref))["mean_polyphony"]
    d2 = difficulty(STEPS["melody_only"](ref))["mean_polyphony"]
    assert d0 >= d1 >= d2 == 1.0
