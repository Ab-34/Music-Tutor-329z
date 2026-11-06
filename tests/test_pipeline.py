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


def test_single_line_simplifications():
    from tutor.notes import Note
    from tutor.tools.simplify import drop_ornaments, thin_to_grid
    run = [Note(71, 0.0, 0.24), Note(69, 0.25, 0.49), Note(68, 0.5, 0.74), Note(69, 0.75, 0.99),
           Note(67, 1.47, 1.49), Note(72, 1.5, 1.99)]  # four sixteenths, a grace note, an eighth
    assert [n.pitch for n in drop_ornaments(run)] == [71, 69, 68, 69, 72]
    thin = thin_to_grid(drop_ornaments(run), grid=0.5)
    assert [(n.pitch, n.onset) for n in thin] == [(71, 0.0), (68, 0.5), (72, 1.5)]
    assert thin[0].offset == 0.5 and thin[1].offset == 1.0  # held for the whole eighth
