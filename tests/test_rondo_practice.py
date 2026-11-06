"""The one-note-at-a-time practice sheet: both versions are single lines with whole bars."""
from examples import rondo_practice as P


def _bar_units(v):
    return [sum(it.get("units", it.get("rest", 0)) for it in b["items"]) for b in P.bars(v["notes"], 60 / v["bpm"])]


def test_both_versions_are_one_note_at_a_time_with_whole_bars():
    for v in P.versions():
        onsets = [round(n.onset, 3) for n in v["notes"]]
        assert len(onsets) == len(set(onsets)), v["id"]   # never two keys at once
        assert _bar_units(v) == [4] + [8] * 8, v["id"]     # pickup + eight full 2/4 bars


def test_full_melody_is_the_score_and_simplified_is_easier():
    full, simple = P.versions()
    assert [n.pitch for n in full["notes"][:9]] == [71, 69, 68, 69, 72, 74, 72, 71, 72]
    assert sum(1 for n in full["notes"] if n.offset - n.onset < 0.1) == 6     # grace notes kept
    assert all(n.offset - n.onset >= 0.1 for n in simple["notes"])           # ...and dropped
    assert len(simple["notes"]) < len(full["notes"])
    d_full = P.difficulty(full["notes"])["notes_per_sec"]
    d_simple = P.difficulty(simple["notes"])["notes_per_sec"]
    assert d_simple < d_full / 2


def test_page_has_both_versions_and_the_keyboard():
    page = P.render()
    assert 'id="full"' in page and 'id="simple"' in page and 'class="kb"' in page
    assert page.count('class="n ') == sum(1 for v in P.versions() for n in v["notes"] if n.offset - n.onset >= 0.1)
