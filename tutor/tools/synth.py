"""Notes -> audio. Uses FluidSynth + a soundfont when available, else sine waves."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pretty_midi

from tutor.notes import Note

SR = 22050

# General MIDI program numbers for instruments we may want to render.
PROGRAMS = {"piano": 0, "guitar": 24, "violin": 40, "flute": 73, "sax": 65}


def synthesize(notes: list[Note], instrument: str = "piano", sr: int = SR) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=PROGRAMS.get(instrument, 0))
    for n in notes:
        inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.onset, max(n.offset, n.onset + 0.03)))
    pm.instruments.append(inst)
    sf2 = os.environ.get("SOUNDFONT")
    if sf2 and Path(sf2).exists():
        try:
            return pm.fluidsynth(fs=sr, sf2_path=sf2)
        except Exception:  # pyfluidsynth missing or broken -> fall back
            pass
    return pm.synthesize(fs=sr)


def write_wav(audio: np.ndarray, path: str | Path, sr: int = SR) -> None:
    import soundfile as sf  # audio extra

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(str(path), audio / peak * 0.9, sr)
