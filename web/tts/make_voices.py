"""Render the reference voices Chatterbox clones (web/tts/voices/*.wav).

Chatterbox takes timbre *and* delivery from a ~10–15 s reference clip, so the
clip sets the character of the spoken words.  These are rendered with Kokoro
(82M, Apache-2.0, synthetic speakers — no real person is cloned) reading a
slow, meditative passage at a reduced speed, so the clone inherits a calm,
unhurried, confident delivery.

    PYTHONNOUSERSITE=1 uv run --isolated --no-project --python 3.11 \
        --with kokoro --with soundfile --with pip python web/tts/make_voices.py

(PYTHONNOUSERSITE keeps a user-site torchvision from breaking transformers;
pip is there because misaki downloads its spaCy model with it.)

Kokoro needs its own throwaway environment (its pins differ from .venv-tts);
the WAVs are committed, so this only has to run to add or redo a voice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT = Path(__file__).resolve().parent / "voices"
SR = 24000
SPEED = 0.82

TEXT = (
    "Breathe in, slowly. Let the silence settle around you. "
    "There is nothing to hold on to, and nothing to chase. "
    "Each word arrives, rests for a moment, and passes on, "
    "like light moving across still water. You are here. That is enough."
)

# name → (Kokoro voice, language code: a = American, b = British)
VOICES = {
    "michael": ("am_michael", "a"),
    "onyx": ("am_onyx", "a"),
    "george": ("bm_george", "b"),
    "emma": ("bf_emma", "b"),
    "heart": ("af_heart", "a"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipes: dict[str, KPipeline] = {}
    for name, (voice, lang) in VOICES.items():
        pipe = pipes.setdefault(lang, KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M"))
        parts = [np.asarray(audio, np.float32) for _, _, audio in pipe(TEXT, voice=voice, speed=SPEED)]
        gap = np.zeros(int(0.35 * SR), np.float32)
        wav = np.concatenate([x for p in parts for x in (p, gap)])
        wav *= 0.89 / max(1e-6, float(np.abs(wav).max()))
        sf.write(OUT / f"{name}.wav", wav, SR)
        print(f"{name:8s} {voice:11s} {len(wav) / SR:5.1f}s")


if __name__ == "__main__":
    main()
