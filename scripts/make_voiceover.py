"""Generate the film's voice-over with Kokoro (open-source neural TTS, Apache-2.0, runs offline).

    .venv-tts/Scripts/python scripts/make_voiceover.py [--voice bm_george]

Writes video/public/vo/<scene>.wav and video/src/vo.json (seconds per line), which the Remotion
composition uses to time each scene to its narration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video" / "public" / "vo"
SR = 24000

# Spoken form (numbers written as they should be read). On-screen text keeps the digits.
LINES = {
    "cold": "Every attack has an anatomy. Patterns you can measure, and explain.",
    "problem": "The scoreline tells you what happened. Fans want to know why.",
    "reveal": "Hunt Autopsy works backwards from the outcome. Live, fact-checked, and made for every fan.",
    "live": ("Synthetic match events stream in as they happen. We turn them into timed, machine-readable overlays, "
             "that a broadcaster can draw over the live feed. Pass difficulty, expected goals, shot speed, pressing, "
             "momentum, control versus chaos, and live win probability."),
    "fans": ("One stream, two experiences. The analyst gets every number, in English. A Northbridge fan gets short, "
             "emotional lines, in Spanish. Eight agents share one blackboard."),
    "fact": "A fact-check agent blocks any number that isn't in the evidence.",
    "recap": "Recaps are written for each viewer's language, and club.",
    "autopsy": ("Why was this shot saved? Keeper reaction: zero point two one seconds. Saves in six hundred and fifty six "
                "of seven hundred and twenty one similar shots. Every shot has a planted cause, and the explanation matches it."),
    "hood": ("The live path is deterministic. About one millisecond per event. The language model writes recaps and new "
             "languages, and every draft is fact-checked. It runs on Azure Container Apps, with Azure OpenAI, or GitHub Models."),
    "proof": ("Ninety one point nine percent of planted causes recovered. Zero invented numbers, in twelve hundred narrations. "
              "Eleven evals, including the one we still fail."),
    "close": "Hunt Autopsy. The why, behind every moment.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="bm_george")
    ap.add_argument("--speed", type=float, default=1.05)
    a = ap.parse_args()
    import soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="b" if a.voice.startswith("b") else "a")
    OUT.mkdir(parents=True, exist_ok=True)
    durs = {}
    for key, text in LINES.items():
        audio = [np.asarray(chunk.audio if hasattr(chunk, "audio") else chunk[2]) for chunk in pipe(text, voice=a.voice, speed=a.speed)]
        wav = np.concatenate(audio)
        sf.write(OUT / f"{key}.wav", wav, SR)
        durs[key] = round(len(wav) / SR, 2)
        print(f"{key:8s} {durs[key]:5.2f}s  {len(text.split())} words")
    (ROOT / "video" / "src" / "vo.json").write_text(json.dumps({"voice": a.voice, "speed": a.speed, "seconds": durs}, indent=1))
    print("total speech", round(sum(durs.values()), 1), "s")


if __name__ == "__main__":
    main()
