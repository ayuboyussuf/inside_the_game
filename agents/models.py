"""Trained autopsy models, built once and cached under data/cache/models/."""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

from events.schema import Episode

from explain.evidence import Autopsy
from football.adapter import shot_episodes
from football.synth import generate_shots
from sim.chase import RULES, generate

# Shots: the keeper's dive speed is only visible after the strike, so speed features are excluded.
FOOTBALL_FEATURES = ["reaction_delay", "screening", "approach_angle", "separation_at_commit", "direction_changes",
                     "group_help", "fatigue", "closing_speed"]

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "models"
EPISODES = ROOT / "data" / "episodes"   # real-clip episodes written by perception/run.py
VERSION = "v4"


def _cached(name: str, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}-{VERSION}.pkl"
    if p.exists():
        try:
            return pickle.loads(p.read_bytes())
        except Exception:
            pass
    obj = build()
    p.write_bytes(pickle.dumps(obj))
    return obj


@lru_cache(maxsize=1)
def shot_autopsy() -> Autopsy:
    def build():
        eps = shot_episodes(generate_shots(1200, seed=100))
        return Autopsy(kind="lgbm", seed=0, features=FOOTBALL_FEATURES).fit(eps)
    return _cached("football-shots", build)


@lru_cache(maxsize=1)
def hunt_library() -> dict:
    """Demo chase library: single-predator (compound rule), pack and herd scenarios, + any real clips."""
    def build():
        lib = {}
        for rule, n in (("compound", 360), ("pack", 160), ("screening", 160)):
            eps = generate(n, rule, seed=11)
            lib[rule] = {"episodes": eps, "autopsy": Autopsy(kind="lgbm", seed=0).fit(eps), "rule": RULES[rule].description}
        return lib
    lib = _cached("hunt-library", build)
    real = load_real_episodes()
    if real:
        lib["real"] = {"episodes": real, "autopsy": Autopsy(kind="logreg", seed=0).fit(real) if len(real) >= 8 else lib["compound"]["autopsy"],
                       "rule": "real clips (hand-labelled outcomes)"}
    return lib


def load_real_episodes() -> list[Episode]:
    if not EPISODES.exists():
        return []
    out = []
    for p in sorted(EPISODES.glob("*.json")):
        try:
            out.append(Episode.model_validate(json.loads(p.read_text())))
        except Exception:
            continue
    return out
