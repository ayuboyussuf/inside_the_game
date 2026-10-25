import os

import numpy as np
import pytest

os.environ.setdefault("HUNT_AUTOPSY_OFFLINE", "1")

from football.adapter import shot_episode, shot_episodes, to_events  # noqa: E402
from football.intel import MatchState, win_probability  # noqa: E402
from football.overlays import OverlayCue  # noqa: E402
from football.synth import generate_match, generate_shots  # noqa: E402
from narrate.personalize import Profile, wants  # noqa: E402
from perception.football_autoevent import events_from_tracks  # noqa: E402
from perception.keypoints import blocks_line, heading_from_person, in_vision_cone, screened  # noqa: E402


@pytest.fixture(scope="module")
def match():
    return generate_match(3)


def test_match_is_synthetic_and_plausible(match):
    m = match["metadata"]
    assert m["synthetic"] is True
    types = {e["type"] for e in match["events"]}
    assert {"pass", "shot", "tackle", "pressure", "possession_change", "kickoff", "full_time"} <= types
    passes = [e for e in match["events"] if e["type"] == "pass"]
    acc = np.mean([p["outcome"] == "complete" for p in passes])
    assert 0.65 < acc < 0.9
    ts = [e["timestamp"] for e in match["events"]]
    assert ts == sorted(ts)


def test_adapter_and_intel(match):
    evs = to_events(match)
    st = MatchState(match["metadata"])
    kinds = set()
    for e in evs:
        kinds |= {s["kind"] for s in st.apply(e)}
    snap = st.snapshot()
    assert sum(snap["score"].values()) == sum(match["metadata"]["final_score"].values())
    assert abs(sum(t["possession_pct"] for t in snap["teams"].values()) - 100) < 0.5
    assert "full_time" in kinds and "shot" in kinds


def test_win_probability_sums_to_one():
    wp = win_probability(1, 0, 80, 1.4, 1.2)
    assert abs(sum(wp.values()) - 1) < 1e-9 and wp["home"] > wp["away"]


def test_shot_episode_features():
    shots = generate_shots(30, seed=5)
    eps = shot_episodes(shots)
    assert len(eps) == 30
    ep = eps[0]
    assert ep.domain == "football" and ep.distance_unit == "m"
    assert ep.actor_by_role("attacker") and ep.actor_by_role("defender")
    from events.features import extract_features
    f = extract_features(ep)
    d = shots[0]["detail"]
    assert abs(f["reaction_delay"] - d["keeper_reaction_s"]) < 1e-3
    assert f["screening"] == d["screening_defenders"]


def test_personalization_differs(match):
    from agents.orchestrator import LiveMatch
    feed = {"metadata": match["metadata"], "events": match["events"][:900]}
    profs = {"a": Profile(persona="analyst"), "b": Profile(persona="casual", language="es"),
             "c": Profile(persona="player_focus", favorite_player=match["metadata"]["lineups"]["NBR"][9]["id"])}
    live = LiveMatch(feed, profs)
    lines = {k: [] for k in profs}
    while (p := live.step()) is not None:
        for k, o in p["outputs"].items():
            lines[k] += o["lines"]
            for c in o["cues"]:
                OverlayCue.model_validate(c)
    assert len(lines["a"]) > len(lines["b"])
    assert all(ln["lang"] == "es" for ln in lines["b"])
    assert not live.rejected  # fact-check never had to reject a template line


def test_wants_filters_casual():
    m = {"kind": "difficult_pass", "importance": 0.4, "team": "NBR", "player": "x"}
    assert wants(m, Profile(persona="analyst"))
    assert not wants(m, Profile(persona="casual"))


def test_autoevent_detects_pass_and_turnover():
    t = np.arange(0, 4, 0.1)
    ball = np.zeros((len(t), 2))
    ball[t < 1] = [10, 10]
    mv = (t >= 1) & (t < 1.5)
    ball[mv] = np.c_[10 + (t[mv] - 1) * 40, np.full(mv.sum(), 10)]
    ball[(t >= 1.5) & (t < 2.5)] = [30, 10]
    mv2 = (t >= 2.5) & (t < 3)
    ball[mv2] = np.c_[30 + (t[mv2] - 2.5) * 20, np.full(mv2.sum(), 10)]
    ball[t >= 3] = [40, 10]
    players = {"a": np.tile([10, 10], (len(t), 1)), "b": np.tile([30, 10], (len(t), 1)), "c": np.tile([40, 10], (len(t), 1))}
    ev = events_from_tracks(t, ball, players, {"a": "H", "b": "H", "c": "A"})
    assert ev[0]["type"] == "pass" and ev[0]["outcome"] == "complete" and ev[0]["recipient_id"] == "b"
    assert ev[0]["ball_speed_kmh"] > 100
    assert any(e["type"] == "possession_change" and e["player_id"] == "c" for e in ev)


def test_vision_geometry():
    assert blocks_line(np.array([0, 0]), np.array([10, 0]), [4, -1, 6, 1])
    assert not blocks_line(np.array([0, 0]), np.array([10, 0]), [4, 2, 6, 3])
    h = np.array([1.0, 0.0])
    assert in_vision_cone(np.zeros(2), h, np.array([[5, 1], [-5, 0]])).tolist() == [True, False]
    assert screened(np.zeros(2), np.array([10, 0]), [[4, -1, 6, 1], [4, 5, 6, 6]], heading=h) == 1
    kp = np.zeros((17, 2))
    kp[0], kp[5], kp[6] = [0, -1], [-1, 0], [1, 0]
    v = heading_from_person(kp)
    assert v is not None and v[1] < 0
