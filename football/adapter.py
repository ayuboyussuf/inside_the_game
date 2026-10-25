"""Map a football event feed into the shared event schema.

* ``to_events``    - match stream -> list[events.schema.Event] (live layer)
* ``shot_episode`` - one on-target shot (with freeze frame + tracking snippet) -> Episode
                     (shooter = attacker, keeper = defender), so the SAME explainer that
                     reads hunts explains why the keeper couldn't reach it.

The provider field names are configurable via FIELD_MAP so the hackathon's own synthetic
format can be plugged in without touching anything downstream.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from events.schema import EVENT_TYPES, Actor, Episode, Event, Track

FIELD_MAP = {
    "id": "event_id", "t": "timestamp", "type": "type", "team": "team_id", "player": "player_id",
    "player_name": "player_name", "location": "location", "end_location": "end_location",
}
TYPE_MAP = {  # provider type -> schema type
    "ball_recovery": "interception", "duel": "tackle", "shot_saved": "save", "throw_in": "pass",
}


def load_feed(src) -> dict:
    if isinstance(src, (str, Path)):
        return json.loads(Path(src).read_text())
    return src


def to_event(raw: dict, fmap: dict = FIELD_MAP) -> Event | None:
    typ = TYPE_MAP.get(raw.get(fmap["type"]), raw.get(fmap["type"]))
    if typ not in EVENT_TYPES:
        return None
    loc = raw.get(fmap["location"]) or [None, None]
    skip = set(fmap.values())
    return Event(t=float(raw.get(fmap["t"], 0.0)), type=typ, actor=raw.get(fmap["player"]),
                 x=loc[0], y=loc[1], attrs={"team": raw.get(fmap["team"]), "id": raw.get(fmap["id"]),
                                            "player_name": raw.get(fmap["player_name"]),
                                            "end_location": raw.get(fmap["end_location"]),
                                            **{k: v for k, v in raw.items() if k not in skip}})


def to_events(feed: dict, fmap: dict = FIELD_MAP) -> list[Event]:
    return [e for e in (to_event(r, fmap) for r in feed["events"]) if e is not None]


# ---------------------------------------------------------------------------
def shot_episode(shot: dict) -> Episode | None:
    d = shot.get("detail")
    if not d or "tracking" not in d or shot.get("outcome") not in ("goal", "saved"):
        return None
    tr = d["tracking"]
    ts = [float(t) for t in tr["t"]]
    sh, kp = np.array(tr["shooter"]), np.array(tr["keeper"])
    shooter_id, keeper_id = shot.get("player_id") or "shooter", d.get("keeper_id") or "keeper"
    actors = [Actor(id=shooter_id, role="attacker", name=shot.get("player_name"), team=shot.get("team_id")),
              Actor(id=keeper_id, role="defender", name=d.get("keeper_name"))]
    tracks = [Track(actor=shooter_id, t=ts, x=sh[:, 0].tolist(), y=sh[:, 1].tolist()),
              Track(actor=keeper_id, t=ts, x=kp[:, 0].tolist(), y=kp[:, 1].tolist())]
    keeper0 = np.array(d["keeper_location"])
    near_keeper = 0
    for k, ff in enumerate(d.get("freeze_frame", [])):
        role = "defender_support" if ff["role"] == "defender" else "attacker_support"
        aid = f"{'def' if role == 'defender_support' else 'mate'}{k}"
        actors.append(Actor(id=aid, role=role))
        loc = ff["location"]
        tracks.append(Track(actor=aid, t=[ts[0], ts[-1]], x=[loc[0], loc[0]], y=[loc[1], loc[1]]))
        if role == "attacker_support" and np.linalg.norm(np.array(loc) - keeper0) < 5.0:
            near_keeper += 1
    t_strike = 1.5
    t_react = t_strike + float(d["keeper_reaction_s"])
    t_arrive = t_strike + float(d["ball_flight_s"])
    events = [Event(t=0.0, type="track_start", actor=shooter_id), Event(t=0.0, type="track_start", actor=keeper_id),
              Event(t=0.0, type="sprint_start", actor=shooter_id),
              Event(t=t_strike, type="strike", actor=shooter_id, attrs={"shot_speed_kmh": shot.get("shot_speed_kmh"), "xg": shot.get("xg")}),
              Event(t=round(t_react, 3), type="flee_start", actor=keeper_id, attrs={"what": "keeper dive"}),
              Event(t=round(t_arrive, 3), type="goal" if shot["outcome"] == "goal" else "save", actor=shooter_id if shot["outcome"] == "goal" else keeper_id)]
    outcome = int(shot["outcome"] == "goal")
    overrides = {
        "approach_angle": d["shot_angle_deg"], "screening": d["screening_defenders"],
        "separation_at_commit": d["keeper_distance_m"], "direction_changes": d.get("feints", 0),
        "fatigue": d.get("time_since_last_sprint_s", np.nan), "group_help": near_keeper,
    }
    return Episode(episode_id=shot.get("event_id", "shot"), domain="football", outcome=outcome,
                   outcome_label="goal" if outcome else "saved", actors=actors, events=events, tracks=tracks,
                   fps=10.0, distance_unit="m",
                   meta={"feature_overrides": overrides, "true_causes": d.get("outcome_model", {}).get("true_causes"),
                         "shot": {k: shot.get(k) for k in ("event_id", "timestamp", "team_id", "player_name", "xg", "shot_speed_kmh", "location")},
                         "keeper_name": d.get("keeper_name"), "source": "football/synth.py"})


def shot_episodes(feed_or_shots) -> list[Episode]:
    shots = feed_or_shots["events"] if isinstance(feed_or_shots, dict) else feed_or_shots
    out = [shot_episode(s) for s in shots if s.get("type") == "shot"]
    return [e for e in out if e is not None]
