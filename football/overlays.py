"""Machine-readable, timed overlay cues for a rendering partner.

A cue says WHAT to draw (template + payload), WHEN (match time + duration), WHERE in the
layer stack (layer + priority) and for WHOM (profile). Renderers can consume the JSON
directly (see ``OverlayCue.model_json_schema()``, served at /api/overlay-schema).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Template = Literal["scorebug", "lower_third", "player_tag", "stat_card", "momentum_bar", "flow_meter",
                   "autopsy_card", "win_probability", "recap_card"]
Layer = Literal["persistent", "info", "alert"]

DURATION = {"lower_third": 6.0, "player_tag": 4.0, "stat_card": 8.0, "autopsy_card": 10.0, "momentum_bar": 5.0,
            "flow_meter": 6.0, "win_probability": 6.0, "recap_card": 15.0, "scorebug": 0.0}


class OverlayCue(BaseModel):
    cue_id: str
    match_id: str
    t_match: float = Field(..., description="match clock in seconds when the cue should appear")
    clock: str
    template: Template
    layer: Layer = "info"
    priority: int = Field(50, ge=0, le=100)
    duration_s: float = 5.0
    text: str | None = None
    payload: dict = Field(default_factory=dict)
    profile: str = "default"
    language: str = "en"
    framing: str | None = None  # "for" / "against" the viewer's club


TEMPLATE_FOR = {
    "goal": ("lower_third", "alert"), "goal_autopsy": ("autopsy_card", "alert"), "save_autopsy": ("autopsy_card", "alert"),
    "shot": ("lower_third", "info"), "big_chance_missed": ("lower_third", "info"), "speed_tag": ("player_tag", "info"),
    "distance_tag": ("player_tag", "info"), "milestone": ("stat_card", "info"), "pass_streak": ("player_tag", "info"),
    "difficult_pass": ("player_tag", "info"), "momentum_swing": ("momentum_bar", "info"), "flow_shift": ("flow_meter", "info"),
    "pressure_shift": ("flow_meter", "info"), "card": ("lower_third", "alert"), "substitution": ("lower_third", "info"),
    "half_time": ("recap_card", "alert"), "full_time": ("recap_card", "alert"),
}


def cue_for(m: dict, text: str | None, match_id: str, profile: str, lang: str, framing: str | None) -> OverlayCue:
    tpl, layer = TEMPLATE_FOR.get(m["kind"], ("lower_third", "info"))
    payload = {"kind": m["kind"], "team": m.get("team"), "team_name": m.get("team_name"), "player": m.get("player"),
               "player_name": m.get("player_name"), "evidence": m.get("evidence", {})}
    if m.get("autopsy"):
        a = m["autopsy"]
        payload["factors"] = [{"label": f["label"], "display": f["display"], "contribution": f["contribution"]} for f in a["top_factors"]]
        payload["decisive"] = a["decisive_moments"][:1]
        payload["p_outcome"] = a["model"]["p_outcome"]
    return OverlayCue(cue_id=f"{match_id}-{m['id']}-{profile}", match_id=match_id, t_match=round(m["t"], 2), clock=m.get("clock", ""),
                      template=tpl, layer=layer, priority=int(min(100, 100 * m.get("importance", 0.5))),
                      duration_s=DURATION.get(tpl, 5.0), text=text, payload=payload, profile=profile, language=lang, framing=framing)


def scorebug(snapshot: dict, meta: dict) -> dict:
    h, a = meta["home"], meta["away"]
    return {"template": "scorebug", "layer": "persistent", "clock": snapshot["clock"],
            "home": {"id": h["id"], "short": h["short"], "color": h["colors"][0], "score": snapshot["score"][h["id"]]},
            "away": {"id": a["id"], "short": a["short"], "color": a["colors"][0], "score": snapshot["score"][a["id"]]},
            "win_probability": snapshot["win_probability"], "momentum": snapshot["momentum"]}
