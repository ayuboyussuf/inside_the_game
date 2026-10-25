"""Viewer profiles: the same intelligence, genuinely different experiences.

analyst      - every moment above a low bar, numbers, autopsy factors, flow metrics
casual       - only big moments, short emotional lines, few numbers
player_focus - everything about one player (tags, milestones, involvement) + their card
Favourite club flips the framing ("your team") and boosts that club's moments.
"""
from __future__ import annotations

from pydantic import BaseModel


class Profile(BaseModel):
    persona: str = "casual"          # analyst | casual | player_focus | kids
    language: str = "en"
    favorite_club: str | None = None
    favorite_player: str | None = None
    focus_metric: str | None = None  # e.g. "pass_accuracy_pct", "xg", "distance_km"
    min_importance: float | None = None

    @property
    def voice(self) -> str:
        return "analyst" if self.persona in ("analyst", "player_focus") else "casual"

    def threshold(self) -> float:
        if self.min_importance is not None:
            return self.min_importance
        return {"analyst": 0.35, "casual": 0.45, "player_focus": 0.3, "kids": 0.6}.get(self.persona, 0.45)


NEVER_CASUAL = {"difficult_pass", "pass_streak", "distance_tag", "pressure_shift"}
PLAYER_KINDS = {"speed_tag", "distance_tag", "milestone", "pass_streak", "difficult_pass", "shot", "goal", "card",
                "substitution", "big_chance_missed", "save_autopsy", "goal_autopsy"}


def score(m: dict, p: Profile) -> float:
    s = float(m.get("importance", 0.0))
    if p.favorite_club and m.get("team") == p.favorite_club:
        s += 0.12
    if p.favorite_player and (m.get("player") == p.favorite_player or m.get("related_player") == p.favorite_player):
        s += 0.4
    return s


def wants(m: dict, p: Profile) -> bool:
    k = m["kind"]
    if k in ("half_time", "full_time", "goal"):
        return True
    if p.persona == "player_focus":
        involved = m.get("player") == p.favorite_player or m.get("related_player") == p.favorite_player
        return involved and k in PLAYER_KINDS or (k in ("momentum_swing",) and m.get("importance", 0) >= 0.5)
    if p.persona in ("casual", "kids") and k in NEVER_CASUAL and m.get("player") != p.favorite_player:
        return False
    if k == "distance_tag" and m.get("player") != p.favorite_player:
        return False
    return score(m, p) >= p.threshold()


def framing(m: dict, p: Profile) -> str | None:
    """'for' / 'against' the viewer's club, used by the UI for colour and tone."""
    if not p.favorite_club or not m.get("team"):
        return None
    return "for" if m["team"] == p.favorite_club else "against"
