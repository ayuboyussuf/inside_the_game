"""The one event schema shared by hunts, simulated chases and football.

The explainer and narrator only ever read these objects, never raw video.
Roles are domain-neutral:
  attacker  - predator / shooter (the side trying to make the outcome happen)
  defender  - prey / goalkeeper  (the side trying to stop it)
  attacker_support - pack mates / teammates making decoy runs
  defender_support - herd members / outfield defenders (can screen)
Outcome is always from the attacker's point of view: 1 = caught / scored, 0 = escaped / saved.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

Domain = Literal["hunt", "sim", "football"]
Role = Literal["attacker", "defender", "attacker_support", "defender_support", "other"]

# Event vocabulary. Hunt/sim and football share the generic attack events;
# football match streams add the on-ball vocabulary.
ATTACK_EVENTS = {
    "track_start",   # first frame an actor is tracked
    "stalk",         # attacker moving slowly toward defender
    "sprint_start",  # attacker commits (predator sprint / shooter run-up)
    "flee_start",    # defender's first reaction (prey flees / keeper starts dive)
    "turn",          # sharp heading change (prey cut / shooter feint)
    "strike",        # football: ball struck
    "contact",       # attacker reaches defender (catch)
    "abort",         # attacker gives up (escape)
    "goal", "save", "miss", "block",
}
MATCH_EVENTS = {
    "kickoff", "pass", "carry", "dribble", "shot", "tackle", "interception",
    "pressure", "possession_change", "clearance", "foul", "card", "substitution",
    "corner", "free_kick", "offside", "half_time", "full_time", "goal", "save",
}
EVENT_TYPES = ATTACK_EVENTS | MATCH_EVENTS


class Actor(BaseModel):
    id: str
    role: Role
    name: str | None = None
    team: str | None = None
    species: str | None = None


class Event(BaseModel):
    t: float = Field(..., description="seconds from episode/match start")
    type: str
    actor: str | None = None
    target: str | None = None
    x: float | None = None
    y: float | None = None
    attrs: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:  # noqa: D401
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}")


class Track(BaseModel):
    """Positions of one actor over time, in the episode's distance unit (world frame, camera removed)."""
    actor: str
    t: list[float]
    x: list[float]
    y: list[float]
    visible: list[bool] | None = None

    def array(self) -> np.ndarray:
        return np.column_stack([self.t, self.x, self.y]).astype(float)


class Episode(BaseModel):
    """One attack: a chase or a shot, plus its known outcome."""
    episode_id: str
    domain: Domain
    outcome: int = Field(..., ge=0, le=1)
    outcome_label: str  # "caught" / "escaped" / "goal" / "saved"
    actors: list[Actor]
    events: list[Event]
    tracks: list[Track] = Field(default_factory=list)
    fps: float = 30.0
    distance_unit: str = "body_lengths"  # football: "m"
    time_unit: str = "s"
    meta: dict = Field(default_factory=dict)  # free-form: source, licence, planted rule, ground truth...

    # ---- helpers -------------------------------------------------------
    def actor_by_role(self, role: Role) -> Actor | None:
        return next((a for a in self.actors if a.role == role), None)

    def actors_by_role(self, role: Role) -> list[Actor]:
        return [a for a in self.actors if a.role == role]

    def track(self, actor_id: str) -> Track | None:
        return next((tr for tr in self.tracks if tr.actor == actor_id), None)

    def first_event(self, type_: str, actor: str | None = None) -> Event | None:
        for e in sorted(self.events, key=lambda e: e.t):
            if e.type == type_ and (actor is None or e.actor == actor):
                return e
        return None

    def events_of(self, type_: str, actor: str | None = None) -> list[Event]:
        return [e for e in self.events if e.type == type_ and (actor is None or e.actor == actor)]

    @property
    def outcome_names(self) -> tuple[str, str]:
        """(name for 0, name for 1)."""
        return ("saved", "goal") if self.domain == "football" else ("escaped", "caught")

    @property
    def duration(self) -> float:
        ts = [e.t for e in self.events] + [tr.t[-1] for tr in self.tracks if tr.t]
        return max(ts) if ts else 0.0
