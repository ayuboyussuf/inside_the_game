"""Turn an Episode into ~12 named, domain-neutral features.

Every feature has a hunt meaning and a football twin, so a model validated on chases
transfers to shots. Features only use data before the outcome (no leakage): the
window ends 0.3 s before the first contact/abort/goal/save event.

Adapters may supply values they can compute better (e.g. football screening from a
freeze frame) via ``episode.meta["feature_overrides"]``; overrides win.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import Episode

OUTCOME_EVENTS = ("contact", "abort", "goal", "save", "miss", "block")
SCREEN_CORRIDOR = 1.0  # distance units either side of the attacker-defender line
GROUP_RADIUS = 5.0


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    unit: str
    hunt: str
    football: str
    decimals: int = 2


FEATURES: list[FeatureSpec] = [
    FeatureSpec("closing_speed", "Closing speed", "BL/s", "predator speed toward prey", "shooter's run-up speed toward goal", 1),
    FeatureSpec("approach_angle", "Approach angle", "deg", "angle between predator and prey headings", "shot angle relative to the keeper", 0),
    FeatureSpec("reaction_delay", "Reaction delay", "s", "predator acceleration -> prey's first flee move", "strike -> keeper's first dive movement", 2),
    FeatureSpec("separation_at_commit", "Separation at commit", "BL", "distance when the predator commits", "shooter-to-keeper distance at the strike", 1),
    FeatureSpec("direction_changes", "Direction changes", "count", "number of prey cuts", "shooter feints before striking", 0),
    FeatureSpec("turn_sharpness", "Turn sharpness", "deg", "mean angle of prey cuts", "sharpness of the feint", 0),
    FeatureSpec("screening", "Screening", "count", "herd members between predator and target", "defenders blocking the keeper's view", 0),
    FeatureSpec("group_help", "Group help", "count", "pack hunters near the prey", "teammates making decoy runs", 0),
    FeatureSpec("fatigue", "Chase duration", "s", "sprint duration before the final lunge", "time since the shooter's last sprint", 1),
    FeatureSpec("attacker_top_speed", "Attacker top speed", "BL/s", "predator top speed", "shooter top speed", 1),
    FeatureSpec("defender_top_speed", "Defender top speed", "BL/s", "prey top speed", "keeper dive speed", 1),
    FeatureSpec("speed_ratio", "Speed ratio", "x", "predator / prey top speed", "shooter / keeper speed", 2),
]
FEATURE_NAMES = [f.name for f in FEATURES]
FOOTBALL_LABELS = {
    "reaction_delay": "Keeper reaction time", "screening": "Defenders screening the keeper",
    "approach_angle": "Shot angle away from the keeper", "separation_at_commit": "Shooter-to-keeper distance",
    "direction_changes": "Feints before the shot", "group_help": "Teammates near the keeper",
    "fatigue": "Time since last sprint", "closing_speed": "Run-up speed",
}
SPEC = {f.name: f for f in FEATURES}


def label_for(name: str, domain: str) -> str:
    return FOOTBALL_LABELS.get(name, SPEC[name].label) if domain == "football" else SPEC[name].label


def unit_for(name: str, distance_unit: str) -> str:
    u = SPEC[name].unit
    if distance_unit == "m":
        u = u.replace("BL", "m")
    return u


# ---------------------------------------------------------------------------
def _resample(track_arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of (t,x,y) track onto times t -> (len(t), 2)."""
    return np.column_stack([np.interp(t, track_arr[:, 0], track_arr[:, 1]),
                            np.interp(t, track_arr[:, 0], track_arr[:, 2])])


def velocity(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros_like(pos)
    return np.gradient(pos, t, axis=0)


def speed(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity(pos, t), axis=1)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return np.nan
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def key_times(ep: Episode) -> dict[str, float]:
    """t_commit, t_trigger, t_end (end of the pre-outcome window)."""
    att = ep.actor_by_role("attacker")
    arr = ep.track(att.id).array() if att and ep.track(att.id) else None
    t0 = float(arr[0, 0]) if arr is not None else 0.0
    t_last = float(arr[-1, 0]) if arr is not None else ep.duration
    sprint = ep.first_event("sprint_start", att.id if att else None)
    if sprint is not None:
        t_commit = sprint.t
    elif arr is not None and len(arr) > 3:
        s = speed(arr[:, 1:], arr[:, 0])
        t_commit = float(arr[np.argmax(s > 0.6 * s.max()), 0])
    else:
        t_commit = t0
    strike = ep.first_event("strike")
    t_trigger = strike.t if strike is not None else t_commit
    outs = [e.t for e in ep.events if e.type in OUTCOME_EVENTS]
    t_end = (min(outs) - 0.3) if outs else t_last
    t_end = float(np.clip(t_end, t_trigger + 1e-3, max(t_last, t_trigger + 1e-3)))
    return {"t0": t0, "t_commit": float(t_commit), "t_trigger": float(t_trigger), "t_end": t_end}


def extract_features(ep: Episode) -> dict[str, float]:
    f = {n: np.nan for n in FEATURE_NAMES}
    att, dfd = ep.actor_by_role("attacker"), ep.actor_by_role("defender")
    kt = key_times(ep)
    tc, tt, te = kt["t_commit"], kt["t_trigger"], kt["t_end"]

    ta = ep.track(att.id) if att else None
    td = ep.track(dfd.id) if dfd else None
    if ta is not None and td is not None and len(ta.t) >= 3:
        A = ta.array()
        grid = A[:, 0]
        grid = grid[(grid >= kt["t0"]) & (grid <= te + 1e-9)]
        if len(grid) >= 3:
            pa = _resample(A, grid)
            pd_ = _resample(td.array(), grid)
            va, vd = velocity(pa, grid), velocity(pd_, grid)
            dist = np.linalg.norm(pa - pd_, axis=1)
            win = grid >= tc - 1e-9
            if win.sum() >= 2:
                f["closing_speed"] = float(np.mean(-np.gradient(dist, grid)[win]))
                f["attacker_top_speed"] = float(np.max(np.linalg.norm(va[win], axis=1)))
                f["defender_top_speed"] = float(np.max(np.linalg.norm(vd[win], axis=1)))
            f["separation_at_commit"] = float(np.interp(tt, grid, dist))
            # approach angle: headings once the defender is moving (first 0.5 s after reaction)
            flee = ep.first_event("flee_start", dfd.id)
            ta_ = (flee.t + 0.3) if flee is not None else te
            i = int(np.clip(np.searchsorted(grid, min(ta_, te)), 0, len(grid) - 1))
            f["approach_angle"] = _angle_between(va[i], vd[i])
            # screening: defender_support inside the corridor between attacker and defender
            screens = []
            for s in ep.actors_by_role("defender_support"):
                tr = ep.track(s.id)
                if tr is None:
                    continue
                ps = _resample(tr.array(), grid)
                seg = pd_ - pa
                L = np.linalg.norm(seg, axis=1) + 1e-9
                proj = np.einsum("ij,ij->i", ps - pa, seg) / L
                rel = ps - pa
                perp = np.abs(seg[:, 0] * rel[:, 1] - seg[:, 1] * rel[:, 0]) / L
                screens.append(((proj > 0) & (proj < L) & (perp < SCREEN_CORRIDOR)).astype(float))
            f["screening"] = float(np.round(np.mean(np.sum(screens, axis=0)[win]))) if screens and win.any() else 0.0
            # group help at the trigger
            k = int(np.clip(np.searchsorted(grid, tt), 0, len(grid) - 1))
            n = 0
            for s in ep.actors_by_role("attacker_support"):
                tr = ep.track(s.id)
                if tr is not None and np.linalg.norm(_resample(tr.array(), grid[k:k + 1])[0] - pd_[k]) < GROUP_RADIUS:
                    n += 1
            f["group_help"] = float(n)

    flee = ep.first_event("flee_start", dfd.id if dfd else None)
    f["reaction_delay"] = float((flee.t if flee is not None else te) - tt)
    turns = [e for e in ep.events_of("turn", dfd.id if dfd else None) if tt - 1e-9 <= e.t <= te]
    f["direction_changes"] = float(len(turns))
    f["turn_sharpness"] = float(np.mean([abs(e.attrs.get("angle", 0.0)) for e in turns])) if turns else 0.0
    f["fatigue"] = float(te - tc)
    if np.isfinite(f["attacker_top_speed"]) and np.isfinite(f["defender_top_speed"]):
        f["speed_ratio"] = float(f["attacker_top_speed"] / max(f["defender_top_speed"], 1e-3))

    for k, v in ep.meta.get("feature_overrides", {}).items():
        if k in f:
            f[k] = float(v)
    return f


def feature_table(episodes: list[Episode]) -> pd.DataFrame:
    rows = []
    for ep in episodes:
        r = extract_features(ep)
        r["episode_id"] = ep.episode_id
        r["outcome"] = ep.outcome
        rows.append(r)
    return pd.DataFrame(rows).set_index("episode_id")


def mirror_episode(ep: Episode) -> Episode:
    """Horizontal flip (x -> -x). Used by the flip-invariance eval."""
    e2 = ep.model_copy(deep=True)
    for tr in e2.tracks:
        tr.x = [-x for x in tr.x]
    for ev in e2.events:
        if ev.x is not None:
            ev.x = -ev.x
        if "angle" in ev.attrs:
            ev.attrs = {**ev.attrs, "angle": -ev.attrs["angle"]}
    return e2
