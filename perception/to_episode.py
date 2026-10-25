"""Perception outputs -> Episode (the only thing the explainer reads).

Per actor: position = median of its visible torso dots (robust to grass hiding legs),
falling back to the box centre; mapped into the camera-compensated reference frame;
expressed in predator body lengths (BL) and seconds.
"""
from __future__ import annotations

import numpy as np

from events.detect import detect_events, smooth
from events.schema import Actor, Episode, Event, Track

from .camera import apply, scale


def actor_paths(tracks_json: dict, tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, to_ref: np.ndarray,
                min_frames: int = 10) -> dict[int, dict]:
    T = tracks.shape[1]
    by_id: dict[int, dict] = {}
    for fr in tracks_json["frames"]:
        f = fr["f"]
        if f >= T:
            break
        for d in fr["dets"]:
            by_id.setdefault(d["id"], {"cls": d["cls"], "f": [], "box": []})
            by_id[d["id"]]["f"].append(f)
            by_id[d["id"]]["box"].append(d["xyxy"])
    out = {}
    for tid, info in by_id.items():
        if len(info["f"]) < min_frames:
            continue
        fs = np.array(info["f"])
        boxes = np.array(info["box"], float)
        centres = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        mine = owner == tid
        for k, f in enumerate(fs):
            v = mine & visible[:, f]
            if v.sum() >= 5:
                pts = tracks[v, f]
                x0, y0, x1, y1 = boxes[k]
                inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                if inside.sum() >= 5:
                    centres[k] = np.median(pts[inside], axis=0)
        ref = apply(to_ref, fs, centres)
        size = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / scale(to_ref)[fs]
        out[tid] = {"cls": info["cls"], "f": fs, "xy": ref, "size_ref": size}
    return out


def approach_score(paths: dict[int, dict], i: int, j: int, fps: float) -> float:
    """How much i's fastest movement is directed TOWARD j (attackers close in; prey moves away)."""
    a, b = paths[i], paths[j]
    common = np.intersect1d(a["f"], b["f"])
    if len(common) < 5:
        return 0.0
    ia, ib = np.searchsorted(a["f"], common), np.searchsorted(b["f"], common)
    pa, pb = smooth(a["xy"][ia], fps), smooth(b["xy"][ib], fps)
    v = np.gradient(pa, common / fps, axis=0)
    to_b = pb - pa
    to_b /= np.linalg.norm(to_b, axis=1, keepdims=True) + 1e-9
    s = np.linalg.norm(v, axis=1)
    top = s >= np.quantile(s, 0.75)
    return float(np.mean(np.einsum("ij,ij->i", v[top], to_b[top])))


def assign_roles(paths: dict[int, dict], predator: int | None, prey: int | None, fps: float,
                 pair: list[int] | None = None) -> dict[int, str]:
    ids = sorted(paths, key=lambda i: -len(paths[i]["f"]))
    if predator is None or prey is None:
        cand = [i for i in (pair or []) if i in paths] or ids[:2]
        if len(cand) < 2:
            raise ValueError("need at least two tracked animals")
        a, b = cand[:2]
        # the attacker is the one whose fastest movement closes the gap
        predator, prey = (a, b) if approach_score(paths, a, b, fps) >= approach_score(paths, b, a, fps) else (b, a)
    roles = {predator: "attacker", prey: "defender"}
    pcls, qcls = paths[predator]["cls"], paths[prey]["cls"]
    for i in ids:
        if i not in roles:
            c = paths[i]["cls"]
            roles[i] = "other" if pcls == qcls else ("attacker_support" if c == pcls else "defender_support" if c == qcls else "other")
    return roles


def build_episode(clip_id: str, tracks_json: dict, tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray,
                  to_ref: np.ndarray, outcome: str, fps: float, predator: int | None = None, prey: int | None = None,
                  meta: dict | None = None) -> Episode:
    paths = actor_paths(tracks_json, tracks, visible, owner, to_ref)
    roles = assign_roles(paths, predator, prey, fps, pair=tracks_json.get("pair"))
    pred = next(i for i, r in roles.items() if r == "attacker")
    prey_id = next(i for i, r in roles.items() if r == "defender")
    bl_px = float(np.median(paths[pred]["size_ref"])) * 0.9  # box long side ~ body length
    actors, trs = [], []
    for tid, role in roles.items():
        p = paths[tid]
        aid = {"attacker": "pred", "defender": "prey"}.get(role) or (f"{role.split('_')[0][:4]}{tid}" if role.endswith("support") else f"oth{tid}")
        p["aid"] = aid
        actors.append(Actor(id=aid, role=role, species=p["cls"]))
        xy = smooth(p["xy"] / bl_px, fps)
        trs.append(Track(actor=aid, t=(p["f"] / fps).round(4).tolist(), x=xy[:, 0].round(4).tolist(), y=xy[:, 1].round(4).tolist()))
    # events on the common time base of predator and prey
    a, b = paths[pred], paths[prey_id]
    common = np.intersect1d(a["f"], b["f"])
    ia, ib = np.searchsorted(a["f"], common), np.searchsorted(b["f"], common)
    evs = detect_events(common / fps, a["xy"][ia] / bl_px, b["xy"][ib] / bl_px, "pred", "prey", fps=fps)
    caught = outcome.startswith("caught") or outcome in ("1", "goal")
    has_out = any(e.type in ("contact", "abort") for e in evs)
    if not has_out or (caught and not any(e.type == "contact" for e in evs)) or (not caught and any(e.type == "contact" for e in evs)):
        evs = [e for e in evs if e.type not in ("contact", "abort")]
        evs.append(Event(t=float(common[-1] / fps), type="contact" if caught else "abort", actor="pred", target="prey",
                         attrs={"source": "manifest outcome"}))
    return Episode(episode_id=clip_id, domain="hunt", outcome=int(caught), outcome_label="caught" if caught else "escaped",
                   actors=actors, events=sorted(evs, key=lambda e: e.t), tracks=trs, fps=fps,
                   meta={"body_length_px": round(bl_px, 2), "track_ids": {p["aid"]: int(t) for t, p in paths.items() if "aid" in p},
                         **(meta or {})})
