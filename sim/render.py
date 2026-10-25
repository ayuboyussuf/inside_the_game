"""Render a simulated chase to video with a panning, zooming camera.

Gives the perception pipeline a clip with KNOWN world positions and KNOWN camera motion,
so camera compensation (eval 6), event timing (eval 7) and track health (eval 8) can be
measured on CPU without any downloaded footage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from events.schema import Episode

# deliberately saturated, unnatural colours so the test blob detector is unambiguous (BGR)
COLORS = {"attacker": (0, 140, 255), "defender": (255, 0, 200), "attacker_support": (0, 255, 255), "defender_support": (255, 200, 0)}


def _texture(rng, h, w):
    import cv2
    base = rng.integers(70, 140, (h // 8, w // 8, 3)).astype(np.uint8)
    tex = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(rng.integers(-30, 30, (h, w, 3)).astype(np.float32), (0, 0), 1.5)
    tex = np.clip(tex.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    tex[..., 1] = np.clip(tex[..., 1].astype(int) + 40, 0, 255)  # savannah-ish green/yellow
    for _ in range(int(h * w / 2500)):  # grey rocks / tufts = trackable background corners
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        g = int(rng.integers(20, 230))
        cv2.circle(tex, c, int(rng.integers(3, 7)), (g, g, g), -1)
    return tex


def render(ep: Episode, out: Path, px_per_bl: float = 34.0, size=(960, 540), seed: int = 0, zoom_drift: float = 0.15) -> dict:
    """Write an mp4 and a ground-truth json (camera offsets, world positions, boxes)."""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = size
    arrs = {tr.actor: tr.array() for tr in ep.tracks}
    roles = {a.id: a.role for a in ep.actors}
    t = arrs[ep.actor_by_role("attacker").id][:, 0]
    allxy = np.concatenate([a[:, 1:] for a in arrs.values()])
    lo, hi = allxy.min(0) - 25, allxy.max(0) + 25
    world_w, world_h = int((hi[0] - lo[0]) * px_per_bl), int((hi[1] - lo[1]) * px_per_bl)
    world_w, world_h = max(world_w, W * 2), max(world_h, H * 2)
    tex = _texture(rng, world_h, world_w)
    pred, prey = ep.actor_by_role("attacker").id, ep.actor_by_role("defender").id
    centre = (arrs[pred][:, 1:] + arrs[prey][:, 1:]) / 2
    # camera follows the action with lag + shake; zoom slowly drifts
    cam = np.zeros_like(centre)
    cam[0] = centre[0]
    for i in range(1, len(centre)):
        cam[i] = cam[i - 1] + 0.25 * (centre[i] - cam[i - 1])
    cam += rng.normal(0, 0.05, cam.shape)
    zoom = 0.85 + zoom_drift * np.sin(np.linspace(0, np.pi, len(t)))
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), ep.fps, (W, H))
    gt = {"fps": ep.fps, "px_per_bl": px_per_bl, "frames": []}
    for i, ti in enumerate(t):
        s = px_per_bl * zoom[i]
        # world (BL) -> texture px
        cx, cy = (cam[i] - lo) * px_per_bl
        half_w, half_h = W / 2 / zoom[i], H / 2 / zoom[i]
        x0, y0 = int(np.clip(cx - half_w, 0, world_w - 2 * half_w - 1)), int(np.clip(cy - half_h, 0, world_h - 2 * half_h - 1))
        crop = tex[y0:y0 + int(2 * half_h), x0:x0 + int(2 * half_w)]
        frame = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
        boxes = []
        for aid, a in arrs.items():
            j = min(i, len(a) - 1)
            wx, wy = a[j, 1], a[j, 2]
            px = ((wx - lo[0]) * px_per_bl - x0) * zoom[i]
            py = ((wy - lo[1]) * px_per_bl - y0) * zoom[i]
            v = a[min(j + 1, len(a) - 1), 1:] - a[max(j - 1, 0), 1:]
            ang = float(np.degrees(np.arctan2(v[1], v[0]))) if np.linalg.norm(v) > 1e-6 else 0.0
            L = s * (1.0 if roles[aid].startswith("attacker") else 0.8)
            col = COLORS.get(roles[aid], (200, 200, 200))
            cv2.ellipse(frame, (int(px), int(py)), (int(L / 2), int(L / 4.5)), ang, 0, 360, col, -1)
            cv2.ellipse(frame, (int(px), int(py)), (int(L / 2), int(L / 4.5)), ang, 0, 360, (20, 20, 20), 1)
            for k in range(4):  # spots: texture for the point tracker
                off = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * (k - 1.5) * L / 5
                cv2.circle(frame, (int(px + off[0]), int(py + off[1])), max(1, int(L / 20)), (25, 25, 25), -1)
            boxes.append({"actor": aid, "role": roles[aid], "xyxy": [px - L / 2, py - L / 2, px + L / 2, py + L / 2], "world": [wx, wy]})
        vw.write(frame)
        gt["frames"].append({"t": float(ti), "x0": x0, "y0": y0, "zoom": float(zoom[i]), "boxes": boxes})
    vw.release()
    gt["lo"] = lo.tolist()
    (out.with_suffix(".gt.json")).write_text(json.dumps(gt))
    return gt
