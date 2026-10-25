"""Keeper reaction time from real penalty / shot video, using TAPNet point tracks.

    python -m perception.keeper_reaction --clip data/raw/football/keller_saves_henry_pk.ogv [--points tapnet|lk]

This is the football twin of the hunt feature 'reaction_delay' (strike -> keeper's first dive movement):
  1. ball: tiled detection + Kalman (perception/football_video.py) and camera motion from background corners
     -> the strike is the onset of the camera-compensated ball speed
  2. keeper: the player closest to where the ball goes 0.4 s after the strike, excluding the kicker
  3. TAPNet (BootsTAPIR) dots on the keeper's body + background dots, in a window around the strike
  4. keeper speed (camera-compensated, in keeper body-heights/s) -> first sustained movement = dive onset
reaction_delay = onset - strike (negative = the keeper moved before the ball was struck, i.e. guessed early).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_clip(path: Path, max_side: int = 960):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        frames.append(img)
    cap.release()
    return frames, fps


def camera_steps(frames):
    """Per-frame similarity transform frame t -> frame t-1 from background corners (LK); cumulative to frame 0."""
    import cv2
    to_ref = [np.eye(3)]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for img in frames[1:]:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 10)
        A = None
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
            ok = st.ravel() == 1
            if ok.sum() >= 8:
                A, _ = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        M = np.eye(3)
        if A is not None:
            M[:2] = A
        to_ref.append(to_ref[-1] @ M)
        prev = g
    return np.array(to_ref)


def apply(M, xy):
    return (M @ np.r_[xy, 1.0])[:2]


def find_strike(frames, fps, to_ref, models):
    from perception.football_video import BallTracker, ball_candidates, pitch_mask
    tr = BallTracker(fps, max_gap_s=0.3)
    pos = np.full((len(frames), 2), np.nan)
    for k, img in enumerate(frames):
        xy, st = tr.step(ball_candidates(models, img, mask=pitch_mask(img)))
        if xy is not None and st == "detected":
            pos[k] = apply(to_ref[k], xy)  # into frame-0 coordinates: camera motion removed
    ok = ~np.isnan(pos[:, 0])
    if ok.sum() < 6:
        raise ValueError("ball not tracked well enough to find the strike")
    idx = np.arange(len(frames))
    p = np.c_[np.interp(idx, idx[ok], pos[ok, 0]), np.interp(idx, idx[ok], pos[ok, 1])]
    sp = np.linalg.norm(np.gradient(p, axis=0), axis=1) * fps
    # strike: first frame where the (previously slow) ball suddenly moves fast and keeps moving
    base = np.median(sp[: max(5, len(sp) // 4)])
    hi = max(base * 4, np.percentile(sp, 90) * 0.5)
    for k in range(3, len(sp) - 3):
        if sp[k] > hi and sp[k + 1] > hi * 0.6 and sp[max(0, k - 3):k].mean() < hi * 0.4:
            return k, p, sp
    return int(np.argmax(sp)), p, sp


def pick_keeper(frames, k_strike, ball_ref, to_ref, fps, models):
    from perception.football_video import COCO_PERSON
    r = models.players.predict(frames[k_strike], classes=[COCO_PERSON], conf=0.2, imgsz=1280, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    if len(boxes) < 2:
        raise ValueError("need at least kicker and keeper in view at the strike")
    inv = np.linalg.inv(to_ref[k_strike])
    ball_now = apply(inv, ball_ref[k_strike])
    ball_later = apply(inv, ball_ref[min(len(ball_ref) - 1, k_strike + int(0.4 * fps))])
    centres = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    kicker = int(np.argmin(np.linalg.norm(centres - ball_now, axis=1)))
    d = np.linalg.norm(centres - ball_later, axis=1)
    d[kicker] = np.inf
    return boxes[int(np.argmin(d))], boxes[kicker]


def keeper_motion(frames, fps, k0, k1, keeper_box, points: str = "tapnet"):
    from perception.camera import apply as cam_apply
    from perception.camera import estimate_from_points
    from perception.points import seed_queries, track_points
    win = np.ascontiguousarray(np.array(frames[k0:k1])[..., ::-1])  # RGB, contiguous for torch
    # the keeper is near-stationary before the strike, so the strike-time box is a good seed region at the window start
    kb = {1: {0: list(keeper_box)}}
    q, owner = seed_queries(win, kb, n_per_box=40, n_bg=40)
    tracks, vis = track_points(win, q, points)
    to_ref = estimate_from_points(tracks, vis, owner, frames=win)
    body = np.nonzero(owner == 1)[0]
    h = keeper_box[3] - keeper_box[1]
    pos = []
    for t in range(len(win)):
        v = body[vis[body, t]]
        pos.append(np.median(cam_apply(to_ref, t, tracks[v, t]), axis=0) if len(v) >= 5 else [np.nan, np.nan])
    pos = np.array(pos, float)
    ok = ~np.isnan(pos[:, 0])
    idx = np.arange(len(pos))
    pos = np.c_[np.interp(idx, idx[ok], pos[ok, 0]), np.interp(idx, idx[ok], pos[ok, 1])]
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * fps / max(h, 1.0)  # body heights per second
    alive = float(vis[body, -1].mean())
    return speed, alive


def reaction(clip: Path, points: str = "tapnet", pre_s: float = 0.6, post_s: float = 0.9) -> dict:
    from events.detect import onset, smooth
    from perception.football_video import FootballModels
    frames, fps = read_clip(clip)
    to_ref = camera_steps(frames)
    models = FootballModels.load()
    k_strike, ball_ref, ball_speed = find_strike(frames, fps, to_ref, models)
    keeper_box, kicker_box = pick_keeper(frames, k_strike, ball_ref, to_ref, fps, models)
    k0, k1 = max(0, k_strike - int(pre_s * fps)), min(len(frames), k_strike + int(post_s * fps))
    speed, alive = keeper_motion(frames, fps, k0, k1, keeper_box, points)
    t = np.arange(len(speed)) / fps
    s = np.linalg.norm(smooth(np.c_[speed, np.zeros_like(speed)], fps, 0.15), axis=1)
    i = onset(s, t, start=0, rise=0.15, sustain_s=0.1, min_peak=0.3)
    t_strike = k_strike / fps
    out = {"clip": clip.name, "points": points, "fps": fps, "strike_t": round(t_strike, 3),
           "keeper_box": [round(float(v), 1) for v in keeper_box], "kicker_box": [round(float(v), 1) for v in kicker_box],
           "dive_onset_t": None if i is None else round((k0 + i) / fps, 3),
           "reaction_delay_s": None if i is None else round((k0 + i) / fps - t_strike, 3),
           "keeper_peak_speed_bh_s": round(float(s.max()), 2), "keeper_dots_alive_at_end": round(alive, 2),
           "window": [round(k0 / fps, 2), round(k1 / fps, 2)], "speed_trace": [round(float(x), 3) for x in s]}
    od = ROOT / "data" / "evals" / "keeper_reaction"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{clip.stem}_{points}.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--points", default="tapnet", choices=["tapnet", "lk", "cotracker3"])
    a = ap.parse_args()
    r = reaction(Path(a.clip), a.points)
    print(json.dumps({k: v for k, v in r.items() if k != "speed_trace"}, indent=1))


if __name__ == "__main__":
    main()
