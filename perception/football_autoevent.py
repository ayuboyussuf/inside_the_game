"""Auto-eventing: derive pass and speed metrics from a football video feed instead of hand tagging.

Pipeline (GPU box): YOLO(+ByteTrack) players & ball -> team assignment by jersey colour
(k-means on torso crops, the roboflow/sports TeamClassifier idea) -> image->pitch homography
from 4+ pitch keypoints (roboflow/sports pitch-keypoint model or manual clicks) -> metric tracks
-> ``events_from_tracks``. The last step is pure numpy and unit-tested on synthetic tracks;
its output is the same provider format football/adapter.py ingests.
"""
from __future__ import annotations

import numpy as np

POSSESSION_RADIUS_M = 1.5


def pitch_homography(img_pts: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray:
    import cv2
    H, _ = cv2.findHomography(np.asarray(img_pts, np.float32), np.asarray(pitch_pts, np.float32), cv2.RANSAC)
    return H


def to_pitch(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def team_colors(crops: list[np.ndarray], k: int = 2, seed: int = 0) -> np.ndarray:
    """k-means on mean torso colour -> team label per crop."""
    from sklearn.cluster import KMeans
    feats = np.array([c[c.shape[0] // 5: c.shape[0] // 2].reshape(-1, 3).mean(0) for c in crops])
    return KMeans(k, n_init=10, random_state=seed).fit_predict(feats)


def events_from_tracks(t: np.ndarray, ball: np.ndarray, players: dict[str, np.ndarray], team_of: dict[str, str],
                       min_hold: int = 3) -> list[dict]:
    """t (T,), ball (T,2) metres, players {id: (T,2) metres, NaN when unseen}. Returns provider-style events."""
    T = len(t)
    ids = list(players)
    P = np.stack([players[i] for i in ids])  # (n,T,2)
    d = np.linalg.norm(P - ball[None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(d, axis=0)
    owner = np.where(d[near, np.arange(T)] < POSSESSION_RADIUS_M, near, -1)
    # debounce: owner must persist min_hold frames
    stable = np.full(T, -1)
    run_id, run_len = -1, 0
    for k in range(T):
        if owner[k] == run_id:
            run_len += 1
        else:
            run_id, run_len = owner[k], 1
        if run_id >= 0 and run_len >= min_hold:
            stable[k - run_len + 1:k + 1] = run_id
    v = np.gradient(ball, t, axis=0)
    speed_kmh = np.linalg.norm(v, axis=1) * 3.6
    events = []
    last_owner, last_k = None, None
    for k in range(T):
        o = stable[k]
        if o < 0:
            continue
        if last_owner is not None and o != last_owner:
            a, b = ids[last_owner], ids[o]
            same = team_of[a] == team_of[b]
            events.append({"type": "pass", "timestamp": float(t[last_k]), "player_id": a, "team_id": team_of[a],
                           "recipient_id": b if same else None, "outcome": "complete" if same else "incomplete",
                           "location": ball[last_k].round(2).tolist(), "end_location": ball[k].round(2).tolist(),
                           "length_m": round(float(np.linalg.norm(ball[k] - ball[last_k])), 1),
                           "ball_speed_kmh": round(float(np.nanmax(speed_kmh[last_k:k + 1])), 1), "source": "auto-eventing"})
            if not same:
                events.append({"type": "possession_change", "timestamp": float(t[k]), "player_id": b, "team_id": team_of[b],
                               "location": ball[k].round(2).tolist(), "reason": "interception", "source": "auto-eventing"})
        last_owner, last_k = o, k
    return events
