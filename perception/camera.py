"""Camera compensation: wildlife cameras pan to follow the chase, so raw pixel speed is mostly camera speed.

Background dots move only because of the camera. Per frame pair we fit a similarity
transform (pan + zoom + roll, RANSAC) on background dots, chain them into a map from
every frame to the reference frame, and express all animal positions in that stable frame.
If too few background dots survive, fresh corners outside the animal boxes are matched
with LK for that frame pair (the approach used by YOLO+camera-motion football projects).
"""
from __future__ import annotations

import numpy as np


def _affine3(A: np.ndarray) -> np.ndarray:
    M = np.eye(3)
    M[:2] = A
    return M


def estimate_from_points(tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, frames: np.ndarray | None = None,
                         boxes: dict | None = None, min_pts: int = 6) -> np.ndarray:
    """Return (T,3,3) matrices mapping pixel coords in frame t -> reference frame 0."""
    import cv2
    T = tracks.shape[1]
    bg = owner < 0
    to_ref = np.tile(np.eye(3), (T, 1, 1))
    last = np.eye(3)
    for t in range(1, T):
        ok = bg & visible[:, t] & visible[:, t - 1]
        A = None
        if ok.sum() >= min_pts:
            A, inl = cv2.estimateAffinePartial2D(tracks[ok, t].astype(np.float32), tracks[ok, t - 1].astype(np.float32),
                                                 method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if A is None and frames is not None:
            A = _frame_pair(frames[t - 1], frames[t], boxes, t)
        step = _affine3(A) if A is not None else last  # constant-motion fallback
        last = step
        to_ref[t] = to_ref[t - 1] @ step
    return to_ref


def _frame_pair(prev_rgb, cur_rgb, boxes, t):
    import cv2
    g0 = cv2.cvtColor(prev_rgb[..., ::-1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur_rgb[..., ::-1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask = np.full(g0.shape, 255, np.uint8)
    for bx in (boxes or {}).values():
        if t - 1 in bx:
            x0, y0, x1, y1 = [int(v) for v in bx[t - 1]]
            mask[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = 0
    p0 = cv2.goodFeaturesToTrack(g0, 200, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 6:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 6:
        return None
    A, _ = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return A


def apply(to_ref: np.ndarray, t: int | np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    if np.ndim(t) == 0:
        M = to_ref[int(t)]
        h = np.c_[xy.reshape(-1, 2), np.ones(len(xy.reshape(-1, 2)))] @ M.T
        return h[:, :2].reshape(xy.shape)
    out = np.empty_like(xy)
    for k, tk in enumerate(np.asarray(t, int)):
        out[k] = (to_ref[tk] @ np.r_[xy[k], 1.0])[:2]
    return out


def scale(to_ref: np.ndarray) -> np.ndarray:
    """Per-frame zoom factor of frame t relative to the reference (for body-length conversion)."""
    return np.sqrt(np.abs(np.linalg.det(to_ref[:, :2, :2])))


def compensated_speed(tracks: np.ndarray, visible: np.ndarray, to_ref: np.ndarray, fps: float, idx: np.ndarray) -> np.ndarray:
    """Median speed (px/s in reference frame) of the selected dots, per dot."""
    out = []
    T = tracks.shape[1]
    for i in idx:
        v = visible[i]
        ts = np.nonzero(v)[0]
        if len(ts) < 3:
            continue
        p = apply(to_ref, ts, tracks[i, ts])
        d = np.linalg.norm(np.diff(p, axis=0), axis=1) / (np.diff(ts) / fps)
        out.append(np.median(d))
    return np.array(out)
