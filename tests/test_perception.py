import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from perception.camera import apply, estimate_from_points  # noqa: E402


def test_camera_compensation_removes_pan_and_zoom():
    rng = np.random.default_rng(0)
    T, N = 40, 40
    world = rng.uniform(50, 450, (N, 2))
    tracks = np.zeros((N, T, 2))
    for t in range(T):
        s = 1 + 0.004 * t
        shift = np.array([3.0 * t, -1.5 * t])
        tracks[:, t] = (world - 250) * s + 250 - shift + rng.normal(0, 0.05, (N, 2))
    vis = np.ones((N, T), bool)
    owner = -np.ones(N, int)
    to_ref = estimate_from_points(tracks, vis, owner)
    back = np.stack([apply(to_ref, t, tracks[:, t]) for t in range(T)], axis=1)
    drift = np.abs(back - back[:, :1]).max()
    assert drift < 1.0


def test_lk_tracks_a_moving_square():
    from perception.points import track_lk
    T = 12
    frames = np.zeros((T, 120, 160, 3), np.uint8)
    rng = np.random.default_rng(1)
    tex = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
    for t in range(T):
        frames[t, 40:60, 20 + 3 * t:40 + 3 * t] = tex
    q = np.array([[0, 30.0, 50.0]])
    tr, vis = track_lk(frames, q)
    assert vis[0, -1]
    assert abs(tr[0, -1, 0] - (30 + 3 * (T - 1))) < 1.5
