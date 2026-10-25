"""Dense point tracks ("the dots") seeded inside each body box plus on the background.

Backends
  tapnet      Google DeepMind TAPNet (BootsTAPIR / TAPIR PyTorch). Set TAPNET_CHECKPOINT to the .pt file.
  cotracker3  Meta CoTracker3 via torch.hub (fallback, easiest GPU setup).
  lk          OpenCV pyramidal Lucas-Kanade with forward-backward check (CPU, no downloads).

Output (points.npz): tracks (N,T,2) pixel xy, visible (N,T) bool, owner (N,) track id or -1 for background,
queries (N,3) [t, x, y].
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

N_PER_BOX, N_BG = 50, 30


def read_frames(video: Path, max_side: int | None = None) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if max_side and max(img.shape[:2]) > max_side:
            s = max_side / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        out.append(img[..., ::-1])  # RGB
    cap.release()
    return np.stack(out)


def seed_queries(frames: np.ndarray, boxes: dict[int, dict[int, list[float]]], n_per_box: int = N_PER_BOX,
                 n_bg: int = N_BG, seed: int = 0, masks: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """~50 points inside each box (inner ellipse, or SAM 2 mask if given) at its first frame + ~30 background corners."""
    import cv2
    rng = np.random.default_rng(seed)
    T, H, W = frames.shape[:3]
    q, owner = [], []
    for tid, bx in boxes.items():
        f0 = min(bx)
        x0, y0, x1, y1 = bx[f0]
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max(1.0, (x1 - x0) * 0.32), max(1.0, (y1 - y0) * 0.32)
        m = masks.get(tid) if masks else None
        k = 0
        while k < n_per_box:
            a, r = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 1))
            x, y = cx + r * rx * np.cos(a), cy + r * ry * np.sin(a)
            if m is not None and not m[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]:
                continue
            q.append([f0, x, y])
            owner.append(tid)
            k += 1
    gray = cv2.cvtColor(frames[0][..., ::-1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask = np.full(gray.shape, 255, np.uint8)
    for bx in boxes.values():
        if 0 in bx:
            x0, y0, x1, y1 = [int(v) for v in bx[0]]
            pad = int(0.3 * max(x1 - x0, y1 - y0)) + 4
            mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=n_bg * 3, qualityLevel=0.01, minDistance=max(8, W // 40), mask=mask)
    if corners is not None:
        c = corners.reshape(-1, 2)
        c = c[rng.permutation(len(c))[:n_bg]]
        for x, y in c:
            q.append([0, float(x), float(y)])
            owner.append(-1)
    return np.array(q, float), np.array(owner, int)


# ----------------------------------------------------------------------- LK
def track_lk(frames: np.ndarray, queries: np.ndarray, fb_thresh: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    T = len(frames)
    N = len(queries)
    tracks = np.full((N, T, 2), np.nan)
    vis = np.zeros((N, T), bool)
    gray = [cv2.cvtColor(f[..., ::-1].astype(np.uint8), cv2.COLOR_BGR2GRAY) for f in frames]
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    for i in range(N):
        t0 = int(queries[i, 0])
        tracks[i, t0] = queries[i, 1:]
        vis[i, t0] = True
    for t in range(1, T):
        alive = np.nonzero(vis[:, t - 1])[0]
        if len(alive):
            p0 = tracks[alive, t - 1].astype(np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], p0, None, **lk)
            pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - 1], p1, None, **lk)
            fb = np.linalg.norm(p0 - pb, axis=2).ravel()
            ok = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < fb_thresh)
            H, W = gray[t].shape
            inb = (p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < W) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < H)
            ok &= inb
            tracks[alive[ok], t] = p1[ok, 0]
            vis[alive[ok], t] = True
        born = np.nonzero(queries[:, 0] == t)[0]
        vis[born, t] = True
    return tracks, vis


# --------------------------------------------------------------- CoTracker3
def track_cotracker3(frames: np.ndarray, queries: np.ndarray, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(dev)
    q = torch.from_numpy(queries).float()[None].to(dev)
    with torch.no_grad():
        tr, vis = model(video, queries=q)
    return tr[0].permute(1, 0, 2).cpu().numpy(), vis[0].permute(1, 0).cpu().numpy().astype(bool)


# -------------------------------------------------------------------- TAPNet
def track_tapnet(frames: np.ndarray, queries: np.ndarray, checkpoint: str | None = None, device: str | None = None,
                 resize: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """TAPIR / BootsTAPIR in PyTorch (pip install 'tapnet @ git+https://github.com/google-deepmind/tapnet')."""
    import torch
    import torch.nn.functional as F
    from tapnet.torch import tapir_model
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint or tapnet_checkpoint() or "checkpoints/bootstapir_checkpoint_v2.pt"
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(dev).eval()
    T, H, W = frames.shape[:3]
    v = torch.from_numpy(frames).float().to(dev).permute(0, 3, 1, 2)
    v = F.interpolate(v, size=(resize, resize), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    v = (v / 255.0) * 2 - 1
    q = queries.copy()
    q = np.stack([q[:, 0], q[:, 2] * resize / H, q[:, 1] * resize / W], axis=1)  # [t, y, x]
    with torch.no_grad():
        out = model(v[None], torch.from_numpy(q).float()[None].to(dev))
    tracks = out["tracks"][0].cpu().numpy()  # (N, T, 2) xy in resized px
    occ, dist = out["occlusion"][0], out["expected_dist"][0]
    visible = ((1 - torch.sigmoid(occ)) * (1 - torch.sigmoid(dist)) > 0.5).cpu().numpy()
    tracks[..., 0] *= W / resize
    tracks[..., 1] *= H / resize
    return tracks, visible


def tapnet_checkpoint() -> str | None:
    p = os.environ.get("TAPNET_CHECKPOINT") or str(Path(__file__).resolve().parents[1] / "checkpoints" / "bootstapir_checkpoint_v2.pt")
    return p if Path(p).exists() else None


def default_backend() -> str:
    """'tapnet' when its checkpoint is present and a GPU is available (or HUNT_POINTS=tapnet forces it); else 'lk'.
    TAPNet is the intended tracker; LK is the CPU fallback (TAPNet on CPU is ~200x slower)."""
    forced = os.environ.get("HUNT_POINTS")
    if forced:
        return forced
    if tapnet_checkpoint():
        try:
            import torch
            if torch.cuda.is_available():
                return "tapnet"
        except Exception:
            pass
    return "lk"


def track_points(frames, queries, backend: str = "lk", **kw):
    if backend == "auto":
        backend = default_backend()
    if backend == "lk":
        return track_lk(frames, queries)
    if backend == "cotracker3":
        return track_cotracker3(frames, queries, **kw)
    if backend == "tapnet":
        return track_tapnet(frames, queries, **kw)
    raise ValueError(backend)


def survival(visible: np.ndarray, owner: np.ndarray) -> dict:
    """Share of dots alive at the clip end (eval 8), split by body/background."""
    end = visible[:, -1]
    body = owner >= 0
    return {"body": float(end[body].mean()) if body.any() else None, "background": float(end[~body].mean()) if (~body).any() else None}
