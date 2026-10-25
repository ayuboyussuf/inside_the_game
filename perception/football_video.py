"""Football video -> tracked players, ball and (when a pitch model is available) pitch coordinates.

This is the production path behind auto-eventing and eval 11. Each piece maps to a failure found in
docs/FOOTBALL_VIDEO_EVAL.md:

  players  fine-tuned football detector (player/goalkeeper/referee/ball) if checkpoints/football_players.pt
           exists, else COCO "person"; BoT-SORT + appearance re-ID + camera-motion compensation
           (configs/football_botsort.yaml), full frame rate, reset at shot cuts
  ball     dedicated ball model if checkpoints/football_ball.pt exists, else COCO "sports ball";
           tiled (SAHI-style) inference so a 6 px ball is seen, then a constant-velocity Kalman tracker
           with gating and gap filling
  pitch    32-keypoint pitch model (checkpoints/football_pitch.pt + football_pitch_keypoints.json from
           training/train_football.py) -> per-frame homography -> metres, km/h; HSV pitch mask otherwise
  cuts     histogram + intensity change; the tracker is reset on a cut
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
TRACKER_CFG = ROOT / "configs" / "football_botsort.yaml"
COCO_PERSON, COCO_BALL = 0, 32


# ------------------------------------------------------------------ models
@dataclass
class FootballModels:
    players: object
    ball: object
    pitch: object | None
    pitch_template: np.ndarray | None
    players_is_football: bool
    ball_is_football: bool
    names: dict = field(default_factory=dict)

    @classmethod
    def load(cls, fallback: str = os.environ.get("HUNT_FOOTBALL_FALLBACK", "yolo11s.pt")) -> "FootballModels":
        from ultralytics import YOLO
        pp, bp, kp = CKPT / "football_players.pt", CKPT / "football_ball.pt", CKPT / "football_pitch.pt"
        players = YOLO(str(pp)) if pp.exists() else YOLO(fallback)
        ball = YOLO(str(bp)) if bp.exists() else (players if pp.exists() else YOLO(fallback))
        pitch, tmpl = None, None
        if kp.exists() and (CKPT / "football_pitch_keypoints.json").exists():
            pitch = YOLO(str(kp))
            tmpl = np.array(json.loads((CKPT / "football_pitch_keypoints.json").read_text())["vertices_m"], float)
        return cls(players, ball, pitch, tmpl, pp.exists(), bp.exists(), dict(players.names))

    def class_ids(self, model, wanted: list[str]) -> list[int]:
        names = model.names
        return [i for i, n in names.items() if n.lower() in wanted]


# ------------------------------------------------------------------ shot cuts
class CutDetector:
    """Hue/saturation histogram correlation + mean intensity jump; robust to pans, catches camera switches."""

    def __init__(self, hist_thr: float = 0.75, jump_thr: float = 28.0):
        self.prev = None
        self.hist_thr, self.jump_thr = hist_thr, jump_thr

    def __call__(self, img) -> bool:
        import cv2
        small = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = False
        if self.prev is not None:
            ph, pg = self.prev
            corr = cv2.compareHist(ph, h, cv2.HISTCMP_CORREL)
            jump = float(np.mean(np.abs(gray - pg)))
            cut = corr < self.hist_thr or (jump > self.jump_thr and corr < 0.9)
        self.prev = (h, gray)
        return cut


# ------------------------------------------------------------------ pitch
def pitch_mask(img):
    """Approximate pitch region (convex hull of the largest grass area). Known limitation (eval 11): it also covers the
    touchline strip, so staff standing there count as on-pitch. Colour rules cannot fix that (fogged grass loses its
    colour; touchline staff often stand on grass); the exact boundary comes from the pitch-keypoint homography."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    filled = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool)


def on_pitch(mask, foot, H=None, length_m: float = 105.0, width_m: float = 68.0) -> bool:
    """With a homography: exact test against the pitch rectangle (1 m tolerance). Without: the approximate mask."""
    if H is not None:
        x, y = to_pitch(H, np.array([foot]))[0]
        return -1.0 <= x <= length_m + 1.0 and -1.0 <= y <= width_m + 1.0
    h, w = mask.shape
    return bool(mask[int(min(h - 1, max(0, foot[1]))), int(min(w - 1, max(0, foot[0])))])


def homography(models: FootballModels, img, min_kps: int = 4, conf: float = 0.5):
    """Image -> pitch (metres) homography from the keypoint model, or None."""
    import cv2
    if models.pitch is None:
        return None
    r = models.pitch.predict(img, verbose=False, conf=0.3)[0]
    if r.keypoints is None or not len(r.keypoints):
        return None
    xy = r.keypoints.xy[0].cpu().numpy()
    kc = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
    ok = kc > conf
    if ok.sum() < min_kps:
        return None
    H, _ = cv2.findHomography(xy[ok].astype(np.float32), models.pitch_template[ok].astype(np.float32), cv2.RANSAC, 3.0)
    return H


def to_pitch(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


# ------------------------------------------------------------------ ball
def ball_candidates(models: FootballModels, img, tile: int = 640, overlap: float = 0.25, conf: float = 0.05,
                    mask=None) -> list[tuple[float, float, float, float]]:
    """Tiled inference: run the ball detector on overlapping 640 px tiles so a 4-10 px ball is visible to it.
    Returns (cx, cy, size, conf) candidates in image pixels, filtered to small, on-pitch blobs."""
    H, W = img.shape[:2]
    model = models.ball
    cls = models.class_ids(model, ["ball", "sports ball", "football"]) or [COCO_BALL]
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(1, W - tile), step)) + [max(0, W - tile)]
    ys = list(range(0, max(1, H - tile), step)) + [max(0, H - tile)]
    tiles = [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]
    crops = [img[y:y + tile, x:x + tile] for x, y in tiles]
    out = []
    for (x, y), r in zip(tiles, model.predict(crops, conf=conf, classes=cls, imgsz=tile, verbose=False)):
        if r.boxes is None:
            continue
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            cx, cy = (b[0] + b[2]) / 2 + x, (b[1] + b[3]) / 2 + y
            size = max(b[2] - b[0], b[3] - b[1])
            if size > 0.06 * H:  # a ball is never that big in a wide shot
                continue
            if mask is not None and not mask[int(min(H - 1, cy)), int(min(W - 1, cx))]:
                continue
            out.append((float(cx), float(cy), float(size), float(c)))
    # merge duplicates from overlapping tiles
    out.sort(key=lambda t: -t[3])
    merged = []
    for c in out:
        if all(np.hypot(c[0] - m[0], c[1] - m[1]) > 6 for m in merged):
            merged.append(c)
    return merged


class BallTracker:
    """Constant-velocity Kalman filter over ball candidates: gating rejects heads/boots, prediction bridges
    short gaps (motion blur, occlusion) so the ball keeps a continuous trajectory."""

    def __init__(self, fps: float, max_gap_s: float = 0.5, gate_px: float = 90.0):
        import cv2
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / fps
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.diag([1, 1, 400, 400]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self.alive, self.missed = False, 0
        self.max_gap = int(max_gap_s * fps)
        self.gate = gate_px

    def reset(self):
        self.alive, self.missed = False, 0

    def step(self, cands) -> tuple[np.ndarray | None, str]:
        if not self.alive:
            if not cands:
                return None, "lost"
            c = max(cands, key=lambda t: t[3])
            if c[3] < 0.2:  # start a track only on a confident detection
                return None, "lost"
            self.kf.statePost = np.array([[c[0]], [c[1]], [0], [0]], np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
            self.alive, self.missed = True, 0
            return np.array(c[:2]), "detected"
        pred = self.kf.predict()[:2, 0]
        near = [c for c in cands if np.hypot(c[0] - pred[0], c[1] - pred[1]) < self.gate * (1 + self.missed * 0.5)]
        if near:
            c = min(near, key=lambda t: np.hypot(t[0] - pred[0], t[1] - pred[1]) - 20 * t[3])
            self.kf.correct(np.array([[c[0]], [c[1]]], np.float32))
            self.missed = 0
            return np.array(c[:2]), "detected"
        self.missed += 1
        if self.missed > self.max_gap:
            self.reset()
            return None, "lost"
        return pred.copy(), "predicted"


# ------------------------------------------------------------------ per-clip run
def iter_frames(path: Path, target_fps: float | None = None, max_side: int = 1280, max_seconds: float | None = None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps))) if target_fps else 1
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            t = i / fps
            if max_seconds and t > max_seconds:
                break
            s = max_side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
            yield t, img, fps / step
        i += 1
    cap.release()


def track_clip(path: Path, models: FootballModels | None = None, target_fps: float = 30.0, max_seconds: float = 12.0,
               imgsz: int = 1280, tiled_ball: bool = True, on_frame=None) -> dict:
    """Run players + ball + pitch on a clip. Returns per-frame records (pixels, and metres when a homography exists)."""
    models = models or FootballModels.load()
    pcls = (models.class_ids(models.players, ["player", "goalkeeper", "referee"]) if models.players_is_football else [COCO_PERSON])
    cuts = CutDetector()
    # pitch size from the keypoint template (roboflow/sports uses a 120 x 70 m configuration)
    dims = (float(models.pitch_template[:, 0].max()), float(models.pitch_template[:, 1].max())) if models.pitch_template is not None else (105.0, 68.0)
    frames, n_cuts, ball_tr = [], 0, None
    for k, (t, img, fps) in enumerate(iter_frames(path, target_fps, max_seconds=max_seconds)):
        if ball_tr is None:
            ball_tr = BallTracker(fps)
        if cuts(img) and k:
            n_cuts += 1
            models.players.predictor = None  # new camera: restart the tracker
            ball_tr.reset()
        mask = pitch_mask(img)
        Hm = homography(models, img)
        r = models.players.track(img, persist=True, classes=pcls, conf=0.15, imgsz=imgsz, tracker=str(TRACKER_CFG), verbose=False)[0]
        H, W = mask.shape
        people = []
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.ones(len(r.boxes), int)
            for b, c, p, i in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), ids):
                foot = (float((b[0] + b[2]) / 2), float(b[3]))
                on = on_pitch(mask, foot, Hm, *dims)
                people.append({"id": int(i), "cls": models.players.names[int(c)], "conf": float(p), "xyxy": [float(v) for v in b],
                               "foot": foot, "on_pitch": on})
        cands = ball_candidates(models, img, mask=mask) if tiled_ball else []
        ball_xy, ball_state = ball_tr.step(cands)
        rec = {"t": t, "cuts": n_cuts, "people": people, "ball": None if ball_xy is None else [float(ball_xy[0]), float(ball_xy[1])],
               "ball_state": ball_state, "n_ball_cands": len(cands), "homography": Hm is not None}
        if Hm is not None:
            rec["people_m"] = to_pitch(Hm, np.array([p["foot"] for p in people])).tolist() if people else []
            rec["ball_m"] = to_pitch(Hm, np.array([ball_xy]))[0].tolist() if ball_xy is not None else None
        frames.append(rec)
        if on_frame:
            on_frame(k, t, img, rec, mask)
    return {"frames": frames, "fps": frames and (1.0 / max(1e-6, np.median(np.diff([f["t"] for f in frames])))) or 0,
            "models": {"players": "football" if models.players_is_football else "coco", "ball": "football" if models.ball_is_football else "coco",
                       "pitch": models.pitch is not None}}
