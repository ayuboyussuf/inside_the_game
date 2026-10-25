"""Detect + track bodies -> tracks.json (boxes and stable IDs per frame).

Backends
  yolo        Ultralytics YOLO (COCO: cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, person,
              sports ball) with built-in ByteTrack / BoT-SORT.
  yolo-world  Open-vocabulary YOLO-World: prompt "cheetah", "gazelle", "lion" when COCO misses animals.
  blobs       Colour-blob detector + Hungarian tracker for clips rendered by sim/render.py (CPU tests).

Output format:
  {"fps": 30, "size": [W, H], "backend": "...", "frames": [{"f": 0, "dets": [{"id": 1, "cls": "cat", "conf": .9, "xyxy": [...]}, ...]}, ...]}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # COCO: cat dog horse sheep cow elephant bear zebra giraffe
FOOTBALL_CLASSES = [0, 32]                              # person, sports ball


def _video_meta(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, W, H


def track_yolo(video: Path, model: str = os.environ.get("HUNT_YOLO_MODEL", "yolo11n.pt"), tracker: str = "bytetrack.yaml", classes=None,
               conf: float = 0.2, prompts: list[str] | None = None, device: str | None = None) -> dict:
    from ultralytics import YOLO
    imgsz = 640
    if prompts:
        # open vocabulary: "cat, pigeon", "cheetah, gazelle" ... Larger input so small prey (a distant bird) is still seen.
        from ultralytics import YOLOWorld
        wm = model if "world" in model else os.environ.get("HUNT_YOLOWORLD_MODEL", "yolov8m-worldv2.pt")
        local = Path(__file__).resolve().parents[1] / "checkpoints" / wm
        m = YOLOWorld(str(local) if local.exists() else wm)
        m.set_classes(prompts)
        classes, imgsz, conf, model = None, 1280, min(conf, 0.1), wm
    else:
        m = YOLO(model)
    fps, W, H = _video_meta(video)
    frames = []
    for f, r in enumerate(m.track(source=str(video), stream=True, persist=True, tracker=tracker, classes=classes,
                                  conf=conf, imgsz=imgsz, verbose=False, device=device)):
        dets = []
        b = r.boxes
        if b is not None and b.id is not None:
            for xyxy, i, c, p in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                dets.append({"id": int(i), "cls": r.names[int(c)], "conf": round(float(p), 3), "xyxy": [round(float(v), 1) for v in xyxy]})
        frames.append({"f": f, "dets": dets})
    return {"fps": fps, "size": [W, H], "backend": "yolo-world" if prompts else "yolo", "model": model, "tracker": tracker, "frames": frames}


# --------------------------------------------------------------------- blobs
BLOB_CLASSES = {"predator": (0, 140, 255), "prey": (255, 0, 200), "pack": (0, 255, 255), "herd": (255, 200, 0)}


def track_blobs(video: Path, tol: int = 45, min_area: int = 40, max_jump: float = 60.0) -> dict:
    import cv2
    fps, W, H = _video_meta(video)
    cap = cv2.VideoCapture(str(video))
    tracks: dict[int, dict] = {}
    next_id = 1
    frames = []
    f = 0
    kernel = np.ones((5, 5), np.uint8)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        dets = []
        for cls, bgr in BLOB_CLASSES.items():
            d = np.abs(img.astype(int) - np.array(bgr)).max(axis=2)
            mask = cv2.morphologyEx((d < tol).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
            for k in range(1, n):
                x, y, w, h, a = stats[k]
                if a >= min_area:
                    dets.append({"cls": cls, "xyxy": [float(x), float(y), float(x + w), float(y + h)], "c": cent[k], "conf": 1.0})
        # Hungarian association per class against live tracks
        for cls in BLOB_CLASSES:
            D = [d for d in dets if d["cls"] == cls]
            T = [tid for tid, tr in tracks.items() if tr["cls"] == cls and f - tr["last"] <= 15]
            if D and T:
                C = np.array([[np.linalg.norm(d["c"] - tracks[t]["c"]) for t in T] for d in D])
                ri, ci = linear_sum_assignment(C)
                used = set()
                for r_, c_ in zip(ri, ci):
                    if C[r_, c_] < max_jump:
                        D[r_]["id"] = T[c_]
                        used.add(r_)
            for d in D:
                if "id" not in d:
                    d["id"] = next_id
                    next_id += 1
                tracks[d["id"]] = {"cls": cls, "c": d["c"], "last": f}
        frames.append({"f": f, "dets": [{"id": d["id"], "cls": d["cls"], "conf": 1.0, "xyxy": [round(v, 1) for v in d["xyxy"]]} for d in dets]})
        f += 1
    cap.release()
    return {"fps": fps, "size": [W, H], "backend": "blobs", "frames": frames}


# --------------------------------------------------------------------- motion (class-agnostic fallback)
def track_motion(video: Path, min_area: int = 12, max_area_frac: float = 0.25, max_jump: float = 90.0,
                 warmup: int = 15, keep: int = 6) -> dict:
    """Moving-object detector: frames are stabilised (global similarity transform) to remove hand-held shake,
    then MOG2 background subtraction + size-gated Hungarian association. Finds animals YOLO's COCO classes miss (snakes, small birds, rodents)."""
    import cv2
    fps, W, H = _video_meta(video)
    cap = cv2.VideoCapture(str(video))
    bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=24, detectShadows=False)
    k3, k7 = np.ones((3, 3), np.uint8), np.ones((7, 7), np.uint8)
    tracks: dict[int, dict] = {}
    next_id, f, frames = 1, 0, []
    prev_gray, M = None, np.eye(3)  # M maps the current frame into the first frame (stabilisation)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(prev_gray, 300, 0.01, 12)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
                ok_ = st.ravel() == 1
                if ok_.sum() >= 8:
                    A, _ = cv2.estimateAffinePartial2D(p1[ok_], p0[ok_], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                    if A is not None:
                        M = M @ np.vstack([A, [0, 0, 1]])
        prev_gray = gray
        stab = cv2.warpAffine(img, M[:2], (W, H), borderMode=cv2.BORDER_REPLICATE)
        blur = cv2.GaussianBlur(stab, (5, 5), 0)
        m = bg.apply(blur, learningRate=-1 if f < warmup else 0.005)
        Minv = np.linalg.inv(M)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k7)
        dets = []
        if f >= warmup // 2:
            n, lab, stats, cent = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
            comps = [(stats[k], cent[k]) for k in range(1, n) if min_area <= stats[k][4] <= max_area_frac * W * H]
            comps.sort(key=lambda c: -c[0][4])
            for (x, y, w, h, a), c in comps[:keep]:
                # back to original (unstabilised) frame coordinates, like every other detector's boxes
                corners = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]], float) @ Minv[:2].T
                x0, y0 = corners.min(0)
                x1, y1 = corners.max(0)
                dets.append({"cls": "moving", "xyxy": [float(x0), float(y0), float(x1), float(y1)],
                             "c": np.array([(x0 + x1) / 2, (y0 + y1) / 2])})
        T = [tid for tid, tr in tracks.items() if f - tr["last"] <= 15]
        if dets and T:
            # gate by distance relative to body size AND by size similarity, so a small rat and a long snake don't swap IDs
            C = np.full((len(dets), len(T)), 1e6)
            for r_, d in enumerate(dets):
                ad = _area(d["xyxy"])
                for c_, t in enumerate(T):
                    tr_ = tracks[t]
                    dist = np.linalg.norm(d["c"] - tr_["c"])
                    ratio = max(ad, tr_["area"]) / max(1.0, min(ad, tr_["area"]))
                    if dist < min(max_jump, 1.5 * tr_["diag"] + 25) and ratio < 4.0:
                        C[r_, c_] = dist / (tr_["diag"] + 1) + np.log(ratio)
            for r_, c_ in zip(*linear_sum_assignment(C)):
                if C[r_, c_] < 1e6:
                    dets[r_]["id"] = T[c_]
        for d in dets:
            if "id" not in d:
                d["id"] = next_id
                next_id += 1
            x0, y0, x1, y1 = d["xyxy"]
            tracks[d["id"]] = {"c": d["c"], "last": f, "area": _area(d["xyxy"]), "diag": float(np.hypot(x1 - x0, y1 - y0))}
        frames.append({"f": f, "dets": [{"id": d["id"], "cls": d["cls"], "conf": 1.0, "xyxy": [round(v, 1) for v in d["xyxy"]]} for d in dets]})
        f += 1
    cap.release()
    return {"fps": fps, "size": [W, H], "backend": "motion", "frames": frames}


def _area(b) -> float:
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def merge_detectors(primary: dict, secondary: dict, id_offset: int = 100000) -> dict:
    """Add secondary detections (e.g. motion blobs) that do not overlap any primary (YOLO) box."""
    sec = {fr["f"]: fr["dets"] for fr in secondary["frames"]}
    frames = []
    for fr in primary["frames"]:
        pb = [d["xyxy"] for d in fr["dets"]]
        extra = []
        for d in sec.get(fr["f"], []):
            cx, cy = (d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2
            inside = any(b[0] - 0.2 * (b[2] - b[0]) <= cx <= b[2] + 0.2 * (b[2] - b[0]) and
                         b[1] - 0.2 * (b[3] - b[1]) <= cy <= b[3] + 0.2 * (b[3] - b[1]) for b in pb)
            if not inside:
                extra.append({**d, "id": d["id"] + id_offset})
        frames.append({"f": fr["f"], "dets": fr["dets"] + extra})
    return {**primary, "frames": frames, "backend": primary.get("backend", "yolo") + "+motion"}


# --------------------------------------------------------------------- track post-processing
def moving_ids(tr: dict, min_bodies: float = 1.0, min_frames: int = 10) -> set[int]:
    """Tracks that travel at least `min_bodies` of their own size (min_bodies=1 drops jittering static edges;
    a lower value keeps a slowly stalking animal)."""
    pts: dict[int, list] = {}
    for fr in tr["frames"]:
        for d in fr["dets"]:
            x0, y0, x1, y1 = d["xyxy"]
            pts.setdefault(d["id"], []).append(((x0 + x1) / 2, (y0 + y1) / 2, np.hypot(x1 - x0, y1 - y0)))
    out = set()
    for i, p in pts.items():
        if len(p) < min_frames:
            continue
        a = np.array(p)
        spread = np.max(np.linalg.norm(a[:, :2] - np.median(a[:, :2], axis=0), axis=1))
        if spread >= min_bodies * np.median(a[:, 2]):
            out.add(i)
    return out



def _spans(tr: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for fr in tr["frames"]:
        for d in fr["dets"]:
            s = out.setdefault(d["id"], {"first": fr["f"], "last": fr["f"], "n": 0, "first_box": d["xyxy"], "votes": {}})
            s["last"], s["last_box"], s["n"] = fr["f"], d["xyxy"], s["n"] + 1
            s["votes"][d["cls"]] = s["votes"].get(d["cls"], 0) + 1
    for s in out.values():
        s["cls"] = max(s["votes"], key=s["votes"].get)  # majority label (detectors flicker between classes)
    return out


def stitch_tracks(tr: dict, max_gap: int = 20, max_gap_same_class: int = 60) -> dict:
    """Merge fragments of one animal: B starts soon after A ends, near where A ended. Fragments carrying the same
    semantic label (e.g. two "cat" tracks from YOLO-World) may be bridged across a longer occlusion gap."""
    sp = _spans(tr)
    remap: dict[int, int] = {}
    order = sorted(sp, key=lambda i: sp[i]["first"])
    for b in order:
        best, bd = None, None
        for a in order:
            ra = remap.get(a, a)
            same = sp[a]["cls"] == sp[b]["cls"] and sp[a]["cls"] != "moving"
            limit = max_gap_same_class if same else max_gap
            if a == b or ra == b or sp[a]["last"] >= sp[b]["first"] or sp[b]["first"] - sp[a]["last"] > limit:
                continue
            ax0, ay0, ax1, ay1 = sp[a]["last_box"]
            bx0, by0, bx1, by1 = sp[b]["first_box"]
            diag = max(np.hypot(ax1 - ax0, ay1 - ay0), np.hypot(bx1 - bx0, by1 - by0))
            dist = np.hypot((ax0 + ax1 - bx0 - bx1) / 2, (ay0 + ay1 - by0 - by1) / 2)
            gap = sp[b]["first"] - sp[a]["last"]
            if dist <= 1.5 * diag + 8 * gap and (bd is None or dist < bd):
                # no fragment already merged into `ra` may overlap b in time
                clash = any(remap.get(x, x) == ra and x != b and not (sp[x]["last"] < sp[b]["first"] or sp[x]["first"] > sp[b]["last"])
                            for x in order)
                if not clash:
                    best, bd = ra, dist
        if best is not None:
            remap[b] = best
            for k, v in list(remap.items()):
                if v == b:
                    remap[k] = best
    if not remap:
        return tr
    frames = []
    for fr in tr["frames"]:
        seen, dets = set(), []
        for d in sorted(fr["dets"], key=lambda d: -d.get("conf", 1.0)):
            nid = remap.get(d["id"], d["id"])
            if nid not in seen:
                seen.add(nid)
                dets.append({**d, "id": nid})
        frames.append({"f": fr["f"], "dets": dets})
    return {**tr, "frames": frames, "stitched": {str(k): v for k, v in remap.items()}}


def best_pair(tr: dict, min_overlap: int = 10) -> tuple[int, int, int] | None:
    """The two tracks visible together for the most frames: (id_a, id_b, overlap)."""
    present: dict[int, set] = {}
    for fr in tr["frames"]:
        for d in fr["dets"]:
            present.setdefault(d["id"], set()).add(fr["f"])
    # both animals must not be frozen, and at least one must really travel (static edges never do)
    alive, mov = moving_ids(tr, 0.3, min_overlap), moving_ids(tr, 1.0, min_overlap)
    ids = [i for i, s in present.items() if len(s) >= min_overlap and i in alive]
    best = None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ov = len(present[ids[i]] & present[ids[j]])
            if (ids[i] in mov or ids[j] in mov) and ov >= min_overlap and (best is None or ov > best[2]):
                best = (ids[i], ids[j], ov)
    return best


def id_switches(tr: dict) -> int:
    """Count track-ID births after the first frame (proxy for ID switches / fragmentation)."""
    seen, births = set(), 0
    for fr in tr["frames"]:
        for d in fr["dets"]:
            if d["id"] not in seen:
                if fr["f"] > 0:
                    births += 1
                seen.add(d["id"])
    return births


def boxes_by_id(tr: dict) -> dict[int, dict[int, list[float]]]:
    out: dict[int, dict[int, list[float]]] = {}
    for fr in tr["frames"]:
        for d in fr["dets"]:
            out.setdefault(d["id"], {})[fr["f"]] = d["xyxy"]
    return out
