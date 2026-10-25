"""Named joints -> heading, plus the 'what the player saw' vision-cone geometry.

* YOLO-pose (17 human joints) for football players.
* DeepLabCut SuperAnimal-Quadruped for animals (nose, spine, legs) -> cleaner heading/gait than raw dots.
Both are optional heavy deps; the geometry helpers are pure numpy and unit-tested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COCO_NOSE, COCO_LSH, COCO_RSH = 0, 5, 6


def yolo_pose(video: Path, model: str = "yolo11m-pose.pt"):  # pragma: no cover - needs weights
    from ultralytics import YOLO
    m = YOLO(model)
    out = []
    for f, r in enumerate(m.track(source=str(video), stream=True, persist=True, verbose=False)):
        k = r.keypoints
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.id is not None else []
        out.append({"f": f, "people": [{"id": int(i), "kp": kp.tolist()} for i, kp in zip(ids, k.xy.cpu().numpy())] if k is not None else []})
    return out


def superanimal_quadruped(video: Path, dest: Path):  # pragma: no cover - needs DeepLabCut
    import deeplabcut
    deeplabcut.video_inference_superanimal([str(video)], "superanimal_quadruped", model_name="hrnet_w32",
                                           detector_name="fasterrcnn_resnet50_fpn_v2", dest_folder=str(dest))
    return dest


def heading_from_person(kp: np.ndarray) -> np.ndarray | None:
    """Facing direction ~ perpendicular to the shoulder line, oriented toward the nose."""
    ls, rs, nose = kp[COCO_LSH], kp[COCO_RSH], kp[COCO_NOSE]
    if np.any(np.isnan([*ls, *rs])) or np.allclose(ls, 0) or np.allclose(rs, 0):
        return None
    sh = rs - ls
    perp = np.array([-sh[1], sh[0]])
    mid = (ls + rs) / 2
    if not np.allclose(nose, 0) and np.dot(nose - mid, perp) < 0:
        perp = -perp
    n = np.linalg.norm(perp)
    return perp / n if n > 1e-6 else None


def heading_from_spine(nose: np.ndarray, tail_base: np.ndarray) -> np.ndarray:
    v = np.asarray(nose, float) - np.asarray(tail_base, float)
    return v / (np.linalg.norm(v) + 1e-9)


def in_vision_cone(eye: np.ndarray, heading: np.ndarray, pts: np.ndarray, fov_deg: float = 120.0) -> np.ndarray:
    d = np.asarray(pts, float) - np.asarray(eye, float)
    cos = (d @ heading) / (np.linalg.norm(d, axis=1) * np.linalg.norm(heading) + 1e-9)
    return cos >= np.cos(np.radians(fov_deg / 2))


def blocks_line(a: np.ndarray, b: np.ndarray, box: list[float], pad: float = 0.0) -> bool:
    """Does an axis-aligned box intersect segment a->b (Liang-Barsky)?"""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def screened(keeper_eye: np.ndarray, ball: np.ndarray, defender_boxes: list[list[float]], heading: np.ndarray | None = None,
             fov_deg: float = 120.0) -> int:
    """Screened-keeper feature: defenders whose box sits on the keeper->ball sight line (inside the cone)."""
    n = 0
    for bx in defender_boxes:
        c = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])
        if heading is not None and not in_vision_cone(keeper_eye, heading, c[None], fov_deg)[0]:
            continue
        n += blocks_line(keeper_eye, ball, bx)
    return n
