"""Overlay video: boxes with IDs + coloured dots (+ short trails). Watch it yourself (phase 3 check)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(abs(int(i)) * 7919 + 13)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def render_overlay(video: Path, tracks_json: dict, tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray,
                   out: Path, trail: int = 8, roles: dict[int, str] | None = None) -> Path:
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames = {fr["f"]: fr["dets"] for fr in tracks_json["frames"]}
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for d in frames.get(f, []):
            x0, y0, x1, y1 = [int(v) for v in d["xyxy"]]
            c = _color(d["id"])
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
            label = f"{d['id']} {d['cls']}" + (f" [{roles[d['id']]}]" if roles and d["id"] in roles else "")
            cv2.putText(img, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        if f < tracks.shape[1]:
            for i in range(len(tracks)):
                if not visible[i, f]:
                    continue
                c = (200, 200, 200) if owner[i] < 0 else _color(owner[i])
                for k in range(max(0, f - trail), f):
                    if visible[i, k] and visible[i, k + 1]:
                        cv2.line(img, tuple(int(v) for v in tracks[i, k]), tuple(int(v) for v in tracks[i, k + 1]), c, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(int(v) for v in tracks[i, f]), 2, c, -1, cv2.LINE_AA)
        vw.write(img)
        f += 1
    cap.release()
    vw.release()
    # re-encode to H.264 so browsers can play it
    try:
        import subprocess

        from ingest.normalize import ffmpeg_exe
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", str(out)], check=True)
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.replace(out)
    return out
