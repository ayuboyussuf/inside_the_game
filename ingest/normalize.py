"""Phase 2: trim, resample to 30 fps, resize, and write data/manifest.csv.

Input: a sources CSV (default data/raw/sources.csv) with columns
  clip_id, file, start_s, end_s, outcome, sprint_start_s, decisive_s, licence, source_url, predator_track, prey_track
(only clip_id, file, outcome are required). Times are in the ORIGINAL clip; the manifest
stores frames in the NORMALISED clip.

Uses the system ffmpeg if present, else the static binary shipped by imageio-ffmpeg.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
MANIFEST_COLS = ["clip_id", "file", "outcome", "sprint_start_frame", "decisive_frame", "fps", "duration_s",
                 "licence", "source_url", "predator_track", "prey_track"]


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install ffmpeg or `uv pip install imageio-ffmpeg`") from e


def normalize_clip(src: Path, dst: Path, start: float | None = None, end: float | None = None,
                   fps: int = FPS, max_side: int = 960) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    vf = f"fps={fps},scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    cmd += ["-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def probe_duration(path: Path) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or FPS
        cap.release()
        return float(n / f) if n else 0.0
    except Exception:
        return 0.0


def _f(row, k):
    v = (row.get(k) or "").strip()
    return float(v) if v else None


def run(sources: Path, out_dir: Path, manifest: Path, fps: int = FPS, force: bool = False) -> list[dict]:
    rows = list(csv.DictReader(sources.open(encoding="utf-8")))
    out = []
    for r in rows:
        cid = r["clip_id"].strip()
        src = Path(r["file"])
        src = src if src.is_absolute() else (sources.parent / src)
        dst = out_dir / f"{cid}.mp4"
        start, end = _f(r, "start_s"), _f(r, "end_s")
        if force or not dst.exists():
            normalize_clip(src, dst, start, end, fps)
        off = start or 0.0
        sp, dc = _f(r, "sprint_start_s"), _f(r, "decisive_s")
        out.append({"clip_id": cid, "file": str(dst.relative_to(ROOT)) if dst.is_relative_to(ROOT) else str(dst),
                    "outcome": r["outcome"].strip().lower(),
                    "sprint_start_frame": "" if sp is None else int(round((sp - off) * fps)),
                    "decisive_frame": "" if dc is None else int(round((dc - off) * fps)),
                    "fps": fps, "duration_s": round(probe_duration(dst), 2),
                    "licence": r.get("licence", ""), "source_url": r.get("source_url", ""),
                    "predator_track": r.get("predator_track", ""), "prey_track": r.get("prey_track", "")})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(out)
    return out


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default=str(ROOT / "data/raw/sources.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/clips"))
    ap.add_argument("--manifest", default=str(ROOT / "data/manifest.csv"))
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows = run(Path(a.sources), Path(a.out), Path(a.manifest), a.fps, a.force)
    print(f"normalised {len(rows)} clip(s) -> {a.manifest}")
    for r in rows:
        print(f"  {r['clip_id']:<24} {r['outcome']:<8} {r['duration_s']:>6}s  sprint@{r['sprint_start_frame']}")


if __name__ == "__main__":  # pragma: no cover
    main()
