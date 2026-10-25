"""Eval 11 harness: where does the perception stack fail on real football footage?

    python -m evals.football_video                 # all clips in data/raw/football (SOURCES.json)
    python -m evals.football_video --clips everton_oxford_highlights --model yolo11s.pt

For every clip it measures each stage separately, so failures can be attributed:
  players   detections per frame, on-pitch vs crowd (pitch mask), detector size/model
  ball      share of frames with a ball detection, track continuity
  teams     jersey-colour clustering (k=3: two teams + officials/keepers), per-track label purity
  tracking  ByteTrack ID births per second (fragmentation)
  scene     shot cuts and on-screen graphics (broadcast feeds)
  pitch     can a pitch->image homography be built automatically? (line evidence; no keypoint model yet)
  events    auto-eventing (perception/football_autoevent.py) in image space: passes / turnovers generated

Annotated frames are written to data/evals/football_video/<clip>/ for a human audit; the audit labels
(data/evals/football_video/audit.json) turn the stage metrics into precision/recall in evals.run_all (eval 11).
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "football"
OUT = ROOT / "data" / "evals" / "football_video"
PERSON, BALL = 0, 32


# ------------------------------------------------------------------ frames
def sample(path: Path, target_fps: float = 10.0, max_frames: int = 120, max_side: int = 1280):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps)))
    frames, times, i = [], [], 0
    while len(frames) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            s = max_side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
            frames.append(img)
            times.append(i / fps)
        i += 1
    cap.release()
    return frames, np.array(times), fps, step


# ------------------------------------------------------------------ pitch
def pitch_mask(img):
    """Largest green region (the pitch), closed and hole-filled."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool), 0.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = lab == k
    # fill holes (players stand inside the pitch): convex hull of the region
    ys, xs = np.nonzero(mask)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    filled = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool), float(mask.mean())


def line_evidence(img, mask) -> int:
    """Long white straight lines on the pitch: the raw material for an automatic homography."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255)) & (mask.astype(np.uint8) * 255)
    lines = cv2.HoughLinesP(white, 1, np.pi / 180, 80, minLineLength=int(0.12 * img.shape[1]), maxLineGap=12)
    return 0 if lines is None else len(lines)


def is_cut(a, b) -> bool:
    import cv2
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < 0.6


def graphics_rows(frames) -> float:
    """Share of frame height covered by static, saturated overlays (score bars) across the clip."""
    import cv2
    if len(frames) < 5:
        return 0.0
    st = np.std(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:: max(1, len(frames) // 20)]]), axis=0)
    hsv = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90)  # a static camera's pitch is static too: not a graphic
    overlay = (st < 6) & (hsv[..., 1] > 120) & ~grass
    return float(np.mean(overlay.mean(axis=1) > 0.4))


# ------------------------------------------------------------------ teams
def jersey_feature(img, box, mask):
    import cv2
    x0, y0, x1, y1 = [int(v) for v in box]
    h = y1 - y0
    crop = img[max(0, y0 + int(0.15 * h)):max(0, y0 + int(0.5 * h)), max(0, x0):max(0, x1)]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = ~((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 40))  # drop grass pixels
    if keep.sum() < 10:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[keep].astype(float)
    return np.median(lab, axis=0)


# ------------------------------------------------------------------ main per clip
def run_clip(name: str, path: Path, model: str = "yolo11s.pt", imgsz: int = 1280, conf: float = 0.15) -> dict:
    import cv2
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from ultralytics import YOLO

    from perception.football_autoevent import events_from_tracks

    t0 = time.time()
    frames, times, fps, step = sample(path)
    m = YOLO(model)
    od = OUT / name
    od.mkdir(parents=True, exist_ok=True)
    per_frame, tracks = [], {}
    cuts = 0
    for k, img in enumerate(frames):
        if k and is_cut(frames[k - 1], img):
            cuts += 1
            m.predictor = None  # reset tracker state at a cut (new camera)
        r = m.track(img, persist=True, classes=[PERSON, BALL], conf=conf, imgsz=imgsz, tracker="bytetrack.yaml", verbose=False)[0]
        mask, green = pitch_mask(img)
        H, W = mask.shape
        dets = []
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.ones(len(r.boxes), int)
            for xyxy, c, p, i in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), ids):
                cx, by = int((xyxy[0] + xyxy[2]) / 2), int(min(H - 1, xyxy[3]))
                on = bool(mask[by, min(W - 1, max(0, cx))])
                d = {"cls": "ball" if int(c) == BALL else "person", "conf": float(p), "xyxy": [float(v) for v in xyxy],
                     "id": int(i), "on_pitch": on}
                dets.append(d)
                if i >= 0:
                    tracks.setdefault(int(i), []).append((k, d))
        per_frame.append({"t": float(times[k]), "green": green, "lines": line_evidence(img, mask), "dets": dets})
        if k % max(1, len(frames) // 6) == 0:  # annotated frames for the human audit
            vis = img.copy()
            vis[~mask] = (vis[~mask] * 0.55).astype(np.uint8)
            for d in dets:
                x0, y0, x1, y1 = [int(v) for v in d["xyxy"]]
                col = (0, 255, 255) if d["cls"] == "ball" else ((0, 200, 0) if d["on_pitch"] else (0, 0, 255))
                cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            cv2.putText(vis, f"{name} t={times[k]:.1f}s", (10, 30), 0, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(od / f"audit_{k:04d}.jpg"), vis)
    # ---------- stage metrics
    persons = [[d for d in f["dets"] if d["cls"] == "person"] for f in per_frame]
    on = [sum(d["on_pitch"] for d in p) for p in persons]
    off = [sum(not d["on_pitch"] for d in p) for p in persons]
    balls = [[d for d in f["dets"] if d["cls"] == "ball"] for f in per_frame]
    ball_rate = float(np.mean([len(b) > 0 for b in balls])) if balls else 0.0
    heights = [d["xyxy"][3] - d["xyxy"][1] for p in persons for d in p if d["on_pitch"]]
    dur = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    person_tracks = {i: v for i, v in tracks.items() if v[0][1]["cls"] == "person"}
    # teams: cluster jersey colours of on-pitch players, then purity of each track's label over time
    feats, owners = [], []
    for i, obs in person_tracks.items():
        for k, d in obs[:: max(1, len(obs) // 8)]:
            if d["on_pitch"]:
                f = jersey_feature(frames[k], d["xyxy"], None)
                if f is not None:
                    feats.append(f)
                    owners.append(i)
    team = {"clusters": None, "silhouette": None, "track_purity": None}
    if len(feats) >= 12:
        X = np.array(feats)
        km = KMeans(3, n_init=10, random_state=0).fit(X)
        team["clusters"] = np.bincount(km.labels_, minlength=3).tolist()
        team["silhouette"] = round(float(silhouette_score(X, km.labels_)), 3)
        pur = []
        for i in set(owners):
            lab = km.labels_[np.array(owners) == i]
            if len(lab) >= 3:
                pur.append(np.bincount(lab).max() / len(lab))
        team["track_purity"] = round(float(np.mean(pur)), 3) if pur else None
    # auto-eventing in image space (no homography): possession radius scaled by player height
    events = []
    btr = [v for v in tracks.values() if v[0][1]["cls"] == "ball"]
    if btr:
        main = max(btr, key=len)
        bpos = np.full((len(frames), 2), np.nan)
        for k, d in main:
            bpos[k] = [(d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2]
        ok = ~np.isnan(bpos[:, 0])
        if ok.sum() >= 5:
            idx = np.arange(len(frames))
            bpos = np.c_[np.interp(idx, idx[ok], bpos[ok, 0]), np.interp(idx, idx[ok], bpos[ok, 1])]
            players = {}
            for i, obs in person_tracks.items():
                p = np.full((len(frames), 2), np.nan)
                for k, d in obs:
                    p[k] = [(d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]]
                players[str(i)] = p
            if players:
                scale = (np.median(heights) / 1.8) if heights else 20.0  # px per metre from player height
                ev = events_from_tracks(times, bpos / scale, {k: v / scale for k, v in players.items()},
                                        {k: "A" for k in players})
                events = [e["type"] for e in ev]
    res = {
        "clip": name, "frames_sampled": len(frames), "source_fps": fps, "stride": step, "duration_s": round(dur, 1),
        "model": model, "imgsz": imgsz, "seconds": round(time.time() - t0, 1),
        "players": {"on_pitch_per_frame": round(float(np.mean(on)), 1), "crowd_or_bench_per_frame": round(float(np.mean(off)), 1),
                    "median_height_px": round(float(np.median(heights)), 1) if heights else None},
        "ball": {"frames_with_ball": round(ball_rate, 3), "n_ball_tracks": len(btr),
                 "longest_ball_track_frames": max((len(v) for v in btr), default=0)},
        "tracking": {"person_tracks": len(person_tracks), "id_births_per_s": round(len(person_tracks) / max(dur, 1e-6), 2),
                     "median_track_len_s": round(float(np.median([len(v) for v in person_tracks.values()])) / (fps / step), 2) if person_tracks else 0},
        "teams": team,
        "scene": {"cuts": cuts, "graphics_rows_share": round(graphics_rows(frames), 3),
                  "pitch_green_share": round(float(np.mean([f["green"] for f in per_frame])), 3)},
        "pitch": {"mean_long_lines": round(float(np.mean([f["lines"] for f in per_frame])), 1), "homography": "not available (no pitch keypoint model)"},
        "events": {"generated": len(events), "passes": events.count("pass"), "turnovers": events.count("possession_change")},
    }
    (od / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    src = json.loads((RAW / "SOURCES.json").read_text())
    names = a.clips or [Path(k).stem for k in src]
    allres = []
    for fn in src:
        n = Path(fn).stem
        if n in names:
            r = run_clip(n, RAW / fn, a.model, a.imgsz)
            print(json.dumps(r), flush=True)
            allres.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{Path(a.model).stem}.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__" and not os.environ.get("FOOTBALL_EVAL_V2"):
    main()


# ------------------------------------------------------------------ v2: the production pipeline (perception/football_video.py)
def run_clip_v2(name: str, path: Path, max_seconds: float = 8.0, audit_times: list[float] | None = None) -> dict:
    """Same clip, production path: BoT-SORT + re-ID at full frame rate, tiled ball + Kalman, cut reset, fixed pitch mask."""
    import cv2

    from perception.football_video import FootballModels, track_clip
    od = OUT / name / "v2"
    od.mkdir(parents=True, exist_ok=True)
    want = sorted(audit_times or [])
    saved = {}

    def on_frame(k, t, img, rec, mask):
        for ta in want:
            if ta not in saved and abs(t - ta) < 0.5 / 30 + 1e-6 or (ta not in saved and t > ta):
                vis = img.copy()
                vis[~mask] = (vis[~mask] * 0.55).astype(np.uint8)
                for p in rec["people"]:
                    x0, y0, x1, y1 = [int(v) for v in p["xyxy"]]
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0) if p["on_pitch"] else (0, 0, 255), 2)
                if rec["ball"]:
                    cv2.circle(vis, (int(rec["ball"][0]), int(rec["ball"][1])), 9, (0, 255, 255), 2)
                cv2.imwrite(str(od / f"audit_t{ta:.1f}.jpg"), vis)
                saved[ta] = rec

    t0 = time.time()
    res = track_clip(path, FootballModels.load(), target_fps=30.0, max_seconds=max_seconds, on_frame=on_frame)
    fr = res["frames"]
    fps = res["fps"] or 30.0
    tracks: dict[int, int] = {}
    onp: dict[int, int] = {}
    for f in fr:
        for p in f["people"]:
            if p["id"] >= 0:
                tracks[p["id"]] = tracks.get(p["id"], 0) + 1
                onp[p["id"]] = onp.get(p["id"], 0) + int(p["on_pitch"])
    long_ids = {i for i, n in tracks.items() if n >= 2.0 * (res["fps"] or 30.0)}
    coverage = sum(onp[i] for i in long_ids) / max(1, sum(onp.values()))
    dur = (fr[-1]["t"] - fr[0]["t"]) if len(fr) > 1 else 1.0
    states = [f["ball_state"] for f in fr]
    out = {"clip": name, "pipeline": "v2", "frames": len(fr), "fps": round(fps, 1), "seconds": round(time.time() - t0, 1),
           "models": res["models"],
           "players": {"on_pitch_per_frame": round(float(np.mean([sum(p["on_pitch"] for p in f["people"]) for f in fr])), 1),
                       "off_pitch_per_frame": round(float(np.mean([sum(not p["on_pitch"] for p in f["people"]) for f in fr])), 1)},
           "tracking": {"person_tracks": len(tracks), "id_births_per_s": round(len(tracks) / max(dur, 1e-6), 2),
                        "median_track_len_s": round(float(np.median(list(tracks.values()))) / fps, 2) if tracks else 0.0,
                        "coverage_2s": round(float(coverage), 3)},
           "ball": {"detected_share": round(states.count("detected") / max(1, len(states)), 3),
                    "with_position_share": round((states.count("detected") + states.count("predicted")) / max(1, len(states)), 3)},
           "scene": {"cuts": fr[-1]["cuts"] if fr else 0},
           "pitch": {"homography_share": round(float(np.mean([f["homography"] for f in fr])), 3) if fr else 0.0},
           "audit_frames": {f"{k:.1f}": {"ball": v["ball"], "ball_state": v["ball_state"],
                                          "people": [{"xyxy": p["xyxy"], "on_pitch": p["on_pitch"]} for p in v["people"]]}
                            for k, v in saved.items()}}
    (od / "result.json").write_text(json.dumps(out, indent=1))
    return out


def main_v2(clips: list[str] | None = None, max_seconds: float = 8.0):
    src = json.loads((RAW / "SOURCES.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())["frames"] if (OUT / "audit.json").exists() else []
    allres = []
    for fn in src:
        n = Path(fn).stem
        if clips and n not in clips:
            continue
        times = [a["t"] for a in audit if a["clip"] == n and a["t"] <= max_seconds]
        r = run_clip_v2(n, RAW / fn, max_seconds, times)
        print(json.dumps({k: v for k, v in r.items() if k != "audit_frames"}), flush=True)
        allres.append(r)
    (OUT / "results_v2.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__" and os.environ.get("FOOTBALL_EVAL_V2"):
    main_v2()
