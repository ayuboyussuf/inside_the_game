"""Perception driver with caching: video -> tracks.json, points.npz, camera.npz, episode.json, overlay.mp4.

  python -m perception.run --clip cheetah_01                     # uses data/manifest.csv
  python -m perception.run --all --detector yolo --points cotracker3
  python -m perception.run --video data/clips/x.mp4 --clip x --outcome caught --detector blobs --points lk

Rule: every model output is cached under data/cache/<clip_id>/; vision models never re-run if the cache exists
(pass --force to recompute).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .camera import estimate_from_points
from .detect_track import (ANIMAL_CLASSES, best_pair, boxes_by_id, id_switches, merge_detectors, stitch_tracks,
                           track_blobs, track_motion, track_yolo)
from .overlay import render_overlay
from .points import default_backend, read_frames, seed_queries, survival, track_points
from .to_episode import build_episode

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
EPISODES = ROOT / "data" / "episodes"


def _usable_tracks(tr: dict, min_frames: int = 10) -> int:
    from collections import Counter
    c = Counter(d["id"] for f in tr["frames"] for d in f["dets"])
    return sum(1 for n in c.values() if n >= min_frames)


def process(clip_id: str, video: Path, outcome: str, detector: str = "yolo", points: str = "auto", prompts=None,
            predator: int | None = None, prey: int | None = None, force: bool = False, overlay: bool = True,
            meta: dict | None = None, max_side: int = 512, publish: bool = True, progress=None) -> dict:
    """publish=False keeps the episode out of data/episodes (the "Real clips" library), e.g. for rendered test clips.
    progress: optional callback(step: str) for UI status."""
    say = progress or (lambda s: None)
    if points == "auto":
        points = default_backend()
    d = CACHE / clip_id
    d.mkdir(parents=True, exist_ok=True)
    # 1. boxes + IDs
    tj = d / "tracks.json"
    if force or not tj.exists():
        say("detecting and tracking animals")
        if detector == "blobs":
            tr = stitch_tracks(track_blobs(video))
        elif detector == "motion":
            tr = stitch_tracks(track_motion(video))
        else:
            if prompts:
                say(f"detecting {', '.join(prompts)} with YOLO-World (open vocabulary)")
            tr = stitch_tracks(track_yolo(video, classes=None if prompts else ANIMAL_CLASSES, prompts=prompts))
            if not prompts and best_pair(tr) is None:
                # COCO often mislabels wild animals (cheetah -> "cat"/"dog", gazelle -> "sheep"); retry with every class
                say("animal classes found fewer than two animals together; retrying with all classes")
                tr = track_yolo(video, classes=None)
                tr["frames"] = [{"f": f["f"], "dets": [d for d in f["dets"] if d["cls"] != "person"]} for f in tr["frames"]]
                tr = stitch_tracks(tr)
            if best_pair(tr) is None:
                # small or non-COCO animals (snakes, rodents, distant birds): add class-agnostic motion detections
                say("YOLO could not see both animals at once; adding the motion detector")
                motion = stitch_tracks(track_motion(video))
                hybrid = stitch_tracks(merge_detectors(tr, motion))
                tr = hybrid if best_pair(hybrid) is not None else motion
        pair = best_pair(tr)
        if pair is None:
            raise ValueError("Could not find two animals (predator and prey) visible at the same time in this clip. "
                             "Try a shorter clip where both animals are on screen, or a wider shot.")
        tr["pair"] = list(pair[:2])
        tj.write_text(json.dumps(tr))
    tr = json.loads(tj.read_text())
    fps = float(tr["fps"])
    # 2. dots (seeded in boxes + background)
    pz = d / f"points_{points}.npz"
    frames = None
    W0 = tr["size"][0]
    if force or not pz.exists():
        say("tracking body and background dots")
        frames = read_frames(video, max_side=max_side)
        s = frames.shape[2] / W0
        boxes = {i: {f: [v * s for v in b] for f, b in bx.items()} for i, bx in boxes_by_id(tr).items()}
        q, owner = seed_queries(frames, boxes)
        tracks, vis = track_points(frames, q, points)
        np.savez_compressed(pz, tracks=tracks / s, visible=vis, owner=owner, queries=q / np.array([1, s, s]))
    z = np.load(pz)
    tracks, vis, owner = z["tracks"], z["visible"], z["owner"]
    # 3. camera compensation
    cz = d / f"camera_{points}.npz"
    if force or not cz.exists():
        say("removing camera motion")
        frames = read_frames(video, max_side=None) if frames is None else frames
        s = frames.shape[2] / W0
        if s != 1.0:
            frames = read_frames(video)
        to_ref = estimate_from_points(tracks, vis, owner, frames=frames, boxes=boxes_by_id(tr))
        np.savez_compressed(cz, to_ref=to_ref)
    to_ref = np.load(cz)["to_ref"]
    # 4. episode
    say("building the event timeline")
    ep = build_episode(clip_id, tr, tracks, vis, owner, to_ref, outcome, fps, predator, prey,
                       meta={"video": str(video), "detector": tr.get("backend"), "points": points, **(meta or {})})
    (d / "episode.json").write_text(ep.model_dump_json(indent=1))
    if publish:
        EPISODES.mkdir(parents=True, exist_ok=True)
        (EPISODES / f"{clip_id}.json").write_text(ep.model_dump_json())
    health = {"dots_alive_at_end": survival(vis, owner), "id_births_after_first_frame": id_switches(tr),
              "n_dots": int(len(owner))}
    (d / "health.json").write_text(json.dumps(health, indent=1))
    if overlay and (force or not (d / "overlay.mp4").exists()):
        say("rendering the overlay video")
        ids = ep.meta.get("track_ids", {})
        roles = {tid: next(a.role for a in ep.actors if a.id == aid) for aid, tid in ids.items()}
        render_overlay(video, tr, tracks, vis, owner, d / "overlay.mp4", roles=roles)
    return {"episode": ep, "health": health, "cache": str(d)}


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--outcome", default="escaped")
    ap.add_argument("--manifest", default=str(ROOT / "data/manifest.csv"))
    ap.add_argument("--detector", default="yolo", choices=["yolo", "yolo-world", "blobs"])
    ap.add_argument("--prompts", default="cheetah,gazelle,lion,zebra,wildebeest,antelope,wild dog,impala")
    ap.add_argument("--points", default="auto", choices=["auto", "lk", "cotracker3", "tapnet"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    prompts = a.prompts.split(",") if a.detector == "yolo-world" else None
    jobs = []
    if a.video:
        jobs.append((a.clip or Path(a.video).stem, Path(a.video), a.outcome, None, None, {}))
    else:
        for r in csv.DictReader(open(a.manifest, encoding="utf-8")):
            if a.all or r["clip_id"] == a.clip:
                pid = int(r["predator_track"]) if r.get("predator_track") else None
                qid = int(r["prey_track"]) if r.get("prey_track") else None
                jobs.append((r["clip_id"], ROOT / r["file"], r["outcome"], pid, qid,
                             {"licence": r.get("licence"), "source_url": r.get("source_url"),
                              "sprint_start_frame": r.get("sprint_start_frame"), "decisive_frame": r.get("decisive_frame")}))
    for cid, vid, outc, pid, qid, meta in jobs:
        res = process(cid, vid, outc, a.detector, a.points, prompts, pid, qid, a.force, meta=meta)
        print(cid, res["health"], "->", res["cache"])


if __name__ == "__main__":  # pragma: no cover
    main()
