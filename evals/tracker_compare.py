"""Speed-trace accuracy of the point trackers (LK vs TAPNet/BootsTAPIR) on rendered clips with known world motion.

    python -m evals.tracker_compare --clips 1 --points lk tapnet

Speed is what the explainer consumes (closing speed, top speeds, onsets), so we score the per-actor speed trace
recovered from the dots (after camera compensation, in body lengths/s) against ground truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def speed_error(clip: str, points: str) -> dict:
    from events.detect import smooth
    from evals.run_all import sims
    from perception.run import process
    from sim.render import render
    i = int(clip.split("_")[-1])
    ep = sims("compound", 12, 9)[i]
    vid = ROOT / "data" / "raw" / f"{clip}.mp4"
    if not vid.exists():
        render(ep, vid, seed=i)
    t = time.time()
    res = process(clip, vid, ep.outcome_label, detector="blobs", points=points, overlay=False, publish=False)
    secs = time.time() - t
    out = {"points": points, "seconds": round(secs, 1), "health": res["health"]}
    e2 = res["episode"]
    for actor in ("pred", "prey"):
        g, d = ep.track(actor).array(), e2.track(actor).array()
        tt = d[:, 0]
        gx = np.c_[np.interp(tt, g[:, 0], g[:, 1]), np.interp(tt, g[:, 0], g[:, 2])]
        sg = np.linalg.norm(np.gradient(smooth(gx, 30), tt, axis=0), axis=1)
        sd = np.linalg.norm(np.gradient(smooth(d[:, 1:], 30), tt, axis=0), axis=1)
        out[actor] = {"speed_mae_bl_s": round(float(np.mean(np.abs(sg - sd))), 3),
                      "peak_err_bl_s": round(float(abs(sg.max() - sd.max())), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=1)
    ap.add_argument("--points", nargs="+", default=["lk", "tapnet"])
    a = ap.parse_args()
    rows = []
    for k in range(a.clips):
        for p in a.points:
            r = {"clip": f"eval_render_{k}", **speed_error(f"eval_render_{k}", p)}
            print(json.dumps(r), flush=True)
            rows.append(r)
    (ROOT / "data" / "evals").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "evals" / "tracker_compare.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
