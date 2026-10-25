"""Fine-tune the three football perception models on a GPU (Kaggle / Colab / Azure ML), then drop them in checkpoints/.

    pip install ultralytics roboflow "git+https://github.com/roboflow/sports.git"
    export ROBOFLOW_API_KEY=...          # free Roboflow account -> Settings -> API key
    python -m training.train_football --what players ball pitch

Datasets (Roboflow Universe, CC BY 4.0, from the roboflow/sports project):
  players  roboflow-jvuqo / football-players-detection-3zvbc   classes: ball, goalkeeper, player, referee
  ball     roboflow-jvuqo / football-ball-detection-rejhg      class: ball (high-res, small objects)
  pitch    roboflow-jvuqo / football-field-detection-f07vi     32 pitch keypoints (pose)

Outputs (consumed automatically by perception/football_video.py):
  checkpoints/football_players.pt, checkpoints/football_ball.pt, checkpoints/football_pitch.pt,
  checkpoints/football_pitch_keypoints.json  (the 32 template vertices in metres, from roboflow/sports)
These are fine-tunes of pretrained YOLO11 weights: no training from scratch is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
WORKSPACE = "roboflow-jvuqo"
DATASETS = {
    "players": ("football-players-detection-3zvbc", "yolov8"),
    "ball": ("football-ball-detection-rejhg", "yolov8"),
    "pitch": ("football-field-detection-f07vi", "yolov8"),
}
# model, imgsz, epochs, batch, extra args
RECIPES = {
    "players": ("yolo11m.pt", 1280, 60, 6, {"mosaic": 1.0, "close_mosaic": 10}),
    "ball": ("yolo11s.pt", 1280, 80, 8, {"mosaic": 1.0, "scale": 0.3}),
    "pitch": ("yolo11m-pose.pt", 640, 120, 16, {"mosaic": 0.0, "fliplr": 0.0}),  # no flip: keypoint ids are not mirrored
}


def download(name: str, version: int | None = None) -> Path:
    from roboflow import Roboflow
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY (free Roboflow account -> Settings -> API key)")
    proj_id, fmt = DATASETS[name]
    project = Roboflow(api_key=key).workspace(WORKSPACE).project(proj_id)
    if version is None:
        version = max(int(v.version) for v in project.versions())
    ds = project.version(version).download(fmt, location=str(ROOT / "data" / "datasets" / name))
    return Path(ds.location) / "data.yaml"


def train(name: str, epochs: int | None = None, device: str | int = 0) -> Path:
    from ultralytics import YOLO
    data = download(name)
    model, imgsz, ep, batch, extra = RECIPES[name]
    r = YOLO(model).train(data=str(data), imgsz=imgsz, epochs=epochs or ep, batch=batch, device=device,
                          project=str(ROOT / "runs" / "football"), name=name, exist_ok=True, patience=20, **extra)
    best = Path(r.save_dir) / "weights" / "best.pt"
    CKPT.mkdir(exist_ok=True)
    dst = CKPT / f"football_{name}.pt"
    shutil.copy(best, dst)
    print("saved", dst)
    return dst


def save_pitch_template():
    """The 32 vertices of the roboflow/sports SoccerPitchConfiguration (cm) -> metres, in keypoint-id order."""
    from sports.configs.soccer import SoccerPitchConfiguration
    cfg = SoccerPitchConfiguration()
    verts = [[x / 100.0, y / 100.0] for x, y in cfg.vertices]
    CKPT.mkdir(exist_ok=True)
    (CKPT / "football_pitch_keypoints.json").write_text(json.dumps(
        {"source": "roboflow/sports SoccerPitchConfiguration", "units": "m", "length_m": cfg.length / 100, "width_m": cfg.width / 100,
         "vertices_m": verts}, indent=1))
    print("saved", CKPT / "football_pitch_keypoints.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["players", "ball", "pitch"], choices=list(RECIPES))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    for w in a.what:
        train(w, a.epochs, a.device)
        if w == "pitch":
            save_pitch_template()


if __name__ == "__main__":
    main()
