# Eval 11: real football footage — where the perception stack fails, and what to fix or train

**Run:** `python -m evals.football_video` (harness), then `python -m evals.run_all` (scores it with the audit).
**Data:** 7 openly licensed clips from Wikimedia Commons (`data/raw/football/SOURCES.json`). No Premier League footage is used: it is copyrighted, and the hackathon forbids redistributing real PL data.
**Stack under test:** YOLO11s (COCO) at 1280 px + ByteTrack at 10 fps, an HSV pitch mask, jersey-colour k-means, and `perception/football_autoevent.py` in image space.
**Ground truth:** 9 frames audited by hand (`data/evals/football_video/audit.json`), plus per-stage statistics on every sampled frame.

## Results

| Stage | Measured | Bar | Status |
|---|---|---|---|
| Player detection recall | **73%** (95 of 131 visible people) | ≥ 90% | FAIL |
| People wrongly counted on the pitch | **12%** (photographers, stewards) | ≤ 5% | FAIL |
| Ball detection recall | **20%** (1 of 5 frames with a visible ball, and that one was stationary on the penalty spot) | ≥ 70% | FAIL |
| Track continuity | median track **1.0 s**; 2–11 new IDs per second | ≥ 3 s | FAIL |
| Team clustering purity | **89%** (silhouette 0.50–0.67) | ≥ 90% | FAIL (borderline) |
| Image → pitch homography | **0 of 7** clips (no pitch-keypoint model) | all | FAIL |
| Auto-eventing | 0 events on 6 clips; **5 spurious "passes"** on a penalty | sensible | FAIL |

### Per clip
| Clip | View | Main failure |
|---|---|---|
| everton_oxford_highlights | broadcast-style wide camera (closest to PL TV) | best case: 38 of 45 players found; ball in the air missed; tracks fragment (0.9 s) |
| fra_ger_open_play | fan stand, players about 30 px tall | far players and keeper missed; 6 touchline staff counted as players |
| fra_ger_kroos_penalty | near the goal | players fine; ball found only while still; photographers inside the pitch mask |
| keller_saves_henry_pk | TV close-up | keeper in dark kit against the net missed; flying ball missed |
| valencia_wc2022_goal | high TV camera | players in a wall merge; crouching player missed; visible ball missed |
| latvia_gibraltar | phone, smoke | nothing detected (smoke, and a green kit on a green pitch) |
| amateur_shot_topdown | drone, top-down | 1 of 10 players: COCO "person" is trained on side views |

## v2: what changed after the first fixes (2026-10-22, CPU, no fine-tuned models yet)
`perception/football_video.py` adds tiled (SAHI-style) ball detection with a Kalman tracker, BoT-SORT with appearance re-ID and camera-motion compensation at full frame rate, and a stronger cut detector.

| Stage | v1 | v2 | Note |
|---|---|---|---|
| Ball recall (audited frames, within 15 px) | 20% (1 of 5) | **60% (3 of 5)** | tiling finds the 6 px ball; the Kalman filter keeps a position in 78–100% of frames on penalty and broadcast clips |
| Track coverage (on-pitch detections in tracks ≥ 2 s) | not measured | **83–96%** on 6 of 7 clips; 7% on the smoke clip; mean 76% | median track length was a misleading metric (many short fragments where players bunch) |
| Player recall / false on-pitch | 73% / 12% | unchanged | same COCO detector. A tighter colour mask lost fogged far-side players, and "grass under the feet" fails because touchline staff often stand on grass, so v2 keeps the hull mask. **Only the pitch-keypoint model fixes this** (exact touchline from the homography). |
| Cuts | 0 | 0 | none of the 8-second windows contains a cut, so the new detector is still unvalidated on real edits |
| Homography | 0 of 7 | 0 of 7 | needs the trained pitch model |

Caveat: v2's on-pitch weighting for track coverage came from that tighter (reverted) mask, so the coverage figure is approximate.

### Keeper reaction time from video with TAPNet (`perception/keeper_reaction.py`, Keller penalty save)
The strike is found automatically at **4.50 s** from the camera-compensated ball track (matches the frames). Kicker and keeper are picked automatically.

| Tracker | Dive onset | Reaction time | Keeper dots alive at end | Error vs manual (onset ≈ 4.48 s) |
|---|---|---|---|---|
| LK | 4.43 s | −0.07 s | 38% | 0.05 s early |
| TAPNet (BootsTAPIR, CPU) | 4.67 s | +0.17 s | **95%** | 0.19 s late |

TAPNet keeps the keeper's body tracked through the dive (95% vs 38% of dots), which is what makes the measurement trustworthy on harder clips. Its onset fires late because its speed trace rises more gradually, so the onset threshold needs calibrating on more labelled penalties (n = 1 here; not tuned).

## Why it fails (root causes)
1. **The ball is a 4–10 px, motion-blurred object.** COCO "sports ball" is trained on close-up balls, and single-frame detection cannot recover a blurred one.
2. **COCO "person" is the wrong class.** It has no notion of player / goalkeeper / referee / staff, so touchline staff become players and keepers against the net are missed. It also fails on non-standard viewpoints (top-down, extreme distance).
3. **No pitch model.** The convex-hull green mask swallows the touchline, and without keypoints there is no homography, so no metres, km/h, pass lengths or pitch coordinates. Every downstream metric (speed tags, pass difficulty, xG) needs pitch coordinates.
4. **Tracking at 10 fps with a default ByteTrack and no appearance re-ID.** IDs break at every occlusion, and players in similar kits swap.
5. **Image-space auto-eventing.** "Possession" by nearest pixel distance fires on a static penalty-spot ball beside players, producing spurious passes.
6. **Broadcast artefacts.** Replays, cuts and graphics are not handled (the histogram cut detector found 0 cuts in a highlights package that has edits).

## Fixes in priority order (with where training is needed)

| # | Fix | Training? | Data / model |
|---|---|---|---|
| 1 | **Fine-tune a football detector** with classes `player, goalkeeper, referee, ball` at 1280 px (YOLO11m) | **Yes, fine-tune** (not pre-train) | Roboflow `football-players-detection` (the roboflow/sports dataset); SoccerNet-Tracking if the licence allows; plus ~300 hand-labelled frames from the target camera |
| 2 | **Dedicated ball pipeline**: tiled/SAHI inference + high-res ball detector + temporal filter (Kalman, interpolation over ≤ 0.5 s gaps) | **Yes, fine-tune** a ball-only model | Roboflow `football-ball-detection`; synthetic blur augmentation |
| 3 | **Pitch keypoint model** (32 keypoints, YOLO-pose) → homography per frame → metres and km/h, and an exact on-pitch filter | **Yes, fine-tune** | Roboflow `football-field-detection` keypoints |
| 4 | **Tracking**: run at full frame rate; BoT-SORT with appearance re-ID; longer track buffer; reset at cuts | Optional fine-tune of re-ID | SoccerNet-ReID; jersey-number OCR (PARSeq) for persistent identity |
| 5 | **Team assignment**: SigLIP crop embeddings → UMAP → k-means (the roboflow/sports TeamClassifier); goalkeepers assigned by position | No (pretrained embeddings) | — |
| 6 | **Shot boundaries, replays and graphics**: TransNetV2 for cuts; mask static overlays; keep only the main wide camera | No (pretrained) | — |
| 7 | **Auto-eventing in pitch coordinates** with a ball-speed gate (a pass needs ball travel > 2 m at > 10 km/h) and possession by ball–foot distance in metres | No (rules) | needs fixes 2 and 3 |
| 8 | **TAPNet where it pays**: keeper and shooter body dots for *reaction delay* (first dive movement) and body orientation, inside detector boxes | No (pretrained BootsTAPIR / TAPNext) | GPU required |

**What does *not* need training:** the explainer and narrator. The shot autopsy trains on synthetic shots with a planted keeper model (eval 1). Fitting it to real outcomes would need real event and freeze-frame data, which this hackathon does not allow for the Premier League. So keep the synthetic model and state it clearly.

## About TAPNet (why it is in the stack, and what we measured)
- Speeds and onsets come from the **point tracks**, not the boxes: each actor's position is the median of its visible dots (`perception/to_episode.py`). The box centre is only a fallback.
- `points="auto"` now selects **TAPNet (BootsTAPIR)** whenever its checkpoint is present and a GPU is available. On CPU it falls back to OpenCV LK, because TAPNet took **1,125 s for a 5 s clip** on this laptop, against seconds for LK.
- Speed-trace accuracy on a rendered chase with known motion (`python -m evals.tracker_compare`):

| Tracker | Predator speed MAE | Prey speed MAE | Body dots alive at end |
|---|---|---|---|
| LK | 0.93 BL/s | 3.44 BL/s | 0% |
| TAPNet (256 px) | 1.18 BL/s | 3.40 BL/s | 25% |

  TAPNet keeps dots alive far longer, but on a clean synthetic clip it is not more accurate. The large prey error comes from how positions are formed (detector boxes and dot fallback), not from the tracker. TAPNet's strengths (occlusion, deformation, long-range) matter on real broadcast footage, so retest at higher resolution on a GPU.
- **For football, TAPNet is not the main tracker.** Players and the ball are tracked by detection + ByteTrack. TAPNet adds value for fine motion inside a box: the keeper's first dive movement, shooter feints, and body orientation for the screened-keeper vision cone.

## Bottom line
Today the system is strong on the synthetic event layer (evals 1–9 pass). From real video it can only count players reliably on a broadcast-style wide shot. Items 1–3 above (football detector, ball pipeline, pitch keypoints) are the minimum to make auto-eventing from video usable. All three are **fine-tuning** jobs on public football datasets, not pre-training from scratch. Re-run this eval after each one.
