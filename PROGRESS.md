# PROGRESS

All nine phases were built in one pass (2026-10-19). Each note gives the phase's check and what happened.

| Phase | Check | Result |
|---|---|---|
| 1 Setup | pytest green | ✅ 23 tests pass (`pytest -q`) |
| 2 Ingest | 5 clips normalised | ⏳ code ready (`ingest/normalize.py`, ffmpeg via imageio-ffmpeg). No licensed hunting clips in the repo yet: add them to `data/raw/sources.csv` |
| 3 Perception | overlay video you watch | ✅ on rendered chases: `data/cache/eval_render_0/overlay.mp4` (blobs + LK). YOLO / YOLO-World / CoTracker3 / TAPNet backends are wired but not yet run here (no GPU) |
| 4 Camera compensation | static background ≈ 0 speed | ✅ 0.015 BL/s after compensation vs 2.21 BL/s raw camera motion (eval 6) |
| 5 Event layer | sprint starts within 0.3 s | ✅ tracks: 98% sprint / 97% flee; rendered video: 83% sprint / 100% flee (eval 7 passes after tracklet stitching) |
| 6 Simulator | planted rules → same schema | ✅ 6 rules (`sim/chase.py`) plus a planted keeper model for football shots |
| 7 Explainer | evidence JSON | ✅ LightGBM + TreeSHAP, timing model, similar-clip evidence, counterfactual |
| 8 Narrator + football adapter | grounded 2–3 sentences | ✅ 0 unsupported numbers in 1,200 narrations; fact-check agent rejected 0 live lines |
| 9 Demo + deploy | web page + ACA | ✅ app runs locally (live match + autopsy lab + evals). `infra/main.bicep` + `deploy.sh` written; **not yet deployed** (needs an Azure subscription) |

## Decisions and findings
- **Reported factors rule.** Secondary factors are shown only if they carry at least 25% of the top factor's weight and more than 0.1 log-odds. Ranking near-zero contributions is noise: with that rule, top-2 stability goes from 67% to 80% (top-1 alone: 92%).
- **Onset detection** walks back from the threshold crossing to the speed trough. This removed a constant ~0.2 s late bias: sprint starts within 0.3 s went from 72% to 98% on tracks.
- **Football shots** recover the planted cause less often (about 69% in eval 1) than chases (about 100%). The keeper model is probabilistic, so a shot's "true cause" is itself noisy. In-match shots scored 87%.
- **Track health:** LK body dots rarely survive a whole clip (the bodies rotate and deform), and background dots leave the frame during pans. Positions fall back to box centres, and the camera fit re-detects fresh background corners per frame pair. CoTracker3/TAPNet should do better on real footage.
- The live path is deterministic templates (about 1 ms per event); the LLM is used for recaps, deep-dive autopsies and extra languages.

## Real footage test (2026-10-19)
Two Wikimedia Commons clips were uploaded through the app (`data/raw/commons/`): *Cat chasing a pigeon* (CC0) and *Rattlesnake vs kangaroo rat* (CC BY 4.0, srep40412).
- COCO YOLO labels the cat as "dog" and the kangaroo rat as "bear", splits each into fragments, and never detects the pigeon (about 8 px) or the camouflaged snake.
- Added: tracklet stitching, a check that tracks really move, size-gated association, a class-agnostic motion detector with frame stabilisation, and a hybrid YOLO + motion mode.
- Result: the cat and the kangaroo rat are now tracked cleanly, but the less visible animal (pigeon, snake) is still missed, so neither clip yields a valid predator-prey pair yet.
- **YOLO-World follow-up (prompts "cat, pigeon"; yolov8m-worldv2 at 1280 px):** detects the cat (conf 0.8+) and the pigeon (0.3–0.9). With label-aware stitching across gaps up to 2 s, the cat became one track and the pigeon one track, and roles came out right (cat = attacker, pigeon = defender).
  - Event timing on this clip is still unreliable. The cat mostly stalks, and the smoothed speed shows no sustained sprint, only a single tracking spike, so no sprint start is detected and the features are not trustworthy.
  - The chase model is trained on cheetah-like simulated chases, so a stalk-and-pounce on a bird that flies off is out of its domain.
  - CPU speed: about 5 s per frame, which is impractical for a 17 s clip; a GPU is needed.
- **Snake clip (night infrared):** YOLO-World finds nothing with any prompt (snake, rattlesnake, rat, kangaroo rat, rodent; generic "animal" peaks at 0.26 confidence).
- Detector changes kept: the attacker's commit is now the onset of its FINAL sustained sprint (as the plan defines). Evals unchanged: 8 of 8 pass.
- Conclusion: real footage needs either click-to-mark seeding (the user marks predator and prey once, then point tracking follows them), YOLO-World prompts, or a GPU tracker (CoTracker3/TAPNet).

## Football video + TAPNet (2026-10-22)
- **Eval 11 (real football footage, 7 Commons clips, 9 audited frames): FAIL.** Players 73% recall, 12% false on-pitch, ball 20%, median track 1.0 s, team purity 89%, no homography. Full analysis and the fine-tuning plan: `docs/FOOTBALL_VIDEO_EVAL.md`.
- **TAPNet (BootsTAPIR) installed and wired in:** `points="auto"` picks TAPNet when its checkpoint plus a GPU are present, else LK. CPU timing: 1,125 s for a 5 s clip. On a rendered chase it is not more accurate than LK (predator 1.18 vs 0.93 BL/s MAE; prey 3.40 vs 3.44), but body dots survive longer (25% vs 0%) (`python -m evals.tracker_compare`).

## Football video v2 (2026-10-22, CPU)
- Ball 20% → **60%** recall (tiled detection + Kalman). Track coverage 83–96% on 6 of 7 clips (BoT-SORT re-ID, 30 fps). Player recall and false positives are unchanged: they need the fine-tuned detector and pitch model.
- **Keeper reaction time from real video** (TAPNet): strike found automatically at 4.50 s. TAPNet keeps 95% of keeper dots vs LK's 38%, but its onset is 0.19 s late (n = 1).
- **GPU training kit** ready: `notebooks/train_football_gpu.ipynb` (Kaggle/Colab) → `training/train_football.py` → `checkpoints/football_*.pt`, which the pipeline picks up automatically.
- **Free LLM** without Azure: GitHub Models (`GITHUB_MODELS_TOKEN`, a fine-grained PAT with Models: read).

## Next
0. Football video: fine-tune a player/keeper/referee/ball detector, a ball pipeline and a pitch-keypoint model (see docs/FOOTBALL_VIDEO_EVAL.md), then re-run eval 11.
1. Add 40–100 licensed hunting clips; run `ingest.normalize` then `perception.run --detector yolo-world --points cotracker3` on a GPU.
2. Collect human-agreement responses (eval 10): `data/evals/human_survey.json` → `data/evals/human_responses.csv`.
3. Configure Azure OpenAI (`.env`), then deploy with `infra/deploy.sh`.
4. Swap in the organisers' synthetic feed via `football/adapter.py` `FIELD_MAP` once its format is confirmed.
