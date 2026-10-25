# Hunt Autopsy — explainable match intelligence

**Inside the Game: Developer Hackathon (Microsoft × Premier League)** entry.

Most sports AI answers a question you ask it. Hunt Autopsy works the other way round. It starts from a **known outcome** (caught or escaped, goal or saved) and works **backwards** to the moments and movements that produced it. For example: *"Saved. The biggest factor: keeper reaction time of 0.27 s. When it is at or under 0.44 s, the result was saved in 652 of 737 similar shots."*

The same explainer reads cheetah chases, simulated chases and synthetic football. All three are reduced to **one event schema** first, so the method can be validated on chases with *planted* causes and then applied to the pitch.

```
 hunting video ─┐  YOLO+ByteTrack · TAPNet/CoTracker3/LK dots · camera compensation
 simulated chase├──► events/schema.py ──► explainer (LightGBM + TreeSHAP, timing model) ──► evidence JSON
 synthetic match┘     (one event layer)          │                                            │
   football/synth.py ─ adapter ─► live intel ────┴──► agents: moments · autopsy · personalizer · narrator ⇄ fact-check · overlay · recap
```

## What it does (mapped to the brief)

| Brief stage | Where it lives |
|---|---|
| **(a) Ingest** synthetic events as they happen | `football/synth.py` (feed generator), `football/adapter.py` (configurable field map), SSE stream `/api/matches/{id}/stream` |
| **(b) Interpret** into stats, patterns, context | `football/intel.py`: possession, pass accuracy, **pass difficulty** and passes-over-expected, xG, **shot/ball speed**, PPDA, pressures, momentum, **control vs chaos**, tempo/rhythm, pressing shifts, Poisson win probability; player **speed and distance tags** fire at thresholds |
| **(c) Explain why** a moment matters | win-probability swing per goal; flow/pressure shift evidence; **shot autopsy**: why the keeper couldn't reach it (or could), using the same explainer validated on chases |
| **(d) Render** alongside the match | timed, machine-readable **overlay cues** (`football/overlays.py`, JSON schema at `/api/overlay-schema`): lower thirds, player tags, autopsy cards, momentum and flow meters, scorebug |
| **(e) Personalize** | analyst / casual / player-focus / young-fan personas, favourite club framing, favourite player, 5 template languages + any language through Azure OpenAI. The UI shows **two viewers side by side from one stream** |
| Narratives and recaps | live lines, half-time and full-time recaps per persona and language (`agents/recap.py`) |
| Multi-language | en · es · fr · pt · de templates (offline, grounded); any other language through the LLM narrator, still fact-checked |
| Data innovation | synthetic match feed with 360-style **freeze frames + tracking snippets** for every on-target shot, and a **planted keeper model** so explanations can be scored against ground truth |
| Auto-eventing | `perception/football_autoevent.py`: ball/player tracks → passes, turnovers and ball speed (unit-tested); the video path uses YOLO and a pitch homography |

## Agentic design

The live pipeline is a team of small agents sharing a blackboard (`agents/orchestrator.py`). Each agent's decision appears in the UI's **Agent trace**:

`ingest → stats → moments → autopsy → personalizer → narrator ⇄ fact-check → overlay` (+ `recap`)

- **Generator/critic loop.** The narrator agent (Azure OpenAI) drafts copy. The fact-check agent runs `narrate/grounding.py` and rejects any number that is not in the evidence JSON, naming the offending numbers back to the narrator. After two failed drafts the system falls back to deterministic templates, which are grounded by construction.
- **Per-viewer fan-out.** One event goes through the pipeline once, and the personalizer routes moments to N viewer profiles.
- **Latency-aware LLM use.** The live path is deterministic (about 1 ms per event on CPU). The LLM is used where latency allows: recaps, deep-dive autopsies, and languages without a template bank.

## Microsoft and Azure technologies
- **Azure OpenAI** narrator, recap and translation (`narrate/llm.py`). Supports keys or **Entra ID managed identity** (`DefaultAzureCredential`).
- **Azure Container Apps** deployment, plus Azure Container Registry, Log Analytics, a user-assigned managed identity and role assignments (`infra/main.bicep`, `infra/deploy.sh`).
- Built so it can move to the **Microsoft Agent Framework** (the successor to Semantic Kernel and AutoGen): each agent is a single-purpose callable over a shared board, which maps directly onto its sequential/handoff workflows.

## Quickstart (CPU, about 2 minutes)

```bash
pip install uv
uv venv && uv pip install -e ".[dev,video]"
uv run uvicorn app.main:app            # http://localhost:8000
uv run pytest -q                       # unit + API tests
uv run python -m evals.run_all         # the 10 evals, one table
```
**LLM narrator options**, picked automatically in this order (`narrate/llm.py`):
1. **Azure OpenAI / Microsoft Foundry** deployment: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT` (key or managed identity).
2. **GitHub Models** (free, rate-limited; no Azure needed): create a fine-grained GitHub token with the *Models: read* permission and set `GITHUB_MODELS_TOKEN`. The default model is `openai/gpt-4.1-mini`.
3. Any OpenAI-compatible endpoint: `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`).

Without any of them, everything runs offline on grounded templates.

**Real hunting clips** (optional, needs a GPU for YOLO/CoTracker3/TAPNet):
```bash
uv pip install -e ".[vision]"
# list clips + labels in data/raw/sources.csv (see sources.example.csv), then:
uv run python -m ingest.normalize                     # trim, 30 fps, data/manifest.csv
uv run python -m perception.run --all --detector yolo-world --points cotracker3
```
Every model output is cached under `data/cache/<clip_id>/`. Episodes land in `data/episodes/` and appear in the Autopsy lab as "Real clips".

## Analysing your own video (judges included)

Open the app → **Autopsy lab** → **Analyse your own clip**:
1. Drop a video of **one** attack (a chase or a shot), ideally 5–15 s. Use the *Trim from / to* boxes for longer files.
2. Optionally type the **animals in the clip** (e.g. `cat, pigeon`). This switches detection to YOLO-World (open vocabulary), which finds animals COCO YOLO misses. It is slow on CPU (about 5 s per frame), so use a GPU for anything longer than a few seconds.
3. Pick the **known outcome** (caught / escaped). The system works backwards from it; it does not guess the result.
4. Press **Analyse clip**. The server normalises it to 30 fps, runs YOLO + ByteTrack (falling back to all COCO classes when wild animals are mislabelled), tracks body and background dots, removes camera motion, detects sprint / flee / turn events, then explains the outcome. You get the overlay video (boxes and dots), the timeline with the decisive moment, the top factors and the narration.

API equivalent:
```bash
curl -F file=@chase.mp4 -F outcome=caught http://localhost:8000/api/clips     # -> {"id": "upload-…"}
curl http://localhost:8000/api/clips/upload-…                                  # poll until status == done
```
Limits: on CPU (`yolo11n` + LK dots) expect a few minutes per 10 s clip; the rendered test clip took about 2.5 minutes. The YOLO path has not yet been run on real footage. Both animals must be visible. Uploaded clips are explained by the chase model trained on simulated chases until there are enough labelled real clips. Set `HUNT_YOLO_MODEL=yolo11m.pt` on a GPU for better detection.

**Footage.** Openly licensed full chases with clear outcomes are rare: `data/raw/candidates_commons.csv` lists 16 Wikimedia Commons clips checked for licence and length. Big-cat chases on Pexels and Pixabay are mostly walking or feeding shots. Own footage (for example a dog chasing a ball or a cat stalking a toy) avoids licence questions.

## Football video: models and free GPU training
`perception/football_video.py` runs players, ball and pitch on video:
- **Tracking:** BoT-SORT with appearance re-ID and camera-motion compensation at full frame rate (`configs/football_botsort.yaml`), reset at shot cuts.
- **Ball:** tiled detection plus a Kalman tracker.
- **Pitch:** a 32-keypoint homography, so positions are in metres.
- **Keeper reaction time:** `perception/keeper_reaction.py` measures it from real video with TAPNet point tracks.

It uses fine-tuned football models when `checkpoints/football_*.pt` exist, and COCO YOLO otherwise. To train them for free, open `notebooks/train_football_gpu.ipynb` on **Kaggle** (GPU T4, about 30 h/week free) or **Google Colab** (free T4), add a free `ROBOFLOW_API_KEY` as a secret, and run all cells. It fine-tunes the three models (`training/train_football.py`), re-runs eval 11 and packs `checkpoints/` for download. Eval clips are fetched with `python scripts/fetch_football_clips.py`.

## Evals (`python -m evals.run_all`)

| # | Eval | How | Bar |
|---|---|---|---|
| 1 | Planted-cause recovery | 6 planted chase rules + planted keeper model; top factor must be the planted cause | ≥ 80% |
| 2 | Deletion faithfulness | remove the top feature vs a random one | top drop > 2× random |
| 3 | Outcome prediction | leave-one-clip-out on real clips (5-fold CV on synthetic until ≥ 10 real) | beats majority |
| 4 | Stability | 5 seeds × bootstrap × edge trims | same top-2 in ≥ 4/5 |
| 5 | Flip invariance | mirror clips | unchanged |
| 6 | Camera compensation | background dots on rendered pan+zoom clips | < 0.1 BL/s |
| 7 | Event timing | detected vs true sprint start and flee | within 0.3 s on most |
| 8 | Track health | dots alive at end; ID births | logged |
| 9 | Grounded narration | every number in text is in the evidence | zero unsupported |
| 10 | Human agreement | real vs shuffled explanation survey (`data/evals/human_survey.json`) | > 50% |
| 11 | Football video (real footage) | 7 openly licensed clips; per-stage audit (players, ball, tracking, teams, homography, events) | see `docs/FOOTBALL_VIDEO_EVAL.md` |

Results are written to `data/evals/latest.json` and shown in the app's **Evals** tab. See `PROGRESS.md` for the latest numbers.

**Honesty note:** the model finds **explanatory factors with evidence**, not proof of causation. Evals 1–2 (planted causes, deletion) and the per-clip counterfactual are what back the stronger claim.

## Open-source building blocks we researched and use

| Repo / model | Role here | Status |
|---|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) YOLO11 + ByteTrack/BoT-SORT, **YOLO-World** | boxes + stable IDs; open-vocabulary prompts ("cheetah", "gazelle") when COCO mislabels animals | `perception/detect_track.py` |
| [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) (TAPIR / BootsTAPIR / TAPNext) | dense point tracks with occlusion | `perception/points.py` backend `tapnet` |
| [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) (CoTracker3) | plan-B point tracker via `torch.hub` | backend `cotracker3` |
| OpenCV pyramidal LK | CPU point tracker for tests and evals | backend `lk` (default) |
| [roboflow/sports](https://github.com/roboflow/sports) + [supervision](https://github.com/roboflow/supervision) | pitch-keypoint homography, team classification by jersey colour, ball tracking | design of `perception/football_autoevent.py` |
| [DeepLabCut SuperAnimal-Quadruped](https://github.com/DeepLabCut/DeepLabCut) / YOLO-pose | heading from named joints; vision-cone screening | `perception/keypoints.py` |
| [SAM 2](https://github.com/facebookresearch/sam2) | masks so query points land on the body, not the grass | `seed_queries(masks=...)` hook |
| [LightGBM](https://github.com/microsoft/LightGBM) (Microsoft) | outcome model; **TreeSHAP via `pred_contrib`** (no extra SHAP dependency) | `explain/model.py` |
| [kloppy](https://github.com/PySport/kloppy) | standard loaders for event/tracking feeds; easiest route if the organisers' feed is a known provider format | adapter target (recommended) |
| [socceraction](https://github.com/ML-KULeuven/socceraction) (SPADL, VAEP, xT) | action values: a richer "why it matters" score than our location xG | recommended next step |
| [mplsoccer](https://github.com/andrewRowlands/mplsoccer), [databallpy](https://github.com/Alek050/databallpy), [floodlight](https://github.com/floodlight-sports/floodlight) | recap graphics; event+tracking sync and physical metrics | recommended |
| [lifelines](https://github.com/CamDavidsonPilon/lifelines) | Cox time-varying alternative to our discrete-time timing model | optional extra `survival` |
| Microsoft Agent Framework | orchestration upgrade path for `agents/` | recommended |
| LocoTrack, Animal Kingdom, MammalNet | alternative tracker; animal-behaviour datasets (check licences before demoing clips) | references |

## Data and licences
- All football data is **synthetic** and generated by `football/synth.py`. Clubs and players are fictional, and no real Premier League data is used.
- Hunting clips: use openly licensed footage (Wikimedia Commons CC, Pexels, Pixabay) and record the licence per clip in `sources.csv`. Do not commit documentary footage.
- Ultralytics is AGPL-3.0, so this repo is AGPL-3.0. TAPNet weights are Apache-2.0.

## Repo map
```
events/      schema.py (the one event schema) · features.py (12 named features, hunt ↔ football twins) · detect.py
sim/         chase.py (planted-rule simulator) · render.py (pan/zoom video with known ground truth)
explain/     model.py (logreg | LightGBM + exact contributions) · timeline.py (decisive moment) · evidence.py
football/    synth.py · adapter.py · intel.py · overlays.py
narrate/     narrator.py · grounding.py · templates.py (5 languages) · personalize.py · llm.py (Azure OpenAI)
agents/      orchestrator.py (live agents) · recap.py · models.py (trained, cached explainers)
perception/  detect_track.py · points.py · camera.py · to_episode.py · overlay.py · keypoints.py · football_autoevent.py · run.py
ingest/      normalize.py (ffmpeg → 30 fps + manifest)
app/         main.py (FastAPI + SSE) · static/ (broadcast UI)
evals/       run_all.py
infra/       main.bicep · deploy.sh (Azure Container Apps)
```
