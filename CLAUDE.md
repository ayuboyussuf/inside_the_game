# Hunt Autopsy
Goal: explain WHY an attack succeeded or failed, working backwards from a known outcome.
Architecture: video -> perception (YOLO tracks, TAPNet/CoTracker3/LK points) -> camera compensation -> event layer -> explainer -> narrator.
Football: synthetic match events -> football/adapter.py -> same event layer -> live intelligence, explainer (shot autopsy), narrator, overlays.
Rule: the explainer and narrator only read the event schema in events/schema.py, never raw video.
Rule: cache every model output under data/cache/<clip_id>/; never re-run vision models if the cache exists.
Rule: distances in predator body lengths, time in seconds (football episodes use metres; the unit is declared on the Episode).
Rule: the narrator may only state numbers and timestamps present in the evidence JSON (narrate/grounding.py enforces this).
Work one phase at a time; finish each phase with its check and a short note in PROGRESS.md.
Run the eval suite (evals/run_all.py) after any change to events/ or explain/.

## Commands
- Install: `uv venv && uv pip install -e ".[dev,video]"` (add `vision` for YOLO/CoTracker3 on a GPU box)
- Tests: `uv run pytest -q`
- Evals: `uv run python -m evals.run_all`
- App: `uv run uvicorn app.main:app --reload` then open http://localhost:8000
- Ingest: `uv run python -m ingest.normalize --sources data/raw/sources.csv`
- Perception: `uv run python -m perception.run --clip <clip_id>`

## Layout
- events/   schema, track->event detection, shared feature table
- sim/      planted-rule chase simulator (+ video renderer for perception tests)
- football/ synthetic match generator, adapter, live intelligence, moments, overlays
- explain/  outcome model + per-clip contributions (TreeSHAP/linear SHAP), outcome-probability timeline, evidence JSON
- narrate/  LLM + template narrator, grounding checker, personas, languages
- agents/   orchestrator and agents (stats, moments, autopsy, narrator, fact-check, personalizer, overlay)
- perception/ detection+tracking, point tracking, camera compensation, episode builder, overlay video
- ingest/   ffmpeg normalisation + manifest
- app/      FastAPI + static frontend
- evals/    the 10 evals behind run_all.py
