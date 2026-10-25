"""FastAPI app: live match intelligence (SSE), shot & hunt autopsies, evals, overlay schema.

Run: uvicorn app.main:app --reload   ->   http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.models import hunt_library, shot_autopsy
from agents.orchestrator import LiveMatch
from agents.recap import make_recap
from events.features import FEATURES, label_for
from explain.timeline import state_frame
from football.overlays import OverlayCue
from football.synth import CLUBS, generate_match
from narrate.llm import get_llm
from narrate.narrator import narrate_autopsy
from narrate.personalize import Profile
from narrate.templates import LANG_NAMES

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"
@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm, daemon=True).start()  # train/load explainers without blocking startup
    yield


app = FastAPI(title="Hunt Autopsy — Inside the Game", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

MATCHES: dict[str, dict] = {}
_ready = {"models": False, "error": None}


def _warm():
    try:
        shot_autopsy()
        hunt_library()
        _ready["models"] = True
    except Exception as e:  # pragma: no cover
        _ready["error"] = repr(e)


# --------------------------------------------------------------------- basics
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    llm = get_llm()
    return {"ok": True, "models_ready": _ready["models"], "error": _ready["error"],
            "llm": llm.provider if llm else None, "matches": len(MATCHES)}


@app.get("/api/config")
def config():
    return {"clubs": CLUBS, "languages": LANG_NAMES, "personas": {
        "analyst": "Analyst — every stat, factors and probabilities",
        "casual": "Casual fan — the big moments, short and vivid",
        "player_focus": "Player focus — follow one player all match",
        "kids": "Young fan — one friendly line per big moment"},
        "features": [{"name": f.name, "label": f.label, "hunt": f.hunt, "football": f.football, "football_label": label_for(f.name, "football")} for f in FEATURES]}


@app.get("/api/overlay-schema")
def overlay_schema():
    return OverlayCue.model_json_schema()


# --------------------------------------------------------------------- matches
class NewMatch(BaseModel):
    seed: int = 7
    home: str = "NBR"
    away: str = "RVA"


@app.post("/api/matches")
def new_match(req: NewMatch):
    if req.home == req.away:
        raise HTTPException(400, "pick two different clubs")
    feed = generate_match(req.seed, req.home, req.away)
    mid = feed["metadata"]["match_id"] + f"-{req.home}-{req.away}"
    feed["metadata"]["match_id"] = mid
    MATCHES[mid] = feed
    while len(MATCHES) > 24:
        MATCHES.pop(next(iter(MATCHES)))
    meta = feed["metadata"]
    return {"match_id": mid, "home": meta["home"], "away": meta["away"], "venue": meta["venue"],
            "competition": meta["competition"], "lineups": meta["lineups"], "n_events": len(feed["events"])}


def _feed(mid: str) -> dict:
    if mid not in MATCHES:
        raise HTTPException(404, "unknown match (create one with POST /api/matches)")
    return MATCHES[mid]


@app.get("/api/matches/{mid}/feed")
def match_feed(mid: str):
    """The raw synthetic provider feed (data innovation deliverable)."""
    return _feed(mid)


def _profiles(spec: str | None) -> dict[str, Profile]:
    if not spec:
        return {"a": Profile()}
    raw = json.loads(spec)
    return {k: Profile(**{kk: vv for kk, vv in v.items() if vv not in ("", None)}) for k, v in raw.items()}


def _compact(raw: dict) -> dict:
    keep = ("event_id", "timestamp", "type", "team_id", "player_id", "player_name", "location", "end_location", "outcome",
            "xg", "length_m", "difficulty", "shot_speed_kmh", "speed_kmh", "recipient_name", "card", "replacement_name", "period")
    return {k: raw[k] for k in keep if k in raw}


@app.get("/api/matches/{mid}/stream")
async def stream(mid: str, profiles: str | None = None, speed: float = 60.0, start: float = 0.0, trace: bool = True):
    """Server-sent events. `speed` = match seconds per wall-clock second. `profiles` = JSON {name: Profile}."""
    feed = _feed(mid)
    profs = _profiles(profiles)

    async def gen():
        live = LiveMatch(feed, profs)
        last_t = None
        n = 0
        while True:
            pkt = live.step()
            if pkt is None:
                break
            ev = pkt["event"]
            t = ev.get("timestamp", 0.0)
            if t < start:
                continue
            if last_t is not None and speed > 0:
                await asyncio.sleep(min(1.5, max(0.0, (t - last_t) / speed)))
            last_t = t
            n += 1
            has_out = any(o["lines"] or o["cues"] for o in pkt["outputs"].values())
            msg = {"event": _compact(ev), "outputs": pkt["outputs"]}
            if has_out or n % 8 == 0 or ev["type"] in ("goal", "shot", "half_time", "full_time"):
                msg["snapshot"] = pkt["snapshot"]
                msg["scorebug"] = pkt["scorebug"]
            if trace and pkt["trace"]:
                msg["trace"] = pkt["trace"]
            yield f"data: {json.dumps(msg, default=_np)}\n\n"
        yield f"data: {json.dumps({'done': True, 'snapshot': live.state.snapshot(), 'players': [live.state.player_summary(p) for p in live.state.player if p in live.state.player_team]}, default=_np)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _np(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _run_to(feed: dict, upto: float | None, prof: Profile) -> LiveMatch:
    live = LiveMatch(feed, {"p": prof})
    for raw in feed["events"]:
        if upto is not None and raw.get("timestamp", 0) > upto:
            break
        live.process(raw)
    return live


@app.get("/api/matches/{mid}/recap")
def recap(mid: str, persona: str = "analyst", language: str = "en", favorite_club: str | None = None,
          favorite_player: str | None = None, upto: float | None = None, llm: bool = True):
    prof = Profile(persona=persona, language=language, favorite_club=favorite_club or None, favorite_player=favorite_player or None)
    live = _run_to(_feed(mid), upto, prof)
    r = make_recap(live, prof, use_llm=llm)
    return {**r, "snapshot": live.state.snapshot(), "top_players": live.state.top_players(5)}


@app.get("/api/matches/{mid}/autopsy/{shot_id}")
def match_autopsy(mid: str, shot_id: str, persona: str = "analyst", language: str = "en", llm: bool = False):
    feed = _feed(mid)
    shot = next((e for e in feed["events"] if e.get("event_id") == shot_id), None)
    if shot is None:
        raise HTTPException(404, "shot not found")
    from football.adapter import shot_episode
    ep = shot_episode(shot)
    if ep is None:
        raise HTTPException(400, "not an on-target shot with tracking")
    a = shot_autopsy()
    ev = a.evidence(ep)
    return {"evidence": ev, "narration": narrate_autopsy(ev, persona, language, use_llm=llm),
            "true_causes": ep.meta.get("true_causes"), "episode": _episode_payload(ep, a)}


# --------------------------------------------------------------------- hunts
SCENARIOS = {"compound": "Single predator (reaction + cuts rule)", "pack": "Pack hunt (group help)",
             "screening": "Herd screening", "football": "Football shots (shooter vs keeper)", "real": "Real clips"}


@app.get("/api/hunts")
def hunts():
    if not _ready["models"]:
        _warm()
    lib = hunt_library()
    out = []
    for k, v in lib.items():
        out.append({"scenario": k, "title": SCENARIOS.get(k, k), "rule": v["rule"],
                    "importance": v["autopsy"].feature_importance()[:6],
                    "episodes": [{"id": e.episode_id, "outcome": e.outcome_label} for e in v["episodes"][:48]]})
    from football.adapter import shot_episodes
    from football.synth import generate_shots
    shots = shot_episodes(generate_shots(48, seed=777))
    _SHOTS.update({e.episode_id: e for e in shots})
    out.append({"scenario": "football", "title": SCENARIOS["football"], "rule": "planted keeper model (reaction, screening, angle, distance)",
                "importance": shot_autopsy().feature_importance()[:6],
                "episodes": [{"id": e.episode_id, "outcome": e.outcome_label} for e in shots]})
    return out


_SHOTS: dict = {}


def _episode_payload(ep, autopsy) -> dict:
    tracks = []
    for tr in ep.tracks:
        arr = tr.array()
        step = max(1, len(arr) // 180)
        role = next(a.role for a in ep.actors if a.id == tr.actor)
        tracks.append({"actor": tr.actor, "role": role, "t": arr[::step, 0].round(3).tolist(),
                       "x": arr[::step, 1].round(3).tolist(), "y": arr[::step, 2].round(3).tolist()})
    curve = autopsy.timeline.curve(ep) if autopsy.timeline is not None else None
    return {"episode_id": ep.episode_id, "domain": ep.domain, "outcome": ep.outcome_label, "distance_unit": ep.distance_unit,
            "tracks": tracks, "events": [e.model_dump() for e in ep.events if e.type not in ("track_start",)],
            "curve": curve.to_dict(orient="list") if curve is not None and len(curve) else None,
            "meta": {k: v for k, v in ep.meta.items() if k in ("rule", "rule_description", "true_causes", "decisive_t", "licence",
                                                                 "source_url", "video", "keeper_name", "shot")},
            "video": f"/media/{ep.episode_id}/overlay.mp4" if (ROOT / "data/cache" / ep.episode_id / "overlay.mp4").exists() else None}


@app.get("/api/hunts/{scenario}/{eid}")
def hunt(scenario: str, eid: str, persona: str = "analyst", language: str = "en", llm: bool = False):
    if scenario == "football":
        if eid not in _SHOTS:
            hunts()
        ep = _SHOTS.get(eid)
        a = shot_autopsy()
    else:
        lib = hunt_library()
        if scenario not in lib:
            raise HTTPException(404, "unknown scenario")
        ep = next((e for e in lib[scenario]["episodes"] if e.episode_id == eid), None)
        a = lib[scenario]["autopsy"]
    if ep is None:
        raise HTTPException(404, "unknown episode")
    ev = a.evidence(ep)
    return {"evidence": ev, "narration": narrate_autopsy(ev, persona, language, use_llm=llm),
            "episode": _episode_payload(ep, a), "true_causes": ep.meta.get("true_causes")}


# --------------------------------------------------------------------- uploads (analyse your own clip)
UPLOADS = ROOT / "data" / "uploads"
VIDEO_EXT = {".mp4", ".mov", ".webm", ".ogv", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_MB = 300
JOBS: dict[str, dict] = {}


def _run_clip_job(cid: str, src: Path, outcome: str, detector: str, start: float | None, end: float | None,
                  prompts: list[str] | None = None):
    job = JOBS[cid]
    try:
        from ingest.normalize import normalize_clip
        from perception.run import process

        def say(step):
            job.update(status="running", step=step)
        say("normalising video (30 fps)")
        clip = normalize_clip(src, UPLOADS / f"{cid}.norm.mp4", start, end, max_side=960)
        res = process(cid, clip, outcome, detector=detector, points="auto", prompts=prompts, progress=say, publish=False,
                      meta={"source": "upload", "original_name": job["name"]})
        job.update(status="done", step="done", health=res["health"])
    except Exception as e:  # report the failure to the UI instead of dying silently
        job.update(status="error", step="failed", error=str(e))


@app.post("/api/clips")
async def upload_clip(file: UploadFile = File(...), outcome: str = Form(...), detector: str = Form("yolo"),
                      start_s: float | None = Form(None), end_s: float | None = Form(None), animals: str = Form("")):
    """Upload a clip of ONE attack plus its known outcome; analysis runs in the background (poll /api/clips/{id})."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {ext!r}; use one of {sorted(VIDEO_EXT)}")
    if outcome not in ("caught", "escaped"):
        raise HTTPException(400, "outcome must be 'caught' or 'escaped' (the system works backwards from a known outcome)")
    if detector not in ("yolo", "blobs"):
        raise HTTPException(400, "detector must be 'yolo' (real footage) or 'blobs' (rendered test clips)")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    cid = "upload-" + uuid.uuid4().hex[:8]
    dst = UPLOADS / f"{cid}{ext}"
    size = 0
    with dst.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                fh.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB; trim it to the attack (5-15 s)")
            fh.write(chunk)
    JOBS[cid] = {"id": cid, "name": file.filename, "outcome": outcome, "status": "queued", "step": "queued", "error": None}
    prompts = [a.strip() for a in animals.split(",") if a.strip()][:6] or None  # e.g. "cat, pigeon" -> YOLO-World
    JOBS[cid]["animals"] = prompts
    threading.Thread(target=_run_clip_job, args=(cid, dst, outcome, detector, start_s, end_s, prompts), daemon=True).start()
    return JOBS[cid]


@app.get("/api/clips/{cid}")
def clip_status(cid: str, persona: str = "analyst", language: str = "en", llm: bool = False):
    job = JOBS.get(cid)
    ep_path = ROOT / "data/cache" / cid / "episode.json"
    if job is None and not ep_path.exists():
        raise HTTPException(404, "unknown clip")
    job = job or {"id": cid, "status": "done", "step": "done"}
    if job["status"] != "done":
        return job
    from events.schema import Episode
    ep = Episode.model_validate_json(ep_path.read_text())
    lib = hunt_library()
    a = lib["real"]["autopsy"] if "real" in lib and len(lib["real"]["episodes"]) >= 8 else lib["compound"]["autopsy"]
    ev = a.evidence(ep)
    return {**job, "evidence": ev, "narration": narrate_autopsy(ev, persona, language, use_llm=llm),
            "episode": _episode_payload(ep, a), "true_causes": None,
            "model_note": "explained by the chase model trained on simulated chases (distances in predator body lengths)"}


@app.get("/media/{clip_id}/overlay.mp4")
def media(clip_id: str):
    p = ROOT / "data/cache" / clip_id / "overlay.mp4"
    if not p.exists() or ".." in clip_id:
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


# --------------------------------------------------------------------- evals
@app.get("/api/evals")
def evals():
    p = ROOT / "data/evals/latest.json"
    if not p.exists():
        return JSONResponse({"available": False, "hint": "run: python -m evals.run_all"})
    return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}
