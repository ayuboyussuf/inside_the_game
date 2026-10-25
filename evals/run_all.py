"""All ten evals behind one command; prints one table and writes data/evals/latest.json.

    python -m evals.run_all            # full suite
    python -m evals.run_all --quick    # smaller samples (CI)

Run after any change to events/ or explain/.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.models import FOOTBALL_FEATURES, load_real_episodes  # noqa: E402
from events.detect import detect_events  # noqa: E402
from events.features import FEATURE_NAMES, extract_features, feature_table, mirror_episode  # noqa: E402
from events.schema import Episode  # noqa: E402
from explain.evidence import Autopsy  # noqa: E402
from explain.model import OutcomeExplainer  # noqa: E402
from football.adapter import shot_episodes  # noqa: E402
from football.synth import generate_match, generate_shots  # noqa: E402
from narrate.grounding import check  # noqa: E402
from narrate.narrator import narrate_autopsy, template_autopsy  # noqa: E402
from narrate.templates import AUTOPSY  # noqa: E402
from sim.chase import RULES, generate  # noqa: E402

OUT = ROOT / "data" / "evals"
SIMCACHE = ROOT / "data" / "cache" / "sims"


def sims(rule: str, n: int, seed: int) -> list[Episode]:
    SIMCACHE.mkdir(parents=True, exist_ok=True)
    p = SIMCACHE / f"{rule}-{n}-{seed}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    eps = generate(n, rule, seed=seed)
    p.write_bytes(pickle.dumps(eps))
    return eps


def shots(n: int, seed: int) -> list[Episode]:
    SIMCACHE.mkdir(parents=True, exist_ok=True)
    p = SIMCACHE / f"shots-{n}-{seed}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    eps = shot_episodes(generate_shots(n, seed=seed))
    p.write_bytes(pickle.dumps(eps))
    return eps


def row(i, name, layer, how, result, bar, status, detail=None):
    return {"id": i, "name": name, "layer": layer, "how": how, "result": result, "bar": bar, "status": status, "detail": detail or {}}


# ------------------------------------------------------------------ 1
def eval_planted(n_train, n_test):
    per = {}
    hits_all, tot_all = 0, 0
    for rule in RULES:
        tr, te = sims(rule, n_train, 1), sims(rule, n_test, 2)
        m = OutcomeExplainer("lgbm").fit(feature_table(tr))
        dte = feature_table(te)
        h = t = 0
        for ep in te:
            if ep.meta.get("label_flipped"):
                continue  # noisy label: no true cause to recover
            t += 1
            h += m.top_factor(dte.loc[ep.episode_id], ep.outcome) in ep.meta["true_causes"]
        per[rule] = round(h / t, 3)
        hits_all += h
        tot_all += t
    tr, te = shots(1200, 100), shots(n_test * 2, 101)
    m = OutcomeExplainer("lgbm", features=FOOTBALL_FEATURES).fit(feature_table(tr))
    dte = feature_table(te)
    h = sum(m.top_factor(dte.loc[e.episode_id], e.outcome) in (e.meta["true_causes"] or []) for e in te)
    per["football_shots"] = round(h / len(te), 3)
    hits_all += h
    tot_all += len(te)
    acc = hits_all / tot_all
    worst = min(per.values())
    return row(1, "Planted-cause recovery", "Explainer",
               f"{len(RULES)} planted chase rules + planted keeper model; top factor must be the planted cause",
               f"{acc:.1%} overall (worst rule {worst:.0%})", "≥ 80% of clips", "PASS" if acc >= 0.8 else "FAIL", per)


# ------------------------------------------------------------------ 2
def eval_deletion(n_train, n_test, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for rule in ("compound", "reaction", "turns"):
        tr, te = sims(rule, n_train, 1), sims(rule, n_test, 2)
        dtr, dte = feature_table(tr), feature_table(te)
        m = OutcomeExplainer("lgbm").fit(dtr)
        med = dtr[FEATURE_NAMES].median()
        dtop, drnd = [], []
        for ep in te:
            x = dte.loc[ep.episode_id, FEATURE_NAMES].astype(float)
            p0 = m.predict_proba(x.to_frame().T)[0]
            top = m.top_factor(x, ep.outcome)
            x1 = x.copy(); x1[top] = med[top]
            others = [f for f in m.used_ if f != top]
            r = rng.choice(others)
            x2 = x.copy(); x2[r] = med[r]
            dtop.append(abs(m.predict_proba(x1.to_frame().T)[0] - p0))
            drnd.append(abs(m.predict_proba(x2.to_frame().T)[0] - p0))
        res[rule] = round(float(np.mean(dtop) / max(np.mean(drnd), 1e-9)), 2)
    ratio = float(np.mean(list(res.values())))
    return row(2, "Deletion faithfulness", "Explainer", "replace top feature vs a random feature with the training median; compare |Δp|",
               f"top/random drop = {ratio:.1f}×", "> 2×", "PASS" if ratio > 2 else "FAIL", res)


# ------------------------------------------------------------------ 3
def eval_prediction():
    real = load_real_episodes()
    det = {}
    if len(real) >= 10:
        df = feature_table(real)
        ps = []
        for i in range(len(df)):
            tr = df.drop(index=df.index[i])
            m = OutcomeExplainer("logreg").fit(tr)
            ps.append(m.predict_proba(df.iloc[[i]][FEATURE_NAMES])[0])
        y = df["outcome"].values
        auc = roc_auc_score(y, ps)
        acc = float(np.mean((np.array(ps) > 0.5) == y))
        maj = max(y.mean(), 1 - y.mean())
        return row(3, "Outcome prediction", "Explainer", f"leave-one-clip-out on {len(df)} real clips vs majority baseline",
                   f"AUC {auc:.2f}, acc {acc:.0%} vs majority {maj:.0%}", "beats majority; report AUC",
                   "PASS" if acc > maj else "FAIL", {"auc": auc})
    for name, eps, feats in (("chases (compound, n=200)", sims("compound", 200, 3), FEATURE_NAMES),
                             ("shots (n=400)", shots(400, 102), FOOTBALL_FEATURES)):
        df = feature_table(eps)
        y = df["outcome"].values
        ps = np.zeros(len(df))
        for tri, tei in StratifiedKFold(5, shuffle=True, random_state=0).split(df, y):
            m = OutcomeExplainer("auto", features=feats).fit(df.iloc[tri])
            ps[tei] = m.predict_proba(df.iloc[tei])
        det[name] = {"auc": round(roc_auc_score(y, ps), 3), "acc": round(float(np.mean((ps > 0.5) == y)), 3),
                     "majority": round(float(max(y.mean(), 1 - y.mean())), 3)}
    ok = all(v["acc"] > v["majority"] for v in det.values())
    txt = "; ".join(f"{k}: AUC {v['auc']:.2f}, acc {v['acc']:.0%} vs {v['majority']:.0%}" for k, v in det.items())
    return row(3, "Outcome prediction", "Explainer",
               f"5-fold CV on synthetic stand-ins ({len(real)} real clips < 10; add clips to switch to leave-one-clip-out)",
               txt, "beats majority; report AUC", "PASS" if ok else "FAIL", det)


# ------------------------------------------------------------------ 4
def _trim(ep: Episode, cut: float) -> Episode:
    e = ep.model_copy(deep=True)
    for tr in e.tracks:
        keep = [i for i, t in enumerate(tr.t) if t >= cut]
        tr.t, tr.x, tr.y = [tr.t[i] for i in keep], [tr.x[i] for i in keep], [tr.y[i] for i in keep]
    e.events = [ev for ev in e.events if ev.t >= cut or ev.type not in ("track_start",)]
    return e


def eval_stability(n_train, n_test):
    tr, te = sims("compound", n_train, 1), sims("compound", n_test, 2)
    dtr = feature_table(tr)
    base = OutcomeExplainer("lgbm", seed=0).fit(dtr)
    rng = np.random.default_rng(0)
    runs = []
    for k in range(5):
        boot = dtr.sample(frac=1.0, replace=True, random_state=k)
        m = OutcomeExplainer("lgbm", seed=k).fit(boot)
        cut = float(rng.uniform(0.0, 0.3))
        runs.append((m, cut))
    agree, agree1 = [], []
    for ep in te[: min(len(te), 120)]:
        x = pd.Series(extract_features(ep))
        ref = {f.feature for f in base.meaningful(x, ep.outcome, k=2)}
        ref1 = base.top_factor(x, ep.outcome)
        same = same1 = 0
        for m, cut in runs:
            xe = pd.Series(extract_features(_trim(ep, cut)))
            same += {f.feature for f in m.meaningful(xe, ep.outcome, k=2)} == ref
            same1 += m.top_factor(xe, ep.outcome) == ref1
        agree.append(same >= 4)
        agree1.append(same1 >= 4)
    share, share1 = float(np.mean(agree)), float(np.mean(agree1))
    return row(4, "Stability", "Explainer",
               "5 seeds × bootstrap refits × 0–0.3 s edge trims; reported factors (top + secondaries ≥ 25% of its weight) must repeat",
               f"{share:.0%} of clips keep the same reported top-2 in ≥4/5 runs; top-1 alone {share1:.0%}",
               "same top-2 in ≥ 4 of 5 runs (≥ 80% of clips)", "PASS" if share >= 0.8 else "FAIL", {"share": share, "top1": share1})


# ------------------------------------------------------------------ 5
def eval_flip(n_train, n_test):
    tr, te = sims("compound", n_train, 1), sims("pack", n_test // 2, 2) + sims("compound", n_test // 2, 2)
    m = OutcomeExplainer("lgbm").fit(feature_table(tr))
    fd, pdiff = 0.0, 0.0
    for ep in te:
        a, b = extract_features(ep), extract_features(mirror_episode(ep))
        fd = max(fd, max(abs(a[k] - b[k]) for k in a if np.isfinite(a[k])))
        pa = m.predict_proba(pd.Series(a).to_frame().T)[0]
        pb = m.predict_proba(pd.Series(b).to_frame().T)[0]
        pdiff = max(pdiff, abs(pa - pb))
    ok = fd < 1e-6 and pdiff < 1e-6
    return row(5, "Flip invariance", "Features", "mirror every clip horizontally",
               f"max |Δfeature| = {fd:.1e}, max |Δp| = {pdiff:.1e}", "unchanged", "PASS" if ok else "FAIL")


# ------------------------------------------------------------------ 6, 7, 8 (perception on rendered clips)
def perception_clips(k: int):
    try:
        import cv2  # noqa: F401
    except Exception:
        return None
    from perception.run import process
    from sim.render import render
    out = []
    eps = sims("compound", 12, 9)[:k]
    for i, ep in enumerate(eps):
        vid = ROOT / "data" / "raw" / f"eval_render_{i}.mp4"
        if not vid.exists():
            render(ep, vid, seed=i)
        res = process(f"eval_render_{i}", vid, ep.outcome_label, detector="blobs", points="lk", overlay=(i == 0), publish=False)
        out.append((ep, res))
    return out


def eval_camera(clips):
    if clips is None:
        return row(6, "Camera compensation", "Perception", "", "OpenCV not installed", "< 0.1 BL/s", "SKIP")
    from perception.camera import apply
    comp, raw = [], []
    for ep, res in clips:
        d = ROOT / "data" / "cache" / ep_name(res)
        z = np.load(d / "points_lk.npz")
        to_ref = np.load(d / "camera_lk.npz")["to_ref"]
        bl = res["episode"].meta["body_length_px"]
        tracks, vis, owner = z["tracks"], z["visible"], z["owner"]
        k = 15
        for i in np.nonzero(owner < 0)[0]:
            ts = np.nonzero(vis[i])[0]
            for a in range(0, len(ts) - k, k):
                t0, t1 = ts[a], ts[a + k]
                if t1 - t0 != k:
                    continue
                p0, p1 = apply(to_ref, t0, tracks[i, t0][None])[0], apply(to_ref, t1, tracks[i, t1][None])[0]
                comp.append(np.linalg.norm(p1 - p0) / (k / 30) / bl)
                raw.append(np.linalg.norm(tracks[i, t1] - tracks[i, t0]) / (k / 30) / bl)
    c, r = float(np.median(comp)), float(np.median(raw))
    return row(6, "Camera compensation", "Perception", f"background dots on {len(clips)} rendered pan+zoom clips, 0.5 s windows",
               f"{c:.3f} BL/s after (raw camera motion {r:.2f} BL/s)", "< 0.1 BL/s", "PASS" if c < 0.1 else "FAIL",
               {"compensated": c, "raw": r})


def ep_name(res):
    return Path(res["cache"]).name


def eval_timing(clips, n=60):
    errs = {"sprint_start": [], "flee_start": []}
    for ep in sims("compound", n, 4):
        att, dfd = ep.track("pred").array(), ep.track("prey").array()
        evs = detect_events(att[:, 0], att[:, 1:], dfd[:, 1:], "pred", "prey")
        for k in errs:
            g, d = ep.first_event(k), next((e for e in evs if e.type == k), None)
            errs[k].append(abs(g.t - d.t) if d else np.inf)
    vid = {"sprint_start": [], "flee_start": []}
    if clips:
        for ep, res in clips:
            for k in vid:
                g, d = ep.first_event(k), res["episode"].first_event(k)
                vid[k].append(abs(g.t - d.t) if d else np.inf)
    real = load_real_episodes()
    hand = []
    for e in real:
        f = e.meta.get("sprint_start_frame")
        d = e.first_event("sprint_start")
        if f not in (None, "") and d:
            hand.append(abs(int(f) / e.fps - d.t))
    within = {k: float(np.mean(np.array(v) <= 0.3)) for k, v in errs.items()}
    within_vid = {k: float(np.mean(np.array(v) <= 0.3)) for k, v in vid.items() if v}
    allv = list(within.values()) + list(within_vid.values())
    ok = min(allv) >= 0.7
    txt = (f"tracks: sprint {within['sprint_start']:.0%}, flee {within['flee_start']:.0%} within 0.3 s"
           + (f"; video: sprint {within_vid.get('sprint_start', 0):.0%}, flee {within_vid.get('flee_start', 0):.0%}" if within_vid else "")
           + (f"; real hand-marked: {np.mean(np.array(hand) <= 0.3):.0%} of {len(hand)}" if hand else ""))
    return row(7, "Event timing", "Events", "detected vs ground-truth sprint start / first flee move", txt,
               "within 0.3 s on most clips (≥ 70%)", "PASS" if ok else "FAIL",
               {"median_err_tracks": {k: float(np.median(v)) for k, v in errs.items()}})


def eval_health(clips):
    if not clips:
        return row(8, "Track health", "Perception", "", "OpenCV not installed", "log both numbers", "SKIP")
    body = [r["health"]["dots_alive_at_end"]["body"] for _, r in clips]
    bg = [r["health"]["dots_alive_at_end"]["background"] for _, r in clips]
    ids = [r["health"]["id_births_after_first_frame"] for _, r in clips]
    return row(8, "Track health", "Perception", "share of dots alive at clip end; new track IDs after frame 0 (ID switches)",
               f"body dots alive {np.mean(body):.0%}, background {np.mean(bg):.0%} (pans push them out of frame); ID births/clip {np.mean(ids):.1f}",
               "watch the overlay; log both numbers", "INFO", {"body": body, "background": bg, "id_births": ids})


# ------------------------------------------------------------------ 9
def eval_grounding(n_train, n=60):
    from agents.models import shot_autopsy
    from agents.orchestrator import LiveMatch
    from narrate.personalize import Profile
    tr = sims("compound", n_train, 1)
    a = Autopsy("lgbm").fit(tr)
    sa = shot_autopsy()
    bad, total, examples = 0, 0, []
    evs = [a.evidence(ep) for ep in sims("compound", n, 2)] + [sa.evidence(ep) for ep in shots(n, 103)]
    for ev in evs:
        for persona in ("analyst", "casual"):
            for lang in AUTOPSY:
                txt = template_autopsy(ev, persona, lang)
                g = check(txt, ev)
                total += 1
                if not g["ok"]:
                    bad += 1
                    examples.append((txt, g["unsupported"]))
    llm_res = None
    from narrate.llm import get_llm
    if get_llm() is not None:
        lb = 0
        for ev in evs[:8]:
            r = narrate_autopsy(ev, "analyst", "en")
            lb += (r["source"] != "template") and not r["grounding"]["ok"]
        llm_res = lb
    live = LiveMatch(generate_match(11), {"a": Profile(persona="analyst"), "b": Profile(persona="casual", language="fr", favorite_club="NBR")}).run_all()
    lines = len(live.rejected)
    ok = bad == 0 and lines == 0
    return row(9, "Grounded narration", "Narrator", "every number in the text must appear in the evidence JSON (rule check, no LLM)",
               f"{bad} unsupported in {total} autopsy narrations ({len(AUTOPSY)} languages × 2 personas); {lines} live lines rejected by fact-check"
               + (f"; LLM drafts that failed after retry: {llm_res}" if llm_res is not None else "; LLM not configured"),
               "zero unsupported numbers", "PASS" if ok else "FAIL", {"examples": examples[:3]})


# ------------------------------------------------------------------ 10
def eval_human(n_train):
    OUT.mkdir(parents=True, exist_ok=True)
    resp = OUT / "human_responses.csv"
    survey = OUT / "human_survey.json"
    if not survey.exists():
        tr = sims("compound", n_train, 1)
        a = Autopsy("lgbm").fit(tr)
        eps = sims("compound", 20, 7)[:12]
        evs = [a.evidence(e) for e in eps]
        rnd = random.Random(0)
        items = []
        for i, (ep, ev) in enumerate(zip(eps, evs)):
            other = evs[(i + 5) % len(evs)]
            real_txt, fake_txt = template_autopsy(ev, "casual"), template_autopsy({**other, "outcome": ev["outcome"]}, "casual")
            opts = [("real", real_txt), ("shuffled", fake_txt)]
            rnd.shuffle(opts)
            items.append({"clip": ep.episode_id, "A": opts[0][1], "B": opts[1][1], "answer": "A" if opts[0][0] == "real" else "B"})
        survey.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    if not resp.exists():
        return row(10, "Human agreement", "Whole system", f"show 3–5 people a clip + two explanations ({survey.relative_to(ROOT)}); "
                   "record picks in data/evals/human_responses.csv (rater,clip,pick)", "no responses yet", "real one picked > 50%", "SKIP")
    items = {it["clip"]: it["answer"] for it in json.loads(survey.read_text(encoding="utf-8"))}
    picks = list(csv.DictReader(resp.open(encoding="utf-8")))
    hits = [p["pick"].strip().upper() == items.get(p["clip"]) for p in picks if p["clip"] in items]
    share = float(np.mean(hits)) if hits else 0.0
    return row(10, "Human agreement", "Whole system", f"{len(set(p['rater'] for p in picks))} raters, {len(hits)} judgements",
               f"real explanation picked {share:.0%}", "clearly > 50%", "PASS" if share > 0.6 else "FAIL", {"share": share})


# ------------------------------------------------------------------ 11
FOOTBALL_BARS = {"player_recall": 0.90, "on_pitch_fp": 0.05, "ball_recall": 0.70, "median_track_s": 3.0,
                 "track_coverage_2s": 0.80, "team_purity": 0.90, "homography_clips": 1.0}


def eval_football_video():
    """Real (openly licensed) football footage: per-stage failure analysis. Run evals/football_video.py first (slow, GPU
    recommended); this reads its cached results plus the manual audit labels."""
    d = OUT / "football_video"
    res_p, aud_p = d / "results_yolo11s.json", d / "audit.json"
    if not res_p.exists() or not aud_p.exists():
        return row(11, "Football video (real footage)", "Perception", "python -m evals.football_video, then audit frames",
                   "no results yet", "per-stage bars", "SKIP")
    res = json.loads(res_p.read_text(encoding="utf-8"))
    aud = json.loads(aud_p.read_text(encoding="utf-8"))["frames"]
    vis = sum(f["visible_players"] for f in aud)
    det = sum(f["detected_correct"] for f in aud)
    fp = sum(f["false_pos_on_pitch"] for f in aud)
    bv = [f for f in aud if f["ball_visible"]]
    stages = {
        "player_recall": det / max(1, vis),
        "on_pitch_fp": fp / max(1, det + fp),
        "ball_recall": sum(f["ball_detected"] for f in bv) / max(1, len(bv)),
        "median_track_s": float(np.median([r["tracking"]["median_track_len_s"] for r in res])),
        "team_purity": float(np.mean([r["teams"]["track_purity"] for r in res if r["teams"]["track_purity"] is not None])),
        "homography_clips": 0.0,  # no pitch-keypoint model: no clip gets an automatic image->pitch homography
    }
    v2_p = d / "results_v2.json"
    pipeline = "v1 (COCO YOLO, ByteTrack 10 fps, single-frame ball)"
    if v2_p.exists():
        v2 = {r["clip"]: r for r in json.loads(v2_p.read_text(encoding="utf-8"))}
        if len(v2) >= len(res):
            pipeline = "v2 (" + ", ".join(f"{k} {v}" for k, v in next(iter(v2.values()))["models"].items()) +                        "; tiled ball + Kalman, BoT-SORT re-ID 30 fps)"
            hits = []
            for f in aud:
                if f.get("ball_xy") and f["clip"] in v2:
                    rec = v2[f["clip"]]["audit_frames"].get(f"{f['t']:.1f}")
                    if rec is not None:
                        b = rec["ball"]
                        hits.append(b is not None and np.hypot(b[0] - f["ball_xy"][0], b[1] - f["ball_xy"][1]) < 15)
            if hits:
                stages["ball_recall"] = float(np.mean(hits))
                bv = [None] * len(hits)
                ball_hits = int(sum(hits))
            stages["track_coverage_2s"] = float(np.mean([r["tracking"]["coverage_2s"] for r in v2.values()]))
            stages.pop("median_track_s", None)
            stages["homography_clips"] = float(np.mean([r["pitch"]["homography_share"] > 0.5 for r in v2.values()]))
    ok = {k: (v <= FOOTBALL_BARS[k] if k == "on_pitch_fp" else v >= FOOTBALL_BARS[k]) for k, v in stages.items()}
    spurious = sum(r["events"]["passes"] for r in res if "penalty" in r["clip"])
    track_txt = (f"track coverage {stages['track_coverage_2s']:.0%} (≥2 s)" if "track_coverage_2s" in stages
                 else f"median track {stages['median_track_s']:.1f} s")
    txt = (f"[{pipeline}] players {stages['player_recall']:.0%} recall ({det}/{vis}), {stages['on_pitch_fp']:.0%} false on-pitch; "
           f"ball {stages['ball_recall']:.0%} of {len(bv)} visible; {track_txt}; team purity {stages['team_purity']:.0%}; "
           f"homography {stages['homography_clips']:.0%} of clips; {spurious} spurious passes on penalty clips (v1); "
           f"failing: {', '.join(k for k, v in ok.items() if not v) or 'none'}")
    return row(11, "Football video (real footage)", "Perception",
               f"{len(res)} openly licensed clips (data/raw/football/SOURCES.json), {len(aud)} audited frames",
               txt, "recall ≥ 90%, FP ≤ 5%, ball ≥ 70%, tracks ≥ 3 s, purity ≥ 90%, homography on all clips",
               "PASS" if all(ok.values()) else "FAIL", {"stages": stages, "pass": ok, "per_clip": res})


# ------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    n_train, n_test = (300, 100) if a.quick else (500, 200)
    t0 = time.time()
    clips = None if a.no_video else perception_clips(2 if a.quick else 6)
    steps = [lambda: eval_planted(n_train, n_test), lambda: eval_deletion(n_train, n_test), eval_prediction,
             lambda: eval_stability(n_train, n_test), lambda: eval_flip(n_train, n_test), lambda: eval_camera(clips),
             lambda: eval_timing(clips), lambda: eval_health(clips), lambda: eval_grounding(n_train), lambda: eval_human(n_train),
             eval_football_video]
    results = []
    for s in steps:
        t = time.time()
        try:
            r = s()
        except Exception as e:  # keep the table going; report the failure
            r = row(len(results) + 1, s.__name__, "?", "", f"error: {e!r}", "", "FAIL")
        r["seconds"] = round(time.time() - t, 1)
        results.append(r)
        print(f"  [{r['status']:>4}] {r['id']:>2}. {r['name']:<24} {r['result']}  ({r['seconds']} s)", flush=True)
    secs = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "seconds": secs, "results": results}
    (OUT / "latest.json").write_text(json.dumps(payload, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("\n" + "-" * 110)
    print(f"{'#':>2}  {'Eval':<24} {'Layer':<12} {'Status':<6} Result")
    for r in results:
        print(f"{r['id']:>2}  {r['name']:<24} {r['layer']:<12} {r['status']:<6} {r['result']}")
    print("-" * 110)
    print(f"{sum(r['status'] == 'PASS' for r in results)} pass, {sum(r['status'] == 'FAIL' for r in results)} fail, "
          f"{sum(r['status'] in ('SKIP', 'INFO') for r in results)} skip/info · {secs} s · data/evals/latest.json")
    return 0 if not any(r["status"] == "FAIL" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
