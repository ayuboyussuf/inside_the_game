"""Live match pipeline as a team of small agents sharing a blackboard.

  raw event -> IngestAgent -> StatsAgent -> MomentAgent -> AutopsyAgent
            -> (per viewer profile) PersonalizerAgent -> NarratorAgent <-> FactCheckAgent -> OverlayAgent

Each agent does one job, writes to the blackboard and appends to a trace, so the UI can
show *who decided what*. The live path is deterministic and fast (templates); the LLM is
used where latency allows: recaps, deep-dive autopsies and translation into languages
without a template bank.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from events.schema import Event
from football.adapter import shot_episode, to_event
from football.intel import MatchState
from football.overlays import cue_for, scorebug
from narrate.grounding import check
from narrate.narrator import live_line, narrate_autopsy, template_autopsy
from narrate.personalize import Profile, framing, wants
from narrate.templates import LIVE

from .models import shot_autopsy


@dataclass
class Board:
    raw: dict
    event: Event | None = None
    signals: list[dict] = field(default_factory=list)
    moments: list[dict] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)  # profile name -> {"lines": [...], "cues": [...]}
    trace: list[dict] = field(default_factory=list)

    def log(self, agent: str, msg: str, ms: float):
        self.trace.append({"agent": agent, "msg": msg, "ms": round(ms, 2)})


class Agent:
    name = "agent"

    def __call__(self, b: Board, ctx: "LiveMatch"):
        t = time.perf_counter()
        msg = self.run(b, ctx)
        if msg:
            b.log(self.name, msg, 1000 * (time.perf_counter() - t))

    def run(self, b: Board, ctx: "LiveMatch") -> str | None:  # pragma: no cover
        raise NotImplementedError


class IngestAgent(Agent):
    name = "ingest"

    def run(self, b, ctx):
        ev = to_event(b.raw)
        if ev is None:
            return f"dropped unknown type {b.raw.get('type')!r}"
        if ev.x is not None and not (-5 <= ev.x <= 110 and -5 <= (ev.y or 0) <= 73):
            return "dropped: location off the pitch"
        b.event = ev
        return None


class StatsAgent(Agent):
    name = "stats"

    def run(self, b, ctx):
        if b.event is None:
            return None
        b.signals = ctx.state.apply(b.event)
        return f"{len(b.signals)} signal(s): {', '.join(s['kind'] for s in b.signals)}" if b.signals else None


class MomentAgent(Agent):
    """Turns signals into moments with names, score context and 'why it matters' evidence."""
    name = "moments"

    def run(self, b, ctx):
        st = ctx.state
        for s in b.signals:
            ctx.n_moments += 1
            m = {**s, "id": f"m{ctx.n_moments:04d}", "team_name": ctx.team_name(s.get("team")),
                 "player_name": st.names.get(s.get("player")) if s.get("player") else None,
                 "score_str": ctx.score_str()}
            if s["kind"] == "flow_shift":
                e = m["evidence"]
                e["from_label"], e["to_label"] = ctx.flow_label(e["from"]), ctx.flow_label(e["to"])
            if s["kind"] == "goal":
                e = m["evidence"]
                m["why_matters"] = {"win_prob_swing": round(e["win_prob_after"] - e["win_prob_before"], 3),
                                    "against_the_odds": (e.get("xg") or 1) < 0.1, "late": st.minute() > 80}
                m["importance"] = 0.8 + min(0.2, abs(m["why_matters"]["win_prob_swing"]))
            b.moments.append(m)
            ctx.moments.append(m)
        return f"{len(b.signals)} moment(s)" if b.signals else None


class AutopsyAgent(Agent):
    """On-target shots: rebuild the shot as an Episode and explain why it went in / was saved."""
    name = "autopsy"

    def run(self, b, ctx):
        raw = b.raw
        if raw.get("type") != "shot" or raw.get("outcome") not in ("goal", "saved"):
            return None
        ep = shot_episode(raw)
        if ep is None:
            return None
        ev = shot_autopsy().evidence(ep)
        ev["clock"] = ctx.state.clock()
        ev["shooter"], ev["keeper"] = raw.get("player_name"), ep.meta.get("keeper_name")
        ctx.autopsies[raw["event_id"]] = {"evidence": ev, "true_causes": ep.meta.get("true_causes")}
        ctx.n_moments += 1
        keeper_id = (raw.get("detail") or {}).get("keeper_id")
        m = {"id": f"m{ctx.n_moments:04d}", "t": b.event.t + 0.5, "clock": ctx.state.clock(),
             "kind": "goal_autopsy" if raw["outcome"] == "goal" else "save_autopsy",
             "team": raw.get("team_id") if raw["outcome"] == "goal" else ctx.state.other(raw.get("team_id")),
             "player": raw.get("player_id") if raw["outcome"] == "goal" else keeper_id,
             "related_player": keeper_id if raw["outcome"] == "goal" else raw.get("player_id"),
             "player_name": raw.get("player_name") if raw["outcome"] == "goal" else ep.meta.get("keeper_name"),
             "team_name": None, "importance": 0.75 if raw["outcome"] == "goal" else 0.55,
             "evidence": {"shot_event_id": raw["event_id"]}, "autopsy": ev, "score_str": ctx.score_str()}
        m["team_name"] = ctx.team_name(m["team"])
        b.moments.append(m)
        ctx.moments.append(m)
        top = ev["top_factors"][0]
        return f"{raw['outcome']}: top factor {top['feature']} = {top['display']}"


class PersonalizerAgent(Agent):
    name = "personalizer"

    def run(self, b, ctx):
        n = 0
        for pname, prof in ctx.profiles.items():
            sel = [m for m in b.moments if wants(m, prof)]
            b.outputs[pname] = {"moments": sel, "lines": [], "cues": []}
            n += len(sel)
        return f"{n} moment(s) routed across {len(ctx.profiles)} profile(s)" if b.moments else None


class NarratorAgent(Agent):
    name = "narrator"

    def run(self, b, ctx):
        n = 0
        for pname, out in b.outputs.items():
            prof = ctx.profiles[pname]
            lang = prof.language if prof.language in LIVE else "en"
            for m in out["moments"]:
                if m.get("autopsy"):
                    why = template_autopsy(m["autopsy"], "analyst" if prof.voice == "analyst" else "casual", lang)
                    m2 = {**m, "why": why, "why_short": template_autopsy(m["autopsy"], "casual", lang)}
                    text = live_line(m2, prof.voice, lang)
                    evidence = {"m": m, "a": m["autopsy"]}
                else:
                    text = live_line(m, prof.voice, lang)
                    evidence = m
                if text:
                    out["lines"].append({"moment_id": m["id"], "t": m["t"], "clock": m.get("clock"), "kind": m["kind"], "text": text,
                                         "lang": lang, "framing": framing(m, prof), "_evidence": evidence})
                    n += 1
        return f"{n} line(s)" if n else None


class FactCheckAgent(Agent):
    """Rejects any line with a number not present in its moment's evidence (eval 9 rule)."""
    name = "fact_check"

    def run(self, b, ctx):
        bad = 0
        for out in b.outputs.values():
            keep = []
            for ln in out["lines"]:
                ev = ln.pop("_evidence")
                g = check(ln["text"], {"evidence": ev, "clock": ln["clock"], "score": ctx.state.score})
                ln["grounded"] = g["ok"]
                if g["ok"]:
                    keep.append(ln)
                else:
                    bad += 1
                    ctx.rejected.append({"text": ln["text"], "unsupported": g["unsupported"]})
            out["lines"] = keep
        return f"rejected {bad} ungrounded line(s)" if bad else None


class OverlayAgent(Agent):
    name = "overlay"

    def run(self, b, ctx):
        n = 0
        for pname, out in b.outputs.items():
            prof = ctx.profiles[pname]
            texts = {ln["moment_id"]: ln["text"] for ln in out["lines"]}
            for m in out["moments"]:
                if m["id"] in texts:
                    out["cues"].append(cue_for(m, texts[m["id"]], ctx.match_id, pname, prof.language, framing(m, prof)).model_dump())
                    n += 1
            out.pop("moments")
        return f"{n} overlay cue(s)" if n else None


PIPELINE = [IngestAgent(), StatsAgent(), MomentAgent(), AutopsyAgent(), PersonalizerAgent(), NarratorAgent(), FactCheckAgent(), OverlayAgent()]


class LiveMatch:
    def __init__(self, feed: dict, profiles: dict[str, Profile] | None = None):
        self.feed = feed
        self.meta = feed["metadata"]
        self.match_id = self.meta["match_id"]
        self.state = MatchState(self.meta)
        self.profiles = profiles or {"default": Profile()}
        self.moments: list[dict] = []
        self.autopsies: dict[str, dict] = {}
        self.rejected: list[dict] = []
        self.n_moments = 0
        self.cursor = 0

    def team_name(self, tid):
        if tid == self.meta["home"]["id"]:
            return self.meta["home"]["short"]
        if tid == self.meta["away"]["id"]:
            return self.meta["away"]["short"]
        return None

    def flow_label(self, label: str) -> str:
        if label.startswith("control:"):
            return f"control ({self.team_name(label.split(':')[1])})"
        return label

    def score_str(self) -> str:
        h, a = self.meta["home"], self.meta["away"]
        return f"{h['short']} {self.state.score[h['id']]}-{self.state.score[a['id']]} {a['short']}"

    def process(self, raw: dict) -> dict:
        b = Board(raw=raw)
        for agent in PIPELINE:
            agent(b, self)
            if b.event is None:
                break
        snap = self.state.snapshot() if b.event is not None else None
        return {"event": raw, "snapshot": snap, "scorebug": scorebug(snap, self.meta) if snap else None,
                "outputs": b.outputs, "trace": b.trace}

    def step(self) -> dict | None:
        if self.cursor >= len(self.feed["events"]):
            return None
        raw = self.feed["events"][self.cursor]
        self.cursor += 1
        return self.process(raw)

    def run_all(self) -> "LiveMatch":
        while self.step() is not None:
            pass
        return self


__all__ = ["LiveMatch", "Profile", "narrate_autopsy"]
