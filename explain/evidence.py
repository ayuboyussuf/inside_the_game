"""Autopsy = outcome model + timing model -> evidence JSON.

The evidence JSON is the ONLY thing the narrator sees. Every number carries a
pre-formatted ``display`` string so text and grounding check agree on rounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from events.features import FEATURE_NAMES, SPEC, extract_features, feature_table, label_for, unit_for
from events.schema import Episode

from .model import OutcomeExplainer
from .timeline import OutcomeTimeline

FOOTBALL_EVENT_LABELS = {"sprint_start": "the run-up began", "strike": "the ball was struck", "flee_start": "the keeper moved"}
EVENT_LABELS = {
    "sprint_start": "the sprint started", "flee_start": "the defender reacted", "turn": "a sharp cut",
    "contact": "contact", "abort": "the attacker gave up", "strike": "the strike", "goal": "the goal",
    "save": "the save",
}


def fmt(v: float, decimals: int, unit: str = "") -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{v:.{decimals}f}"
    if unit in ("count", "x", ""):
        return s + ("x" if unit == "x" else "")
    if unit == "deg":
        return f"{s}°"
    return f"{s} {unit}"


def pct(p: float) -> str:
    return f"{100 * p:.0f}%"


class Autopsy:
    def __init__(self, kind: str = "auto", seed: int = 0, features: list[str] | None = None, timeline: bool = True):
        self.model = OutcomeExplainer(kind=kind, seed=seed, features=features)
        self.use_timeline = timeline
        self.timeline: OutcomeTimeline | None = None

    def fit(self, episodes: list[Episode], df: pd.DataFrame | None = None) -> "Autopsy":
        df = feature_table(episodes) if df is None else df
        self.model.fit(df)
        self.names_ = episodes[0].outcome_names if episodes else ("escaped", "caught")
        self.domain_ = episodes[0].domain if episodes else "sim"
        if self.use_timeline:
            self.timeline = OutcomeTimeline().fit(episodes)
        return self

    def evidence(self, ep: Episode, top_k: int = 3) -> dict:
        x = pd.Series(extract_features(ep))
        m = self.model
        p1 = float(m.predict_proba(x.to_frame().T)[0])
        p_actual = p1 if ep.outcome == 1 else 1 - p1
        names = ep.outcome_names
        actual, other = names[ep.outcome], names[1 - ep.outcome]
        factors = m.explain(x, ep.outcome)
        out_factors = []
        for f in m.meaningful(x, ep.outcome, k=top_k):
            spec = SPEC[f.feature]
            unit = unit_for(f.feature, ep.distance_unit)
            item = {"feature": f.feature, "label": label_for(f.feature, ep.domain), "value": round(f.value, 4) if np.isfinite(f.value) else None,
                    "unit": unit, "display": fmt(f.value, spec.decimals, unit),
                    "contribution": round(f.contribution, 3),
                    "direction": f"toward {actual}" if f.contribution >= 0 else f"toward {other}",
                    "meaning": spec.football if ep.domain == "football" else spec.hunt}
            sim = m.similar_clips(f.feature, f.value, ep.outcome, exclude=ep.episode_id)
            if sim:
                sim["threshold_display"] = fmt(sim["threshold"], max(spec.decimals, 1) if unit not in ("count",) else 1, unit)
                sim["outcome"] = actual
                sim["display"] = (f"when {item['label'].lower()} is {'at or under' if sim['side'] == 'below' else 'over'} "
                                  f"{sim['threshold_display']}, the result was {actual} in {sim['k']} of {sim['n']} similar {'shots' if ep.domain == 'football' else 'clips'}")
                item["similar"] = sim
            out_factors.append(item)
        cf = m.counterfactual(x, factors[0].feature, ep.outcome)
        if cf:
            spec = SPEC[cf["feature"]]
            unit = unit_for(cf["feature"], ep.distance_unit)
            pa_from = cf["p_from"] if ep.outcome == 1 else 1 - cf["p_from"]
            pa_to = cf["p_to"] if ep.outcome == 1 else 1 - cf["p_to"]
            lab = label_for(cf["feature"], ep.domain)
            cf = {"feature": cf["feature"], "label": lab,
                  "from_display": fmt(cf["from"], spec.decimals, unit), "to_display": fmt(cf["to"], spec.decimals, unit),
                  "p_actual_from": round(pa_from, 3), "p_actual_to": round(pa_to, 3),
                  "display": (f"if {lab.lower()} had been {fmt(cf['to'], spec.decimals, unit)} instead of "
                              f"{fmt(cf['from'], spec.decimals, unit)}, the model's chance of '{actual}' moves from "
                              f"{pct(pa_from)} to {pct(pa_to)}")}
        moments = []
        if self.timeline is not None:
            dm = self.timeline.decisive_moment(ep)
            if dm:
                p_b = dm["p_before"] if ep.outcome == 1 else 1 - dm["p_before"]
                p_a = dm["p_after"] if ep.outcome == 1 else 1 - dm["p_after"]
                ev = dm["nearest_event"]
                labels = {**EVENT_LABELS, **FOOTBALL_EVENT_LABELS} if ep.domain == "football" else EVENT_LABELS
                moments.append({"t": dm["t"], "t_display": f"{dm['t']:.2f} s", "nearest_event": ev,
                                "nearest_event_label": labels.get(ev, ev) if ev else None,
                                "p_before": round(p_b, 3), "p_after": round(p_a, 3),
                                "display": f"the chance of '{actual}' went from {pct(p_b)} to {pct(p_a)} around {dm['t']:.2f} s"
                                           + (f", right as {labels.get(ev, ev)}" if ev else "")})
        key_events = [{"t": round(e.t, 2), "t_display": f"{e.t:.2f} s", "type": e.type, "actor": e.actor,
                       **({"angle_display": f"{abs(e.attrs['angle']):.0f}°"} if "angle" in e.attrs else {})}
                      for e in sorted(ep.events, key=lambda e: e.t) if e.type not in ("track_start", "stalk")]
        return {
            "episode_id": ep.episode_id, "domain": ep.domain, "outcome": actual, "other_outcome": other,
            "distance_unit": ep.distance_unit,
            "model": {"kind": m.kind_, "p_outcome": round(p_actual, 3), "p_outcome_display": pct(p_actual)},
            "top_factors": out_factors, "counterfactual": cf, "decisive_moments": moments, "key_events": key_events,
            "actors": [a.model_dump(exclude_none=True) for a in ep.actors if a.role in ("attacker", "defender")],
            "caveat": "explanatory factors with evidence, not proof of causation",
        }

    def feature_importance(self) -> list[dict]:
        imp = self.model.global_importance()
        return [{"feature": k, "label": label_for(k, self.domain_), "mean_abs_contribution": round(float(v), 3)} for k, v in imp.items()]


__all__ = ["Autopsy", "FEATURE_NAMES", "fmt", "pct"]
