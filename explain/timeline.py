"""Timing model: when did the outcome get decided?

A discrete-time state-value model (logistic regression on per-step state features, the
same family as a discrete-time hazard model) gives p(attacker succeeds | state at t).
The decisive moment is the 0.3 s window with the largest swing toward the actual
outcome: "the escape chance jumped from 0.31 to 0.74 at t = 2.10 s".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from events.features import _resample, key_times, velocity
from events.schema import Episode

STEP = 0.1
STATE_COLS = ["dist", "closing", "att_speed", "def_speed", "reacted", "unreacted_for", "turns_so_far", "since_commit", "alignment", "screens"]


def state_frame(ep: Episode) -> pd.DataFrame:
    att, dfd = ep.actor_by_role("attacker"), ep.actor_by_role("defender")
    ta, td = ep.track(att.id), ep.track(dfd.id)
    kt = key_times(ep)
    t0, t1 = max(kt["t_commit"], kt["t_trigger"] - 0.3), kt["t_end"]
    if ta is None or td is None or t1 - t0 < STEP:
        return pd.DataFrame(columns=["t"] + STATE_COLS)
    grid = np.arange(t0, t1 + 1e-9, STEP)
    fine = np.arange(max(ta.t[0], t0 - 0.5), t1 + 1e-9, 1 / 30)
    pa, pd_ = _resample(ta.array(), fine), _resample(td.array(), fine)
    va, vd = velocity(pa, fine), velocity(pd_, fine)
    dist = np.linalg.norm(pa - pd_, axis=1)
    closing = -np.gradient(dist, fine)
    sa, sd = np.linalg.norm(va, axis=1), np.linalg.norm(vd, axis=1)
    align = np.einsum("ij,ij->i", va, vd) / (sa * sd + 1e-6)
    flee = ep.first_event("flee_start", dfd.id)
    turns = sorted(e.t for e in ep.events_of("turn", dfd.id))
    sup = ep.actors_by_role("defender_support")
    rows = []
    for t in grid:
        i = int(np.clip(np.searchsorted(fine, t), 0, len(fine) - 1))
        # screening right now
        nscr = 0
        for s in sup:
            tr = ep.track(s.id)
            if tr is None:
                continue
            ps = _resample(tr.array(), np.array([t]))[0]
            seg = pd_[i] - pa[i]
            L = np.linalg.norm(seg) + 1e-9
            proj = np.dot(ps - pa[i], seg) / L
            perp = abs(seg[0] * (ps - pa[i])[1] - seg[1] * (ps - pa[i])[0]) / L
            nscr += int(0 < proj < L and perp < 1.0)
        rows.append({"t": round(float(t), 3), "dist": dist[i], "closing": closing[max(0, i - 3):i + 1].mean(),
                     "att_speed": sa[i], "def_speed": sd[i],
                     "reacted": float(flee is not None and t >= flee.t),
                     "unreacted_for": float(max(0.0, t - kt["t_trigger"]) if (flee is None or t < flee.t) else 0.0),
                     "turns_so_far": float(sum(1 for x in turns if x <= t)),
                     "since_commit": t - t0, "alignment": align[i], "screens": float(nscr)})
    return pd.DataFrame(rows)


class OutcomeTimeline:
    def fit(self, episodes: list[Episode]) -> "OutcomeTimeline":
        frames = []
        for ep in episodes:
            f = state_frame(ep)
            if len(f):
                f["y"] = ep.outcome
                frames.append(f)
        D = pd.concat(frames, ignore_index=True)
        self.sc_ = StandardScaler().fit(D[STATE_COLS])
        self.lr_ = LogisticRegression(C=0.5, max_iter=2000).fit(self.sc_.transform(D[STATE_COLS]), D["y"])
        return self

    def curve(self, ep: Episode) -> pd.DataFrame:
        f = state_frame(ep)
        if not len(f):
            return pd.DataFrame(columns=["t", "p"])
        f["p"] = self.lr_.predict_proba(self.sc_.transform(f[STATE_COLS]))[:, 1]
        return f[["t", "p"]]

    def decisive_moment(self, ep: Episode, window: float = 0.3) -> dict | None:
        c = self.curve(ep)
        if len(c) < 2:
            return None
        sign = 1.0 if ep.outcome == 1 else -1.0
        k = max(1, int(round(window / STEP)))
        p = c["p"].values
        # pad a pre-commit prior so a swing right at the commit is visible
        d = np.array([p[min(i + k, len(p) - 1)] - p[i] for i in range(len(p))])
        i = int(np.argmax(sign * d))
        j = min(i + k, len(p) - 1)
        t_mid = float(c["t"].iloc[i])
        near = [e for e in ep.events if abs(e.t - t_mid) <= 0.45 and e.type not in ("track_start", "stalk")]
        near.sort(key=lambda e: abs(e.t - t_mid))
        return {"t": round(t_mid, 2), "t_end": round(float(c["t"].iloc[j]), 2),
                "p_before": float(p[i]), "p_after": float(p[j]), "swing": float(sign * d[i]),
                "nearest_event": near[0].type if near else None,
                "nearest_event_t": round(near[0].t, 2) if near else None}
