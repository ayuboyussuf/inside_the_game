"""Real-time match intelligence: aggregate a live event stream into football states.

``MatchState.apply(event)`` is O(1)-ish per event and returns *signals* (milestones,
threshold triggers, shifts) for the moment/narration agents. ``snapshot()`` returns a
JSON-ready state for overlays and the UI.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

from events.schema import Event

WINDOW_S = 300.0          # rolling window for flow metrics
MOMENTUM_HALF_LIFE = 180.0
SPEED_TAG_KMH = 31.0      # player speed indicator threshold
DISTANCE_STEP_KM = 2.0    # distance indicator every 2 km
PASS_STREAK = 15
BASE_GOALS_90 = 1.4


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def win_probability(h: int, a: int, minute: float, lam_h90: float, lam_a90: float, red=(0, 0)) -> dict:
    rem = max(0.0, 95 - minute) / 90.0
    lh, la = lam_h90 * rem * (0.75 if red[0] else 1), lam_a90 * rem * (0.75 if red[1] else 1)
    ph = pd = pa = 0.0
    for i in range(11):
        for j in range(11):
            p = _poisson(i, lh) * _poisson(j, la)
            if h + i > a + j:
                ph += p
            elif h + i == a + j:
                pd += p
            else:
                pa += p
    s = ph + pd + pa
    return {"home": ph / s, "draw": pd / s, "away": pa / s}


class TeamStats(defaultdict):
    def __init__(self):
        super().__init__(float)


class MatchState:
    def __init__(self, metadata: dict):
        self.meta = metadata
        self.home, self.away = metadata["home"]["id"], metadata["away"]["id"]
        self.names = {p["id"]: p["name"] for tm in metadata["lineups"].values() for p in tm}
        self.player_team = {p["id"]: p["team"] for tm in metadata["lineups"].values() for p in tm}
        self.positions = {p["id"]: p["position"] for tm in metadata["lineups"].values() for p in tm}
        self.pace = {p["id"]: p["pace"] for tm in metadata["lineups"].values() for p in tm}
        self.skill = {p["id"]: p["skill"] for tm in metadata["lineups"].values() for p in tm}
        self.team = {self.home: TeamStats(), self.away: TeamStats()}
        self.player: dict[str, defaultdict] = defaultdict(lambda: defaultdict(float))
        self.on_pitch = {p["id"] for tm in metadata["lineups"].values() for p in tm[:11]}
        self.score = {self.home: 0, self.away: 0}
        self.reds = {self.home: 0, self.away: 0}
        self.t = 0.0
        self.period = 1
        self.poss_team: str | None = None
        self.poss_since = 0.0
        self.window: deque = deque()
        self.threat = {self.home: 0.0, self.away: 0.0}
        self.last_threat_t = 0.0
        self.last_pass: dict | None = None
        self.flow_label = "balanced"
        self.press_leader: str | None = None
        self.momentum_history: list[tuple[float, float]] = []
        self.wp_history: list[tuple[float, dict]] = []
        self.n_events = 0
        self.last_swing_t = -1e9
        self.prev_wp = self.win_prob()

    # ------------------------------------------------------------------
    def other(self, team: str) -> str:
        return self.away if team == self.home else self.home

    def minute(self) -> float:
        return self.t / 60.0

    def clock(self) -> str:
        m = int(self.t // 60)
        if self.period == 1 and m >= 45:
            return f"45+{m - 44}'"
        if m >= 90:
            return f"90+{m - 89}'"
        return f"{m + 1}'"

    def win_prob(self) -> dict:
        m = max(self.minute(), 1.0)
        lam = {}
        for tm in (self.home, self.away):
            live = self.team[tm]["xg"] / m * 90 if hasattr(self, "team") else BASE_GOALS_90
            w = min(1.0, m / 90.0)
            lam[tm] = (1 - w) * BASE_GOALS_90 * (1.08 if tm == self.home else 0.95) + w * max(live, 0.2)
        return win_probability(self.score[self.home], self.score[self.away], self.minute(), lam[self.home], lam[self.away],
                               (self.reds[self.home], self.reds[self.away]))

    def _decay(self):
        dt = self.t - self.last_threat_t
        if dt > 0:
            f = 0.5 ** (dt / MOMENTUM_HALF_LIFE)
            for k in self.threat:
                self.threat[k] *= f
            self.last_threat_t = self.t

    def momentum(self) -> float:
        h, a = self.threat[self.home], self.threat[self.away]
        return math.tanh(3 * (h - a))

    # ------------------------------------------------------------------
    def apply(self, ev: Event) -> list[dict]:
        signals: list[dict] = []
        a = ev.attrs
        team = a.get("team")
        pid = ev.actor
        self.t = ev.t
        self.n_events += 1
        self.period = int(a.get("period", self.period))
        self._decay()
        # possession clock
        on_ball = ev.type in ("pass", "carry", "shot", "possession_change", "kickoff", "corner", "free_kick")
        if on_ball and team:
            if self.poss_team is not None:
                self.team[self.poss_team]["poss_time"] += max(0.0, ev.t - self.poss_since)
            self.poss_team, self.poss_since = team, ev.t
        self.window.append((ev.t, ev.type, team, a.get("outcome"), a.get("sequence_length"), ev.x))
        while self.window and self.window[0][0] < ev.t - WINDOW_S:
            self.window.popleft()
        T = self.team[team] if team in self.team else None
        P = self.player[pid] if pid else None
        if P is not None and ev.type not in ("substitution",):
            P["touches"] += 1 if on_ball else 0

        if ev.type == "pass" and T is not None:
            ok = a.get("outcome") == "complete"
            T["passes"] += 1
            T["passes_completed"] += ok
            T["pass_length"] += a.get("length_m", 0.0)
            T["pass_difficulty"] += a.get("difficulty", 0.0)
            T["xpass"] += a.get("expected_completion", 0.0)
            T["ball_speed_sum"] += a.get("ball_speed_kmh", 0.0)
            P["passes"] += 1
            P["passes_completed"] += ok
            P["pass_difficulty"] += a.get("difficulty", 0.0)
            P["xpass"] += a.get("expected_completion", 0.0)
            start = ev.x if ev.x is not None else 0.0
            if (start if team == self.home else 105 - start) < 63:
                T["passes_own60"] += 1  # PPDA denominator: passes the opponent allows in this team's build-up zone
            end = a.get("end_location") or [0, 0]
            att_x = end[0] if team == self.home else 105 - end[0]
            if ok and att_x > 70:
                T["final_third_passes"] += 1
                self.threat[team] += 0.02
            if ok and a.get("progressive"):
                T["progressive"] += 1
                self.threat[team] += 0.01
            if ok and a.get("difficulty", 0) >= 0.75:
                T["difficult_completed"] += 1
                signals.append({"kind": "difficult_pass", "importance": 0.1 + 0.3 * a["difficulty"], "team": team, "player": pid,
                                "evidence": {"difficulty": a["difficulty"], "length_m": a.get("length_m"),
                                             "expected_completion": a.get("expected_completion"), "under_pressure": a.get("under_pressure")}})
            if ok:
                P["streak"] += 1
                if P["streak"] == PASS_STREAK:
                    signals.append({"kind": "pass_streak", "importance": 0.3, "team": team, "player": pid,
                                    "evidence": {"consecutive_completed": int(P["streak"])}})
            else:
                P["streak"] = 0
            self.last_pass = {"player": pid, "team": team, "t": ev.t, "recipient": a.get("recipient_id")} if ok else None
            for n in (50, 100):
                if P["passes"] == n:
                    signals.append({"kind": "milestone", "importance": 0.2 + n / 500, "team": team, "player": pid,
                                    "evidence": {"stat": "passes", "value": n,
                                                 "accuracy_pct": round(100 * P["passes_completed"] / P["passes"], 1)}})
        elif ev.type == "carry" and T is not None:
            P["carry_m"] += a.get("distance_m", 0.0)
            sp = a.get("speed_kmh", 0.0)
            if sp > P["top_speed"]:
                P["top_speed"] = sp
            if sp >= SPEED_TAG_KMH:
                signals.append({"kind": "speed_tag", "importance": 0.25, "team": team, "player": pid,
                                "evidence": {"speed_kmh": sp, "threshold_kmh": SPEED_TAG_KMH}})
        elif ev.type == "shot" and T is not None:
            xg = a.get("xg", 0.0)
            T["shots"] += 1
            T["xg"] += xg
            T["on_target"] += bool(a.get("on_target"))
            T["shot_speed_max"] = max(T["shot_speed_max"], a.get("shot_speed_kmh", 0.0))
            P["shots"] += 1
            P["xg"] += xg
            self.threat[team] += xg
            assister = None
            if self.last_pass and self.last_pass["recipient"] == pid and ev.t - self.last_pass["t"] < 12:
                assister = self.last_pass["player"]
                self.player[assister]["key_passes"] += 1
            signals.append({"kind": "shot", "importance": 0.25 + xg + (0.2 if a.get("on_target") else 0), "team": team, "player": pid,
                            "evidence": {"xg": xg, "outcome": a.get("outcome"), "shot_speed_kmh": a.get("shot_speed_kmh"),
                                         "on_target": bool(a.get("on_target")), "key_pass_by": assister},
                            "event_id": a.get("id")})
            if xg >= 0.25 and a.get("outcome") != "goal":
                signals.append({"kind": "big_chance_missed", "importance": min(0.75, 0.35 + 0.5 * xg), "team": team, "player": pid,
                                "evidence": {"xg": xg, "outcome": a.get("outcome")}, "event_id": a.get("id")})
        elif ev.type == "goal" and team in self.score:
            before = self.win_prob()
            self.score[team] += 1
            T["goals"] += 1
            P["goals"] += 1
            after = self.win_prob()
            side = "home" if team == self.home else "away"
            signals.append({"kind": "goal", "importance": 1.0, "team": team, "player": pid,
                            "evidence": {"score": dict(self.score), "xg": a.get("xg"),
                                         "win_prob_before": round(before[side], 3), "win_prob_after": round(after[side], 3),
                                         "player_goals": int(P["goals"])}})
            if P["goals"] == 2:
                signals.append({"kind": "milestone", "importance": 0.6, "team": team, "player": pid, "evidence": {"stat": "goals", "value": 2}})
            if P["goals"] == 3:
                signals.append({"kind": "milestone", "importance": 0.9, "team": team, "player": pid, "evidence": {"stat": "hat-trick", "value": 3}})
        elif ev.type == "save" and T is not None:
            T["saves"] += 1
            P["saves"] += 1
        elif ev.type in ("tackle", "interception", "clearance") and T is not None:
            won = a.get("outcome", "won") == "won"
            T[ev.type + "s"] += 1
            T["def_actions"] += won
            P[ev.type + "s"] += 1
            own_x = ev.x if team == self.home else 105 - (ev.x or 0)
            if own_x is not None and own_x > 40:
                T["high_def_actions"] += 1
            if ev.type == "tackle" and won:
                P["tackles_won"] += 1
        elif ev.type == "pressure" and T is not None:
            T["pressures"] += 1
            P["pressures"] += 1
        elif ev.type == "card" and T is not None:
            if a.get("card") == "red":
                self.reds[team] += 1
                self.on_pitch.discard(pid)
            signals.append({"kind": "card", "importance": 0.8 if a.get("card") == "red" else 0.35, "team": team, "player": pid,
                            "evidence": {"card": a.get("card")}})
        elif ev.type == "substitution" and T is not None:
            self.on_pitch.discard(pid)
            self.on_pitch.add(a.get("replacement_id"))
            self.player[a.get("replacement_id")]["on_at"] = ev.t
            self.player[pid]["off_at"] = ev.t
            signals.append({"kind": "substitution", "importance": 0.2, "team": team, "player": pid,
                            "evidence": {"replacement": a.get("replacement_name")}})
        elif ev.type == "corner" and T is not None:
            T["corners"] += 1
            self.threat[team] += 0.03
        elif ev.type == "foul" and T is not None:
            T["fouls"] += 1
        elif ev.type in ("half_time", "full_time"):
            signals.append({"kind": ev.type, "importance": 0.9, "team": None, "player": None,
                            "evidence": {"score": dict(self.score)}})

        # distance indicator (synthetic: running estimate from minutes on pitch + carries)
        if pid and pid in self.player_team:
            km = self.distance_km(pid)
            step = int(km // DISTANCE_STEP_KM)
            if step > self.player[pid]["dist_step"]:
                self.player[pid]["dist_step"] = step
                signals.append({"kind": "distance_tag", "importance": 0.15, "team": self.player_team[pid], "player": pid,
                                "evidence": {"distance_km": round(km, 1)}})

        # flow shifts (checked every 20 events to keep it cheap)
        if self.n_events % 20 == 0 and self.t > 240:
            signals += self._flow_signals()
        mom = self.momentum()
        if not self.momentum_history or ev.t - self.momentum_history[-1][0] >= 30:
            self.momentum_history.append((ev.t, round(mom, 3)))
            wp = self.win_prob()
            self.wp_history.append((ev.t, {k: round(v, 3) for k, v in wp.items()}))
            if len(self.momentum_history) >= 6:
                old = self.momentum_history[-6][1]
                if abs(mom - old) > 0.8 and (mom > 0) != (old > 0) and ev.t - self.last_swing_t > 420:
                    self.last_swing_t = ev.t
                    leader = self.home if mom > 0 else self.away
                    signals.append({"kind": "momentum_swing", "importance": 0.5, "team": leader, "player": None,
                                    "evidence": {"momentum_before": old, "momentum_now": round(mom, 3),
                                                 "window_s": round(ev.t - self.momentum_history[-6][0])}})
        for s in signals:
            s.setdefault("t", ev.t)
            s.setdefault("clock", self.clock())
        return signals

    def distance_km(self, pid: str) -> float:
        on = self.player[pid].get("on_at", 0.0 if pid in self.on_pitch or "off_at" in self.player[pid] else self.t)
        off = self.player[pid].get("off_at", self.t)
        mins = max(0.0, off - on) / 60.0
        base = 0.105 if self.positions.get(pid) != "GK" else 0.055
        return mins * base * (0.9 + 0.25 * self.skill.get(pid, 0.6)) + self.player[pid]["carry_m"] / 1000.0

    # ------------------------------------------------------------------
    def flow(self) -> dict:
        w = list(self.window)
        span = max(60.0, (w[-1][0] - w[0][0]) if len(w) > 1 else 60.0) / 60.0
        out = {}
        for tm in (self.home, self.away):
            passes = [x for x in w if x[1] == "pass" and x[2] == tm]
            comp = sum(1 for x in passes if x[3] == "complete") / len(passes) if passes else 0.0
            seqs = [x[4] for x in w if x[1] == "possession_change" and x[2] != tm and x[4]]
            seq = sum(seqs) / len(seqs) if seqs else 0.0
            press = sum(1 for x in w if x[1] == "pressure" and x[2] == tm) / span
            out[tm] = {"completion": comp, "avg_sequence": seq, "pressures_per_min": press, "passes": len(passes)}
        turnovers = sum(1 for x in w if x[1] == "possession_change") / span
        duels = sum(1 for x in w if x[1] in ("tackle", "foul")) / span
        tempo = sum(1 for x in w if x[1] in ("pass", "carry", "shot")) / span
        ctrl = {tm: min(100.0, 100 * (0.5 * out[tm]["completion"] + 0.5 * min(1.0, out[tm]["avg_sequence"] / 8.0))
                    * (out[tm]["passes"] / max(1, out[self.home]["passes"] + out[self.away]["passes"])) * 2) for tm in out}
        chaos = min(100.0, 100 * (turnovers / 9.0) * 0.7 + 100 * (duels / 2.0) * 0.3)
        leader = max(ctrl, key=ctrl.get)
        if chaos > 62:
            label = "chaos"
        elif ctrl[leader] > 58:
            label = f"control:{leader}"
        else:
            label = "balanced"
        return {"control": {k: round(v, 1) for k, v in ctrl.items()}, "chaos": round(chaos, 1), "label": label,
                "tempo_per_min": round(tempo, 1), "turnovers_per_min": round(turnovers, 2),
                "pressing": {k: round(v["pressures_per_min"], 2) for k, v in out.items()},
                "completion": {k: round(100 * v["completion"], 1) for k, v in out.items()}}

    def _flow_signals(self) -> list[dict]:
        f = self.flow()
        sig = []
        if f["label"] != self.flow_label:
            sig.append({"kind": "flow_shift", "importance": 0.45, "team": f["label"].split(":")[1] if ":" in f["label"] else None,
                        "player": None, "evidence": {"from": self.flow_label, "to": f["label"], "chaos": f["chaos"],
                                                     "control": f["control"], "turnovers_per_min": f["turnovers_per_min"],
                                                     "tempo_per_min": f["tempo_per_min"]}})
            self.flow_label = f["label"]
        pr = f["pressing"]
        lead = max(pr, key=pr.get)
        if pr[lead] > 1.25 * max(pr[self.other(lead)], 0.1) and lead != self.press_leader:
            sig.append({"kind": "pressure_shift", "importance": 0.4, "team": lead, "player": None,
                        "evidence": {"pressures_per_min": pr[lead], "opponent_pressures_per_min": pr[self.other(lead)]}})
            self.press_leader = lead
        return sig

    # ------------------------------------------------------------------
    def team_summary(self, tm: str) -> dict:
        T = self.team[tm]
        tot_poss = self.team[self.home]["poss_time"] + self.team[self.away]["poss_time"]
        opp = self.team[self.other(tm)]
        ppda = opp["passes_own60"] / max(1.0, T["high_def_actions"])
        return {
            "goals": self.score[tm], "possession_pct": round(100 * T["poss_time"] / tot_poss, 1) if tot_poss else 50.0,
            "passes": int(T["passes"]), "pass_accuracy_pct": round(100 * T["passes_completed"] / T["passes"], 1) if T["passes"] else 0.0,
            "avg_pass_length_m": round(T["pass_length"] / T["passes"], 1) if T["passes"] else 0.0,
            "avg_pass_difficulty": round(T["pass_difficulty"] / T["passes"], 3) if T["passes"] else 0.0,
            "passes_over_expected": round(T["passes_completed"] - T["xpass"], 1),
            "difficult_completed": int(T["difficult_completed"]), "progressive_passes": int(T["progressive"]),
            "final_third_passes": int(T["final_third_passes"]),
            "shots": int(T["shots"]), "on_target": int(T["on_target"]), "xg": round(T["xg"], 2),
            "max_shot_speed_kmh": round(T["shot_speed_max"], 1),
            "tackles": int(T["tackles"]), "interceptions": int(T["interceptions"]), "pressures": int(T["pressures"]),
            "ppda": round(ppda, 1), "corners": int(T["corners"]), "fouls": int(T["fouls"]), "saves": int(T["saves"]),
        }

    def player_summary(self, pid: str) -> dict:
        P = self.player[pid]
        return {"id": pid, "name": self.names.get(pid, pid), "team": self.player_team.get(pid), "position": self.positions.get(pid),
                "touches": int(P["touches"]), "passes": int(P["passes"]),
                "pass_accuracy_pct": round(100 * P["passes_completed"] / P["passes"], 1) if P["passes"] else 0.0,
                "passes_over_expected": round(P["passes_completed"] - P["xpass"], 1), "key_passes": int(P["key_passes"]),
                "shots": int(P["shots"]), "goals": int(P["goals"]), "xg": round(P["xg"], 2),
                "tackles_won": int(P["tackles_won"]), "interceptions": int(P["interceptions"]), "pressures": int(P["pressures"]),
                "saves": int(P["saves"]), "distance_km": round(self.distance_km(pid), 1), "top_speed_kmh": round(P["top_speed"], 1)}

    def top_players(self, n: int = 3) -> list[dict]:
        def rating(pid):
            P = self.player[pid]
            return P["goals"] * 3 + P["xg"] * 2 + P["key_passes"] + 0.03 * P["passes_completed"] + 0.4 * P["tackles_won"] + 0.3 * P["interceptions"] + 0.5 * P["saves"]
        ids = [p for p in self.player if p in self.player_team]
        return [self.player_summary(p) | {"impact": round(rating(p), 2)} for p in sorted(ids, key=rating, reverse=True)[:n]]

    def snapshot(self) -> dict:
        return {"t": round(self.t, 1), "clock": self.clock(), "period": self.period, "score": dict(self.score),
                "possession_team": self.poss_team, "momentum": round(self.momentum(), 3),
                "win_probability": {k: round(v, 3) for k, v in self.win_prob().items()},
                "teams": {tm: self.team_summary(tm) for tm in (self.home, self.away)}, "flow": self.flow()}
