"""Synthetic, football-realistic match generator (no real Premier League data).

Produces a provider-style event feed (passes, carries, pressures, tackles,
interceptions, shots with freeze frames, fouls, cards, corners, subs, metadata) for
two fictional clubs. A latent momentum process drives control/chaos phases so there is
real rhythm to explain.

Data innovation: every on-target shot carries a 360-style freeze frame plus a short
tracking snippet (shooter run-up, keeper dive) and the keeper's reaction delay. Goals vs
saves come from a *planted* keeper model, so the shot autopsy can be evaluated against a
known truth, exactly like the chase simulator.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0
GOAL_Y = (PITCH_W / 2 - 3.66, PITCH_W / 2 + 3.66)

CLUBS = [
    {"id": "NBR", "name": "Northbridge FC", "short": "Northbridge", "colors": ["#d7263d", "#ffffff"]},
    {"id": "RVA", "name": "Riverside Athletic", "short": "Riverside", "colors": ["#1b98e0", "#0b1f3a"]},
    {"id": "HLU", "name": "Harrowgate Lions", "short": "Harrowgate", "colors": ["#f4a259", "#1f1f1f"]},
    {"id": "KSW", "name": "Kingsmere Wanderers", "short": "Kingsmere", "colors": ["#2ec4b6", "#011627"]},
    {"id": "ASH", "name": "Ashford Rovers", "short": "Ashford", "colors": ["#8338ec", "#ffbe0b"]},
    {"id": "PTC", "name": "Portcullis City", "short": "Portcullis", "colors": ["#3a86ff", "#fb5607"]},
]
FIRST = ["Leo", "Kai", "Mateo", "Idris", "Noah", "Elio", "Tomas", "Rafa", "Yusuf", "Arlo", "Joon", "Femi", "Luka",
         "Ravi", "Owen", "Nico", "Sami", "Bram", "Ciro", "Dario", "Ezra", "Felix", "Hugo", "Iker", "Jonah", "Kofi",
         "Lars", "Milo", "Nils", "Otto", "Pavel", "Quinn", "Remy", "Soren", "Teo", "Umar", "Viggo", "Wes", "Xavi", "Zane"]
LAST = ["Ashcombe", "Brennan", "Calloway", "Duarte", "Eklund", "Farrow", "Galvan", "Hartley", "Iwobi", "Janssen",
        "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Pereira", "Quintero", "Rasmussen", "Silva-Kent",
        "Thorne", "Ueda", "Varga", "Whitlock", "Xhaka-Mohr", "Yilmaz", "Zielinski", "Abara", "Blythe", "Cormac", "Delgado"]
FORMATION = [("GK", 5, 34), ("RB", 30, 8), ("CB", 22, 25), ("CB", 22, 43), ("LB", 30, 60), ("DM", 42, 34),
             ("CM", 52, 22), ("CM", 52, 46), ("RW", 72, 10), ("ST", 80, 34), ("LW", 72, 58)]


@dataclass
class Player:
    id: str
    name: str
    team: str
    number: int
    position: str
    bx: float
    by: float
    skill: float
    pace: float


def _name(rng, used):
    while True:
        n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if n not in used:
            used.add(n)
            return n


def make_squad(club: dict, rng, used) -> list[Player]:
    ps = []
    nums = rng.permutation(np.arange(2, 40))[:16]
    for k, (pos, x, y) in enumerate(FORMATION + [("CB", 22, 34), ("CM", 52, 34), ("RW", 72, 12), ("ST", 80, 30), ("GK", 5, 34)]):
        ps.append(Player(id=f"{club['id']}{k:02d}", name=_name(rng, used), team=club["id"], number=1 if k == 0 else int(nums[k]),
                         position=pos, bx=x, by=y, skill=float(rng.uniform(0.45, 0.9)), pace=float(rng.uniform(29, 35.5))))
    return ps


def xg_model(x: float, y: float) -> float:
    """Simple location xG (attacking toward x = 105)."""
    dx, dy = PITCH_L - x, abs(y - PITCH_W / 2)
    dist = math.hypot(dx, dy)
    a = math.atan2(7.32 * dx, dx * dx + dy * dy - (7.32 / 2) ** 2)
    a = a if a > 0 else a + math.pi
    z = -2.5 - 0.1 * dist + 1.4 * a
    return float(1 / (1 + math.exp(-z)))


# Planted keeper model: log-odds of a goal for an on-target shot. Coefficients are the
# ground truth the autopsy must recover (eval 1, football variant).
KEEPER_MODEL = {"intercept": -2.3, "reaction_delay": (0.36, 0.10, 1.3), "screening": (0.6, 1.0, 1.0),
                "approach_angle": (5.0, 6.0, 1.0), "separation_at_commit": (11.0, 6.0, -0.7)}


def keeper_terms(feats: dict) -> dict:
    out = {}
    for k, v in KEEPER_MODEL.items():
        if k == "intercept":
            continue
        mu, sd, w = v
        out[k] = w * (feats[k] - mu) / sd
    return out


class MatchGenerator:
    def __init__(self, seed: int = 7, home: str = "NBR", away: str = "RVA"):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        clubs = {c["id"]: c for c in CLUBS}
        self.home, self.away = clubs[home], clubs[away]
        used: set[str] = set()
        self.squads = {home: make_squad(self.home, self.rng, used), away: make_squad(self.away, self.rng, used)}
        self.on_pitch = {t: self.squads[t][:11] for t in self.squads}
        self.strength = {home: float(self.rng.uniform(0.48, 0.6)), away: float(self.rng.uniform(0.42, 0.55))}
        self.events: list[dict] = []
        self.n = 0

    # -------------------------------------------------------------------
    def _emit(self, t: float, period: int, type_: str, team: str | None, player: Player | None, **kw) -> dict:
        self.n += 1
        e = {"event_id": f"{self.seed}-{self.n:05d}", "match_id": f"SYN-{self.seed}", "period": period,
             "timestamp": round(t, 2), "minute": int(t // 60), "second": int(t % 60), "type": type_,
             "team_id": team, "player_id": player.id if player else None,
             "player_name": player.name if player else None}
        e.update(kw)
        self.events.append(e)
        return e

    def _loc(self, team: str, x: float, y: float) -> list[float]:
        """Generator works with each team attacking +x; the feed stores home-attacks-right coordinates."""
        if team == self.away["id"]:
            return [round(PITCH_L - x, 2), round(PITCH_W - y, 2)]
        return [round(x, 2), round(y, 2)]

    def _pick(self, team: str, x: float, exclude: Player | None = None, gk=False) -> Player:
        ps = [p for p in self.on_pitch[team] if p is not exclude and (gk or p.position != "GK")]
        w = np.array([math.exp(-abs(p.bx - x) / 18) * (0.6 + p.skill) for p in ps])
        return ps[int(self.rng.choice(len(ps), p=w / w.sum()))]

    def _gk(self, team: str) -> Player:
        return next(p for p in self.on_pitch[team] if p.position == "GK")

    # -------------------------------------------------------------------
    def shot_detail(self, x: float, y: float, shooter: Player, defending: str) -> dict:
        """Freeze frame + tracking snippet + planted outcome for an on-target shot (attacking +x)."""
        rng = self.rng
        gk = self._gk(defending)
        goal = np.array([PITCH_L, PITCH_W / 2])
        ball = np.array([x, y])
        to_goal = goal - ball
        dist_goal = float(np.linalg.norm(to_goal))
        # keeper off his line toward the ball
        kp = goal - to_goal / dist_goal * min(rng.uniform(1.0, 5.0), 0.45 * dist_goal) + rng.normal(0, 0.4, 2)
        # shot target inside the goal mouth
        target = np.array([PITCH_L, rng.uniform(GOAL_Y[0] + 0.3, GOAL_Y[1] - 0.3)])
        # defenders: some in the lane between ball and keeper (screening)
        n_screen = int(rng.choice([0, 0, 1, 1, 2, 3], p=[0.3, 0.15, 0.25, 0.1, 0.15, 0.05]))
        defenders = []
        for _ in range(n_screen):
            a = rng.uniform(0.25, 0.75)
            p = ball + (kp - ball) * a + rng.normal(0, 0.5, 2)
            defenders.append(p)
        for _ in range(int(rng.integers(1, 4))):
            defenders.append(ball + rng.normal(0, 7, 2))
        teammates = [ball + rng.normal([6, 0], 6, 2) for _ in range(int(rng.integers(0, 4)))]
        shot_speed = float(np.clip(rng.normal(78 + 0.6 * dist_goal, 10), 45, 125))  # km/h
        reaction = float(np.clip(rng.normal(0.36, 0.11) + 0.05 * n_screen, 0.12, 0.8))
        v1, v2 = kp - ball, target - ball
        angle = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
        sep = float(np.linalg.norm(kp - ball))
        feats = {"reaction_delay": reaction, "screening": float(n_screen), "approach_angle": angle, "separation_at_commit": sep}
        terms = keeper_terms(feats)
        z = KEEPER_MODEL["intercept"] + sum(terms.values())
        p_goal = 1 / (1 + math.exp(-z))
        goal_scored = bool(rng.random() < p_goal)
        # truth: the term(s) pushing hardest toward the outcome
        s = 1 if goal_scored else -1
        ranked = sorted(terms, key=lambda k: s * terms[k], reverse=True)
        causes = [ranked[0]] + ([ranked[1]] if s * terms[ranked[1]] > 0.8 * s * terms[ranked[0]] else [])
        # tracking snippet: 1.5 s run-up + ball flight; keeper dives after the reaction delay
        runup_dir = to_goal / dist_goal
        runup_speed = float(rng.uniform(3.5, 7.5))
        feints = int(rng.choice([0, 0, 1, 1, 2]))
        flight = dist_goal / (shot_speed / 3.6)
        t_strike = 1.5
        ts = np.round(np.arange(0, t_strike + flight + 0.4, 0.1), 3)
        shooter_tr, keeper_tr = [], []
        dive_dir = (target - kp) / (np.linalg.norm(target - kp) + 1e-9)
        dive_speed = float(rng.uniform(4.0, 6.5))
        for t in ts:
            if t <= t_strike:
                sp = ball - runup_dir * runup_speed * (t_strike - t)
            else:
                sp = ball + runup_dir * 1.5 * (t - t_strike)
            shooter_tr.append(sp)
            td = t - (t_strike + reaction)
            keeper_tr.append(kp + dive_dir * dive_speed * max(0.0, td) if td > 0 else kp + rng.normal(0, 0.03, 2))
        return {
            "keeper_id": gk.id, "keeper_name": gk.name, "keeper_location": kp.round(2).tolist(),
            "target": target.round(2).tolist(), "shot_speed_kmh": round(shot_speed, 1),
            "freeze_frame": [{"role": "defender", "location": d.round(2).tolist()} for d in defenders]
                            + [{"role": "teammate", "location": m.round(2).tolist()} for m in teammates],
            "keeper_reaction_s": round(reaction, 3), "screening_defenders": n_screen, "shot_angle_deg": round(angle, 1),
            "keeper_distance_m": round(sep, 2), "runup_speed_ms": round(runup_speed, 2), "feints": feints,
            "time_since_last_sprint_s": round(float(rng.uniform(2, 40)), 1), "ball_flight_s": round(float(flight), 3),
            "tracking": {"t": ts.tolist(), "shooter": np.array(shooter_tr).round(3).tolist(),
                         "keeper": np.array(keeper_tr).round(3).tolist()},
            "outcome_model": {"p_goal": round(p_goal, 3), "terms": {k: round(v, 3) for k, v in terms.items()},
                              "true_causes": causes},
            "is_goal": goal_scored,
        }

    # -------------------------------------------------------------------
    def generate(self) -> dict:
        rng = self.rng
        H, A = self.home["id"], self.away["id"]
        score = {H: 0, A: 0}
        momentum = 0.0  # >0 favours home
        cards: dict[str, int] = {}
        subs = {H: 0, A: 0}
        t, period = 0.0, 1
        poss = H
        self._emit(0, 1, "kickoff", H, None, location=[52.5, 34.0])
        x, y = 52.5, 34.0
        halves = [(1, 0.0, 45 * 60 + rng.integers(60, 240)), (2, 45 * 60.0, 90 * 60 + rng.integers(120, 360))]
        for period, start, end in halves:
            t = max(t, start)
            if period == 2:
                poss = A
                x, y = 52.5, 34.0
                self._emit(t, 2, "kickoff", A, None, location=[52.5, 34.0])
            while t < end:
                momentum = 0.97 * momentum + rng.normal(0, 0.12)
                opp = A if poss == H else H
                bias = (self.strength[poss] - self.strength[opp]) + (momentum if poss == H else -momentum) * 0.5
                press = float(np.clip(0.35 + 0.25 * (-bias) + rng.normal(0, 0.08), 0.05, 0.9))
                # ---- one possession sequence ----
                carrier = self._pick(poss, x)
                seq_len = 0
                while t < end:
                    seq_len += 1
                    t += float(rng.uniform(1.2, 3.2))
                    # opponent pressure
                    under_pressure = bool(rng.random() < press * (0.6 + 0.4 * x / PITCH_L))
                    if under_pressure:
                        presser = self._pick(opp, PITCH_L - x)
                        self._emit(t - 0.4, period, "pressure", opp, presser, location=self._loc(opp, PITCH_L - x, PITCH_W - y),
                                   on_player=carrier.id)
                    # shot?
                    xg = xg_model(x, y)
                    if x > 75 and rng.random() < 0.02 + 0.3 * xg + 0.01 * bias:
                        t = self._shot(t, period, poss, opp, carrier, x, y, xg, score)
                        poss = opp
                        x, y = 15.0, 34.0
                        break
                    # carry or pass
                    if rng.random() < 0.22:
                        dx = float(rng.uniform(3, 15))
                        nx, ny = min(x + dx, 100), float(np.clip(y + rng.normal(0, 5), 2, 66))
                        sp = float(np.clip(rng.normal(19, 4) + (8 if dx > 10 else 0), 8, carrier.pace))
                        self._emit(t, period, "carry", poss, carrier, location=self._loc(poss, x, y), end_location=self._loc(poss, nx, ny),
                                   distance_m=round(math.hypot(nx - x, ny - y), 1), speed_kmh=round(sp, 1), under_pressure=under_pressure)
                        if under_pressure and rng.random() < 0.28:
                            tackler = self._pick(opp, PITCH_L - nx)
                            won = rng.random() < 0.65
                            self._emit(t + 0.6, period, "tackle", opp, tackler, location=self._loc(opp, PITCH_L - nx, PITCH_W - ny),
                                       outcome="won" if won else "lost", on_player=carrier.id)
                            if won:
                                self._emit(t + 0.7, period, "possession_change", opp, tackler, location=self._loc(opp, PITCH_L - nx, PITCH_W - ny),
                                           reason="tackle", sequence_length=seq_len)
                                if rng.random() < 0.12:
                                    t = self._foul(t + 1, period, opp, tackler, poss, carrier, nx, cards)
                                poss, x, y = opp, PITCH_L - nx, PITCH_W - ny
                                break
                        x, y = nx, ny
                        continue
                    receiver = self._pick(poss, x + 12, exclude=carrier)
                    fwd = float(rng.normal(6 + 6 * bias, 11))
                    nx = float(np.clip(x + fwd, 3, 102))
                    ny = float(np.clip(receiver.by + rng.normal(0, 7), 1, 67))
                    length = math.hypot(nx - x, ny - y)
                    lane_block = float(rng.beta(2, 5))
                    difficulty = float(np.clip(0.04 + 0.011 * length + 0.2 * under_pressure + 0.3 * lane_block
                                               + 0.1 * (fwd > 10) + 0.1 * (nx > 85), 0, 1))
                    p_succ = float(np.clip(1.08 - 0.6 * difficulty + 0.25 * (carrier.skill - 0.65), 0.15, 0.98))
                    ok = bool(rng.random() < p_succ)
                    ball_speed = float(np.clip(rng.normal(38 + 1.1 * length, 8), 15, 110))
                    self._emit(t, period, "pass", poss, carrier, location=self._loc(poss, x, y), end_location=self._loc(poss, nx, ny),
                               recipient_id=receiver.id if ok else None, recipient_name=receiver.name if ok else None,
                               outcome="complete" if ok else "incomplete", length_m=round(length, 1), ball_speed_kmh=round(ball_speed, 1),
                               difficulty=round(difficulty, 3), expected_completion=round(p_succ, 3), under_pressure=under_pressure,
                               height="high" if length > 30 and rng.random() < 0.6 else "ground", progressive=bool(fwd > 10))
                    if ok:
                        carrier, x, y = receiver, nx, ny
                        if seq_len > 25:
                            seq_len = 0
                        continue
                    # turnover
                    winner = self._pick(opp, PITCH_L - nx)
                    u = rng.random()
                    kind = "interception" if u < 0.22 else ("clearance" if nx > 80 and u < 0.5 else "loose_ball")
                    if kind != "loose_ball":
                        self._emit(t + 0.8, period, kind, opp, winner, location=self._loc(opp, PITCH_L - nx, PITCH_W - ny))
                    self._emit(t + 0.9, period, "possession_change", opp, winner, location=self._loc(opp, PITCH_L - nx, PITCH_W - ny),
                               reason=kind, sequence_length=seq_len)
                    poss, x, y = opp, PITCH_L - nx, PITCH_W - ny
                    t += 0.9
                    break
                # substitutions around the hour
                if period == 2:
                    for tm in (H, A):
                        if subs[tm] < 3 and t > (58 + 9 * subs[tm]) * 60 and rng.random() < 0.3:
                            out_p = self.on_pitch[tm][int(rng.integers(6, len(self.on_pitch[tm])))]
                            bench = [p for p in self.squads[tm][11:] if p not in self.on_pitch[tm] and p.position != "GK"]
                            if bench:
                                in_p = bench[0]
                                self.on_pitch[tm] = [in_p if p is out_p else p for p in self.on_pitch[tm]]
                                in_p.bx, in_p.by = out_p.bx, out_p.by
                                self._emit(t, 2, "substitution", tm, out_p, replacement_id=in_p.id, replacement_name=in_p.name)
                                subs[tm] += 1
            self._emit(end, period, "half_time" if period == 1 else "full_time", None, None, score=dict(score))
            t = end
        meta = {
            "match_id": f"SYN-{self.seed}", "competition": "Synthetic Premier Division (fictional)", "season": "2026/27",
            "venue": f"{self.home['short']} Park", "kickoff_utc": "2026-10-24T14:00:00Z", "referee": "A. Synthetic",
            "home": self.home, "away": self.away, "pitch": {"length_m": PITCH_L, "width_m": PITCH_W},
            "coordinates": "metres, origin bottom-left, home attacks +x in both halves",
            "lineups": {tm: [p.__dict__ for p in self.squads[tm]] for tm in self.squads},
            "final_score": dict(score), "generator": {"name": "football/synth.py", "seed": self.seed},
            "synthetic": True,
        }
        return {"metadata": meta, "events": sorted(self.events, key=lambda e: e["timestamp"])}

    def _foul(self, t, period, team, player, victim_team, victim, x, cards):
        self._emit(t, period, "foul", team, player, on_player=victim.id, location=self._loc(team, PITCH_L - x, 34))
        if self.rng.random() < 0.3:
            cards[player.id] = cards.get(player.id, 0) + 1
            color = "yellow" if cards[player.id] == 1 else "red"
            self._emit(t + 5, period, "card", team, player, card=color)
            if color == "red":
                self.on_pitch[team] = [p for p in self.on_pitch[team] if p is not player]
        self._emit(t + 15, period, "free_kick", victim_team, victim, location=self._loc(victim_team, x, 34))
        return t + 15

    def _shot(self, t, period, team, opp, shooter, x, y, xg, score) -> float:
        rng = self.rng
        detail = self.shot_detail(x, y, shooter, opp)
        u = rng.random()
        if u < 0.22:
            outcome = "blocked"
        elif u < 0.52:
            outcome = "off_target"
        else:
            outcome = "goal" if detail["is_goal"] else "saved"
        e = self._emit(t, period, "shot", team, shooter, location=self._loc(team, x, y), xg=round(xg, 3),
                       outcome=outcome, shot_speed_kmh=detail["shot_speed_kmh"], body_part=str(rng.choice(["right_foot", "left_foot", "head"], p=[0.6, 0.3, 0.1])))
        if outcome in ("goal", "saved"):
            e["on_target"] = True
            # store the detail in attacking-+x coordinates plus the orientation flag
            e["detail"] = {k: v for k, v in detail.items() if k != "is_goal"}
            e["detail"]["frame"] = "attacking_right"
        if outcome == "goal":
            score[team] += 1
            self._emit(t + 1.0, period, "goal", team, shooter, score=dict(score), xg=round(xg, 3))
            self._emit(t + 60, period, "kickoff", opp, None, location=[52.5, 34.0])
            return t + 60
        if outcome == "saved":
            gk = self._gk(opp)
            self._emit(t + 0.8, period, "save", opp, gk, on_shot=e["event_id"])
            if rng.random() < 0.5:
                self._emit(t + 25, period, "corner", team, self._pick(team, 90))
                return t + 30
        if outcome == "blocked" and rng.random() < 0.4:
            self._emit(t + 20, period, "corner", team, self._pick(team, 90))
        return t + 6


def generate_match(seed: int = 7, home: str = "NBR", away: str = "RVA") -> dict:
    return MatchGenerator(seed, home, away).generate()


def generate_shots(n: int, seed: int = 0) -> list[dict]:
    """Standalone on-target shots (for training / evaluating the shot autopsy)."""
    g = MatchGenerator(seed)
    rng = g.rng
    out = []
    for i in range(n):
        x, y = float(rng.uniform(80, 101)), float(rng.uniform(14, 54))
        shooter = g._pick(g.home["id"], 85)
        d = g.shot_detail(x, y, shooter, g.away["id"])
        out.append({"event_id": f"shot-{seed}-{i:04d}", "type": "shot", "team_id": g.home["id"], "player_id": shooter.id,
                    "player_name": shooter.name, "location": [round(x, 2), round(y, 2)], "xg": round(xg_model(x, y), 3),
                    "outcome": "goal" if d["is_goal"] else "saved", "shot_speed_kmh": d["shot_speed_kmh"], "timestamp": 0.0,
                    "detail": {**{k: v for k, v in d.items() if k != "is_goal"}, "frame": "attacking_right"}})
    return out


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--home", default="NBR")
    ap.add_argument("--away", default="RVA")
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()
    m = generate_match(a.seed, a.home, a.away)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / f"match_{a.seed}.json"
    p.write_text(json.dumps(m, indent=1))
    print(p, len(m["events"]), "events, final", m["metadata"]["final_score"])


if __name__ == "__main__":  # pragma: no cover
    main()
