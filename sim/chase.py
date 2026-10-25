"""Synthetic 2-D chase simulator with planted rules.

Each chase is simulated physically (stalk -> sprint -> prey reaction -> cuts) up to the
end of the decision window. The outcome is then decided by a *planted rule* applied to
the features the explainer will see, so the true cause of every clip is known. Finally
an outcome phase (lunge + contact, or abort) is appended after the window.

Units: predator body lengths (BL) and seconds, like the real-clip pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from events.features import extract_features
from events.schema import Actor, Episode, Event, Track

DT = 1 / 30


@dataclass
class Rule:
    name: str
    description: str
    # returns (outcome, cause features for this clip)
    decide: Callable[[dict], tuple[int, list[str]]]
    planted: list[str]
    pack: bool = False
    herd: bool = False


def _single(feat: str, thr: float, caught_if_below: bool) -> Callable[[dict], tuple[int, list[str]]]:
    def f(x: dict) -> tuple[int, list[str]]:
        below = x[feat] < thr
        return (int(below) if caught_if_below else int(not below)), [feat]
    return f


def _compound(x: dict) -> tuple[int, list[str]]:
    fast = x["reaction_delay"] < 0.6
    cuts = x["direction_changes"] >= 2
    if fast and cuts:
        return 0, ["reaction_delay", "direction_changes"]
    causes = ([] if fast else ["reaction_delay"]) + ([] if cuts else ["direction_changes"])
    return 1, causes


RULES: dict[str, Rule] = {
    "reaction": Rule("reaction", "prey escapes if reaction delay < 0.55 s",
                     lambda x: (int(x["reaction_delay"] >= 0.55), ["reaction_delay"]), ["reaction_delay"]),
    "separation": Rule("separation", "predator catches if it commits closer than 8 BL",
                       _single("separation_at_commit", 8.0, caught_if_below=True), ["separation_at_commit"]),
    "turns": Rule("turns", "prey escapes if it cuts at least twice",
                  lambda x: (int(x["direction_changes"] < 2), ["direction_changes"]), ["direction_changes"]),
    "compound": Rule("compound", "prey escapes if reaction delay < 0.6 s AND it turns at least twice",
                     _compound, ["reaction_delay", "direction_changes"]),
    "pack": Rule("pack", "pack catches if at least one helper is within 5 BL of the prey at the commit",
                 lambda x: (int(x["group_help"] >= 1), ["group_help"]), ["group_help"], pack=True),
    "screening": Rule("screening", "prey escapes if herd members screen it from the predator",
                      lambda x: (int(x["screening"] < 1), ["screening"]), ["screening"], herd=True),
}


@dataclass
class ChaseParams:
    sep0: float
    stalk_s: float
    chase_s: float
    vmax_pred: float
    vmax_prey: float
    reaction: float
    turn_times: list[float] = field(default_factory=list)
    turn_angles: list[float] = field(default_factory=list)
    n_pack: int = 0
    pack_near: bool = False
    n_herd: int = 0
    herd_in_line: bool = False


def sample_params(rng: np.random.Generator, rule: Rule) -> ChaseParams:
    chase_s = float(rng.uniform(2.6, 4.5))
    reaction = float(rng.uniform(0.15, 1.2))
    n_turns = int(rng.integers(0, 5))
    lo, hi = reaction + 0.4, chase_s - 0.5
    times = sorted(rng.uniform(lo, hi, n_turns).tolist()) if hi > lo else []
    # keep turns at least 0.5 s apart so each is detectable
    kept: list[float] = []
    for t in times:
        if not kept or t - kept[-1] >= 0.5:
            kept.append(t)
    return ChaseParams(
        sep0=float(rng.uniform(4.0, 14.0)),
        stalk_s=float(rng.uniform(0.6, 1.8)),
        chase_s=chase_s,
        vmax_pred=float(rng.uniform(14.0, 20.0)),
        vmax_prey=float(rng.uniform(11.0, 17.0)),
        reaction=reaction,
        turn_times=kept,
        turn_angles=[float(rng.uniform(40, 100) * rng.choice([-1, 1])) for _ in kept],
        n_pack=int(rng.integers(1, 3)) if rule.pack else 0,
        pack_near=bool(rng.random() < 0.5),
        n_herd=int(rng.integers(1, 4)) if rule.herd else int(rng.integers(0, 2)),
        herd_in_line=bool(rng.random() < 0.5) if rule.herd else False,
    )


def _rot(v: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def simulate(episode_id: str, rule: Rule, rng: np.random.Generator, label_noise: float = 0.05,
             pos_noise: float = 0.02) -> Episode:
    p = sample_params(rng, rule)
    theta = rng.uniform(0, 2 * np.pi)
    prey = np.array([0.0, 0.0])
    pred = prey - p.sep0 * np.array([math.cos(theta), math.sin(theta)]) - p.stalk_s * 0.8 * np.array([math.cos(theta), math.sin(theta)])
    prey_head = _unit(_rot(np.array([math.cos(theta), math.sin(theta)]), rng.uniform(-60, 60)))
    vp, vq = 0.0, 0.0
    pred_head = _unit(prey - pred)
    a_pred, a_prey, turn_rate_pred = 11.0, 10.0, 220.0  # BL/s^2, BL/s^2, deg/s

    t_commit = p.stalk_s
    t_react = t_commit + p.reaction
    t_pre = t_commit + p.chase_s
    n_steps = int(round(t_pre / DT)) + 1

    # support actors
    pack: list[np.ndarray] = []
    for k in range(p.n_pack):
        side = _rot(prey_head, 90 if k % 2 == 0 else -90)
        r = rng.uniform(2.0, 4.5) if p.pack_near else rng.uniform(7.0, 12.0)
        pack.append(prey + side * r + prey_head * rng.uniform(-1, 3))
    herd: list[np.ndarray] = []
    for k in range(p.n_herd):
        if p.herd_in_line:
            herd.append(prey + (pred - prey) * rng.uniform(0.25, 0.7) + _rot(_unit(pred - prey), 90) * rng.normal(0, 0.25))
        else:
            herd.append(prey + _rot(_unit(pred - prey), rng.choice([90, -90])) * rng.uniform(3, 8))

    T, P, Q = [], [], []
    PK = [[] for _ in pack]
    HD = [[] for _ in herd]
    turn_i = 0
    turning: list[tuple[float, float, float]] = []  # (t0, t1, deg/s)
    for i in range(n_steps):
        t = i * DT
        T.append(t)
        P.append(pred.copy())
        Q.append(prey.copy())
        # predator: stalk then sprint, pure pursuit with limited turn rate
        target_speed = 0.8 if t < t_commit else p.vmax_pred
        vp = min(target_speed, vp + a_pred * DT) if vp < target_speed else max(target_speed, vp - a_pred * DT)
        want = _unit(prey + prey_head * vq * 0.15 - pred)
        ang = math.degrees(math.atan2(pred_head[0] * want[1] - pred_head[1] * want[0], float(np.dot(pred_head, want))))
        pred_head = _unit(_rot(pred_head, float(np.clip(ang, -turn_rate_pred * DT, turn_rate_pred * DT))))
        pred = pred + pred_head * vp * DT
        # prey: graze, then flee away from predator after reacting, executing cuts
        if t >= t_react:
            if turn_i < len(p.turn_times) and t - t_commit >= p.turn_times[turn_i]:
                dur = 0.15
                turning.append((t, t + dur, p.turn_angles[turn_i] / dur))
                turn_i += 1
            active = [r for r in turning if r[0] <= t < r[1]]
            if active:
                prey_head = _unit(_rot(prey_head, active[0][2] * DT))
            elif not turning:
                away = _unit(prey - pred)
                prey_head = _unit(prey_head + 0.6 * (away - prey_head))
            vq = min(p.vmax_prey, vq + a_prey * DT)
        else:
            vq = max(0.0, vq - 2 * DT) if vq > 0.25 else 0.25
        prey = prey + prey_head * vq * DT
        for k in range(len(pack)):
            # pack mates converge on the prey but slower
            pack[k] = pack[k] + _unit(prey - pack[k]) * (min(9.0, 3 + t) if t >= t_commit else 0.3) * DT
            PK[k].append(pack[k].copy())
        for k in range(len(herd)):
            herd[k] = herd[k] + prey_head * (0.5 * vq) * DT if t >= t_react else herd[k]
            HD[k].append(herd[k].copy())

    events = [Event(t=0.0, type="track_start", actor="pred"), Event(t=0.0, type="track_start", actor="prey"),
              Event(t=0.0, type="stalk", actor="pred"),
              Event(t=round(t_commit, 4), type="sprint_start", actor="pred", x=float(P[int(t_commit / DT)][0]), y=float(P[int(t_commit / DT)][1])),
              Event(t=round(t_react, 4), type="flee_start", actor="prey")]
    for tt, aa in zip(p.turn_times, p.turn_angles):
        events.append(Event(t=round(t_commit + tt, 4), type="turn", actor="prey", attrs={"angle": round(aa, 1)}))

    actors = [Actor(id="pred", role="attacker", species="predator"), Actor(id="prey", role="defender", species="prey")]
    actors += [Actor(id=f"pack{k}", role="attacker_support", species="predator") for k in range(len(pack))]
    actors += [Actor(id=f"herd{k}", role="defender_support", species="prey") for k in range(len(herd))]

    def mk(aid, arr):
        a = np.asarray(arr) + rng.normal(0, pos_noise, (len(arr), 2))
        return Track(actor=aid, t=[round(x, 5) for x in T], x=a[:, 0].round(4).tolist(), y=a[:, 1].round(4).tolist())

    tracks = [mk("pred", P), mk("prey", Q)] + [mk(f"pack{k}", PK[k]) for k in range(len(pack))] + [mk(f"herd{k}", HD[k]) for k in range(len(herd))]
    pre = Episode(episode_id=episode_id, domain="sim", outcome=0, outcome_label="escaped", actors=actors,
                  events=events, tracks=tracks, fps=1 / DT)
    feats = extract_features(pre)
    outcome, causes = rule.decide(feats)
    true_outcome = outcome
    if rng.random() < label_noise:
        outcome = 1 - outcome
    # ---- outcome phase (after the decision window) ----------------------
    t_out = t_pre + 0.3
    Pp, Qq = np.array(P[-1]), np.array(Q[-1])
    extra_t = [t_pre + DT * (k + 1) for k in range(int(0.8 / DT))]
    for tr in tracks:
        lx, ly = tr.x[-1], tr.y[-1]
        v_last = np.array([tr.x[-1] - tr.x[-2], tr.y[-1] - tr.y[-2]])  # per-step displacement before the outcome phase
        for k, tt in enumerate(extra_t):
            if tr.actor == "pred":
                if outcome:
                    a = min(1.0, (k + 1) * DT / 0.3)
                    pos = Pp + (Qq - Pp) * a
                else:
                    # decelerate to a stop over 0.8 s
                    steps = sum(max(0.0, 1 - (j + 1) * DT / 0.8) for j in range(k + 1))
                    pos = Pp + v_last * steps
                tr.t.append(round(tt, 5)); tr.x.append(float(pos[0])); tr.y.append(float(pos[1]))
            else:
                tr.t.append(round(tt, 5)); tr.x.append(lx); tr.y.append(ly)
    events.append(Event(t=round(t_out, 4), type="contact" if outcome else "abort", actor="pred", target="prey"))
    decisive = {"reaction_delay": t_react,
                "direction_changes": t_commit + (p.turn_times[1] if len(p.turn_times) >= 2 else (p.turn_times[0] if p.turn_times else p.reaction + 0.3)),
                "separation_at_commit": t_commit, "group_help": t_commit, "screening": t_commit + 0.5 * p.chase_s}
    return Episode(
        episode_id=episode_id, domain="sim", outcome=outcome, outcome_label="caught" if outcome else "escaped",
        actors=actors, events=sorted(events, key=lambda e: e.t), tracks=tracks, fps=1 / DT,
        meta={"rule": rule.name, "rule_description": rule.description, "planted": rule.planted,
              "true_causes": causes, "true_outcome": true_outcome, "label_flipped": outcome != true_outcome,
              "decisive_t": round(float(decisive[causes[0]]), 3), "params": p.__dict__,
              "source": "sim/chase.py"},
    )


def generate(n: int, rule: str | Rule = "compound", seed: int = 0, label_noise: float = 0.05) -> list[Episode]:
    r = RULES[rule] if isinstance(rule, str) else rule
    rng = np.random.default_rng(seed)
    return [simulate(f"sim-{r.name}-{seed}-{i:04d}", r, rng, label_noise=label_noise) for i in range(n)]
