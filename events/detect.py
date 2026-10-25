"""Turn (camera-compensated) tracks into named events: sprint_start, flee_start, turn, contact/abort.

Used for real clips (after perception) and checked against the simulator's ground-truth
events in eval 7.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .schema import Event


def smooth(pos: np.ndarray, fps: float, window_s: float = 0.25) -> np.ndarray:
    w = max(5, int(round(window_s * fps)) | 1)
    if len(pos) <= w:
        return pos
    return savgol_filter(pos, w, 2, axis=0)


def _speed(pos, t):
    return np.linalg.norm(np.gradient(pos, t, axis=0), axis=1)


def onset(s: np.ndarray, t: np.ndarray, start: int = 0, rise: float = 0.10, sustain_s: float = 0.3,
          min_peak: float = 1.0, final: bool = False) -> int | None:
    """Index where speed starts its main rise: crossing baseline + rise*(peak-baseline), sustained.
    final=True: the onset of the LAST sustained fast run (an attacker's committed sprint after stalking spurts);
    final=False: the FIRST one (a defender's first reaction)."""
    if start >= len(s) - 2:
        return None
    seg = s[start:]
    peak = float(np.max(seg))
    base = float(np.median(seg[: max(3, len(seg) // 10)]))
    if peak - base < min_peak:
        return None
    thr = base + rise * (peak - base)
    hi = base + 0.5 * (peak - base)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    need = max(1, int(sustain_s / dt))
    if not final:
        above = seg > thr
        for i in range(len(seg) - need):
            if above[i:i + need].all() and np.any(seg[i:] > hi):
                # a threshold crossing always lags the true onset: walk back to the speed trough where the rise began
                j = i
                while j > 0 and seg[j - 1] < seg[j] - 1e-3 * (peak - base):
                    j -= 1
                return start + j
        return None
    # the commitment is the FINAL sprint: the last sustained run above half of peak speed (a stalking predator
    # makes small earlier spurts that are not the attack)
    fast = seg > hi
    runs, i = [], 0
    while i < len(seg):
        if fast[i]:
            j = i
            while j < len(seg) and fast[j]:
                j += 1
            if j - i >= need:
                runs.append(i)
            i = j
        else:
            i += 1
    if not runs:
        return None
    i = runs[-1]
    while i > 0 and seg[i - 1] > thr:  # back to where speed left the baseline band...
        i -= 1
    j = i  # ...then a threshold crossing always lags the true onset: walk back to the trough where the rise began
    while j > 0 and seg[j - 1] < seg[j] - 1e-3 * (peak - base):
        j -= 1
    return start + j


def detect_turns(pos: np.ndarray, t: np.ndarray, actor: str, start: int, end: int,
                 min_angle: float = 35.0, window_s: float = 0.3, min_gap_s: float = 0.4) -> list[Event]:
    v = np.gradient(pos, t, axis=0)
    s = np.linalg.norm(v, axis=1)
    head = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    dt = float(np.median(np.diff(t)))
    k = max(1, int(window_s / dt / 2))
    moving = s > 0.3 * (s[start:end].max() if end > start else s.max())
    out: list[Event] = []
    last_t = -1e9
    i = start + k
    while i < end - k:
        dh = np.degrees(head[i + k] - head[i - k])
        if moving[i] and abs(dh) >= min_angle and t[i] - last_t >= min_gap_s:
            # refine to the peak of turn rate inside the window
            j0, j1 = i - k, min(end - k, i + 2 * k)
            rates = [abs(head[j + k] - head[j - k]) for j in range(j0, j1)]
            j = j0 + int(np.argmax(rates))
            dh = float(np.degrees(head[min(j + k, len(head) - 1)] - head[max(j - k, 0)]))
            out.append(Event(t=float(t[j]), type="turn", actor=actor, x=float(pos[j, 0]), y=float(pos[j, 1]),
                             attrs={"angle": round(dh, 1)}))
            last_t = t[j]
            i = j + 2 * k
        else:
            i += 1
    return out


def detect_events(t: np.ndarray, att_pos: np.ndarray, def_pos: np.ndarray, att_id: str, def_id: str,
                  fps: float = 30.0, contact_dist: float = 0.6) -> list[Event]:
    t = np.asarray(t, float)
    pa, pd_ = smooth(np.asarray(att_pos, float), fps), smooth(np.asarray(def_pos, float), fps)
    sa, sd = _speed(pa, t), _speed(pd_, t)
    ev = [Event(t=float(t[0]), type="track_start", actor=att_id), Event(t=float(t[0]), type="track_start", actor=def_id)]
    i_sp = onset(sa, t, final=True)
    if i_sp is None:
        return ev
    ev.append(Event(t=float(t[i_sp]), type="sprint_start", actor=att_id, x=float(pa[i_sp, 0]), y=float(pa[i_sp, 1])))
    i_fl = onset(sd, t, start=max(0, i_sp - int(0.2 * fps)))
    if i_fl is not None:
        ev.append(Event(t=float(t[i_fl]), type="flee_start", actor=def_id, x=float(pd_[i_fl, 0]), y=float(pd_[i_fl, 1])))
    dist = np.linalg.norm(pa - pd_, axis=1)
    hit = np.nonzero(dist < contact_dist)[0]
    hit = hit[hit > i_sp]
    if len(hit):  # start of the LAST contiguous close-contact run (pass-bys earlier in a chase are not catches)
        breaks = np.nonzero(np.diff(hit) > 1)[0]
        hit = hit[breaks[-1] + 1:] if len(breaks) else hit
    end = int(hit[0]) if len(hit) else len(t) - 1
    ev += detect_turns(pd_, t, def_id, start=i_fl if i_fl is not None else i_sp, end=end)
    if len(hit):
        ev.append(Event(t=float(t[end]), type="contact", actor=att_id, target=def_id))
    else:
        peak = sa[i_sp:].max()
        slow = np.nonzero((sa < 0.35 * peak) & (np.arange(len(t)) > i_sp + int(0.5 * fps)))[0]
        if len(slow):
            ev.append(Event(t=float(t[slow[0]]), type="abort", actor=att_id))
    return sorted(ev, key=lambda e: e.t)
