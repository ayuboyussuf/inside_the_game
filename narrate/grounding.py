"""Grounded-narration rule check (eval 9): every number in the text must appear in the evidence.

No LLM needed. A number in the text is supported if some number found anywhere in the
evidence JSON (values or display strings) matches it at the precision it was written,
or as a percentage of a probability.
"""
from __future__ import annotations

import re

NUM = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?")


def _walk(obj, out: list[float]):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in NUM.finditer(obj):
            out.append(_parse(m.group()))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, out)


def _parse(s: str) -> float:
    return float(s.replace(",", "."))


def _decimals(s: str) -> int:
    s = s.replace(",", ".")
    return len(s.split(".")[1]) if "." in s else 0


def allowed_numbers(evidence) -> list[float]:
    vals: list[float] = []
    _walk(evidence, vals)
    return vals


def supported(token: str, vals: list[float]) -> bool:
    x, d = _parse(token), _decimals(token)
    tol = 0.5 * 10 ** (-d) + 1e-9
    for v in vals:
        if abs(abs(v) - abs(x)) <= tol:
            return True
        if 0 <= v <= 1 and abs(100 * v - abs(x)) <= max(tol, 0.5):
            return True
    return False


def check(text: str, evidence, extra_allowed: tuple[float, ...] = ()) -> dict:
    vals = allowed_numbers(evidence) + list(extra_allowed)
    tokens = [m.group() for m in NUM.finditer(text)]
    bad = [t for t in tokens if not supported(t, vals)]
    return {"ok": not bad, "numbers": tokens, "unsupported": bad}
