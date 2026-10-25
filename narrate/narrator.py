"""Narrator: evidence JSON -> 2-3 sentences, citing only what's in it.

Generator/critic loop: the LLM drafts, ``grounding.check`` (the fact-check agent's tool)
rejects any unsupported number, the draft is retried with the offending numbers named,
and after two failures the deterministic template is used. Templates are grounded by
construction.
"""
from __future__ import annotations

import json

from .grounding import check
from .llm import LLM, get_llm
from .templates import AUTOPSY, FEATURE_WORDS, LANG_NAMES, LIVE, OUTCOME_WORDS, SIMILAR_SHORT, WORDS, fill


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# ---------------------------------------------------------------- autopsy
def template_autopsy(ev: dict, persona: str = "analyst", lang: str = "en") -> str:
    L = AUTOPSY.get(lang, AUTOPSY["en"])
    ow = OUTCOME_WORDS.get(lang, OUTCOME_WORDS["en"])
    fw = FEATURE_WORDS.get(lang, {})
    f = ev["top_factors"]
    if not f:
        return _cap(ow.get(ev["outcome"], ev["outcome"])) + "."
    f1 = f[0]
    ctx = {"outcome": ow.get(ev["outcome"], ev["outcome"]), "outcome_cap": _cap(ow.get(ev["outcome"], ev["outcome"])),
           "f1_label_l": fw.get(f1["feature"], f1["label"].lower()), "f1_display": f1["display"]}
    m = ev["decisive_moments"][0] if ev.get("decisive_moments") else None
    if persona == "casual":
        ctx["moment_short"] = fill(L["moment_short"], {"t_display": m["t_display"]}) if m else ""
        return fill(L["casual"], ctx)
    text = fill(L["lead"], ctx)
    sim = f1.get("similar")
    if sim:
        text += fill(L["similar"], {"similar": sim["display"], "similar_short": fill(SIMILAR_SHORT.get(lang, SIMILAR_SHORT["en"]), sim)})
    else:
        text += L["nosimilar"]
    if len(f) > 1 and f[1]["contribution"] > 0:
        text += fill(L["second"], {"f2_label_cap": _cap(fw.get(f[1]["feature"], f[1]["label"].lower())), "f2_display": f[1]["display"],
                                   "outcome": ctx["outcome"]})
    if m:
        text += fill(L["moment"], {"moment": m["display"], "t_display": m["t_display"]})
    if ev.get("counterfactual") and L.get("cf"):
        text += fill(L["cf"], {"cf": ev["counterfactual"]["display"]})
    return text


SYSTEM = """You are the narrator of "Hunt Autopsy", an explainable sports-analysis system.
You receive an evidence JSON produced by a model and turn it into broadcast copy.
Hard rules:
1. Use ONLY facts in the JSON. Every number or timestamp you write must appear in the JSON
   (probabilities may be written as whole percentages). Never invent numbers, names or events.
2. Say "explanatory factor", never claim proof of causation.
3. Write in {language}.
4. Persona: {persona_rule}
Return only the narration text."""
PERSONA_RULES = {
    "analyst": "data-hungry analyst: 2-3 sentences, cite the top factor, its value, the similar-clips evidence and the decisive timestamp.",
    "casual": "casual fan: 1-2 short, vivid sentences, at most two numbers, no jargon.",
    "kids": "young fan: one friendly sentence, at most one number.",
}


def llm_text(llm: LLM, evidence: dict, persona: str, lang: str, kind: str = "autopsy", max_attempts: int = 2) -> dict:
    system = SYSTEM.format(language=LANG_NAMES.get(lang, lang), persona_rule=PERSONA_RULES.get(persona, PERSONA_RULES["analyst"]))
    user = f"Task: {kind}.\nEvidence JSON:\n{json.dumps(evidence, ensure_ascii=False)}"
    attempts = []
    for _ in range(max_attempts):
        text = llm.chat(system, user)
        g = check(text, evidence)
        attempts.append({"text": text, "grounding": g})
        if g["ok"]:
            return {"text": text, "source": llm.provider, "grounding": g, "attempts": attempts}
        user += (f"\n\nYour previous draft used numbers that are NOT in the evidence: {g['unsupported']}. "
                 "Rewrite it using only numbers from the evidence.")
    return {"text": None, "source": llm.provider, "grounding": attempts[-1]["grounding"], "attempts": attempts}


def narrate_autopsy(evidence: dict, persona: str = "analyst", lang: str = "en", use_llm: bool = True) -> dict:
    llm = get_llm() if use_llm else None
    if llm is not None:
        try:
            r = llm_text(llm, evidence, persona, lang)
            if r["text"]:
                return r
        except Exception as e:  # network / quota: fall through to template
            evidence = {**evidence}
            _ = e
    text = template_autopsy(evidence, persona, lang if lang in AUTOPSY else "en")
    return {"text": text, "source": "template", "grounding": check(text, evidence), "attempts": []}


# ---------------------------------------------------------------- live lines
def moment_context(m: dict, lang: str) -> dict:
    W = WORDS.get(lang, WORDS["en"])
    e = m.get("evidence", {})
    ctx = {"clock": m.get("clock", ""), "player": m.get("player_name") or "", "team": m.get("team_name") or "",
           "score_str": m.get("score_str", ""), **{k: v for k, v in e.items() if not isinstance(v, (dict, list))}}
    if "xg" in e and e["xg"] is not None:
        ctx["xg_txt"] = f"xG {e['xg']:.2f}"
    if e.get("shot_speed_kmh"):
        ctx["speed_txt"] = f"{e['shot_speed_kmh']:.0f} km/h"
    else:
        ctx["speed_txt"] = ""
    if "outcome" in e:
        ctx["outcome_txt"] = W.get(e["outcome"], e["outcome"])
    for k in ("win_prob_before", "win_prob_after"):
        if k in e:
            ctx[k.replace("win_prob_", "wp_")] = f"{100 * e[k]:.0f}%"
    if "difficulty" in e:
        ctx["difficulty_pct"] = f"{100 * e['difficulty']:.0f}%"
        ctx["xc_pct"] = f"{100 * e.get('expected_completion', 0):.0f}%"
    if "card" in e:
        ctx["card_txt"] = W.get(e["card"], e["card"])
    if "stat" in e:
        ctx["stat"] = W.get(e["stat"], e["stat"])
    if m["kind"] == "flow_shift":
        to = e.get("to", "balanced")
        ctx["flow_from"], ctx["flow_to"] = e.get("from_label", e.get("from")), e.get("to_label", to)
        key = "chaos" if to == "chaos" else ("control" if to.startswith("control") else "balanced")
        ctx["flow_casual"] = fill(W[key], {"team": m.get("team_name") or ""})
    if "why" in m:
        ctx["why"], ctx["why_short"] = m["why"].rstrip("."), m.get("why_short", m["why"]).rstrip(".")
    return ctx


def live_line(m: dict, persona: str = "analyst", lang: str = "en") -> str | None:
    bank = LIVE.get(lang, LIVE["en"])
    tpl = bank.get(m["kind"]) or LIVE["en"].get(m["kind"])
    if not tpl:
        return None
    t = tpl.get("casual" if persona in ("casual", "kids") else "analyst")
    out = fill(t, moment_context(m, lang))
    return " ".join(out.replace(" ,", ",").replace("(, ", "(").split())
