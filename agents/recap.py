"""Recap agent: half-time / full-time match recaps, per persona and language.

Builds a recap evidence JSON (score, stats, flow story, top players, key moments, best
autopsy), then narrates it: LLM when configured (fact-checked), templates otherwise.
"""
from __future__ import annotations

from collections import Counter

from narrate.grounding import check
from narrate.llm import get_llm
from narrate.narrator import live_line, llm_text, template_autopsy
from narrate.personalize import Profile, score as m_score
from narrate.templates import LIVE, WORDS

RECAP = {
    "en": {"head": "{home} {hs}-{as_} {away}.", "stats": "Possession {hp}%-{ap}%, shots {hsh}-{ash}, xG {hxg}-{axg}, pass accuracy {hpa}%-{apa}%.",
           "flow": "The game was mostly {flow}.", "potm": "Standout: {name} ({team}) — {goals} goal(s), xG {xg}, {passes} passes at {pacc}%.",
           "key": "Key moments:", "yourteam": "For {team} fans:", "casual_head": "{home} {hs}-{as_} {away} — {verdict}"},
    "es": {"head": "{home} {hs}-{as_} {away}.", "stats": "Posesión {hp}%-{ap}%, tiros {hsh}-{ash}, xG {hxg}-{axg}, acierto en el pase {hpa}%-{apa}%.",
           "flow": "El partido fue sobre todo {flow}.", "potm": "Destacado: {name} ({team}) — {goals} gol(es), xG {xg}, {passes} pases al {pacc}%.",
           "key": "Momentos clave:", "yourteam": "Para la afición de {team}:", "casual_head": "{home} {hs}-{as_} {away} — {verdict}"},
    "fr": {"head": "{home} {hs}-{as_} {away}.", "stats": "Possession {hp}%-{ap}%, tirs {hsh}-{ash}, xG {hxg}-{axg}, passes réussies {hpa}%-{apa}%.",
           "flow": "Le match a surtout été {flow}.", "potm": "Homme du match : {name} ({team}) — {goals} but(s), xG {xg}, {passes} passes à {pacc}%.",
           "key": "Moments clés :", "yourteam": "Pour les fans de {team} :", "casual_head": "{home} {hs}-{as_} {away} — {verdict}"},
    "pt": {"head": "{home} {hs}-{as_} {away}.", "stats": "Posse {hp}%-{ap}%, remates {hsh}-{ash}, xG {hxg}-{axg}, passes certos {hpa}%-{apa}%.",
           "flow": "O jogo foi sobretudo {flow}.", "potm": "Destaque: {name} ({team}) — {goals} golo(s), xG {xg}, {passes} passes a {pacc}%.",
           "key": "Momentos-chave:", "yourteam": "Para os adeptos do {team}:", "casual_head": "{home} {hs}-{as_} {away} — {verdict}"},
    "de": {"head": "{home} {hs}-{as_} {away}.", "stats": "Ballbesitz {hp}%-{ap}%, Schüsse {hsh}-{ash}, xG {hxg}-{axg}, Passquote {hpa}%-{apa}%.",
           "flow": "Das Spiel war überwiegend {flow}.", "potm": "Herausragend: {name} ({team}) — {goals} Tor(e), xG {xg}, {passes} Pässe mit {pacc}%.",
           "key": "Schlüsselmomente:", "yourteam": "Für die Fans von {team}:", "casual_head": "{home} {hs}-{as_} {away} — {verdict}"},
}
FLOW_WORDS = {"en": {"chaos": "chaotic", "balanced": "balanced", "control": "controlled by {team}"},
              "es": {"chaos": "caótico", "balanced": "equilibrado", "control": "controlado por {team}"},
              "fr": {"chaos": "chaotique", "balanced": "équilibré", "control": "contrôlé par {team}"},
              "pt": {"chaos": "caótico", "balanced": "equilibrado", "control": "controlado pelo {team}"},
              "de": {"chaos": "chaotisch", "balanced": "ausgeglichen", "control": "von {team} kontrolliert"}}
VERDICT = {"en": ("a draw", "{team} win"), "es": ("empate", "gana {team}"), "fr": ("match nul", "victoire de {team}"),
           "pt": ("empate", "vitória do {team}"), "de": ("Unentschieden", "Sieg für {team}")}


def recap_evidence(live) -> dict:
    st, meta = live.state, live.meta
    H, A = meta["home"]["id"], meta["away"]["id"]
    snap = st.snapshot()
    flows = Counter(m["evidence"]["to"] for m in live.moments if m["kind"] == "flow_shift")
    top = sorted(live.moments, key=lambda m: m.get("importance", 0), reverse=True)
    key = sorted([m for m in top if m["kind"] in ("goal", "big_chance_missed", "card", "save_autopsy", "momentum_swing")][:6], key=lambda m: m["t"])
    best_save = max((m for m in live.moments if m["kind"] == "save_autopsy"), key=lambda m: m["autopsy"]["model"]["p_outcome"], default=None)
    return {"match_id": live.match_id, "clock": snap["clock"], "home": meta["home"]["short"], "away": meta["away"]["short"],
            "score": snap["score"], "teams": snap["teams"], "dominant_flow": flows.most_common(1)[0][0] if flows else "balanced",
            "top_players": st.top_players(3), "key_moments": key, "best_save": best_save,
            "win_probability_now": snap["win_probability"], "H": H, "A": A}


def template_recap(ev: dict, prof: Profile, live) -> str:
    lang = prof.language if prof.language in RECAP else "en"
    R, FW = RECAP[lang], FLOW_WORDS[lang]
    H, A = ev["H"], ev["A"]
    th, ta = ev["teams"][H], ev["teams"][A]
    hs, as_ = ev["score"][H], ev["score"][A]
    lines = []
    if prof.voice == "casual":
        winner = None if hs == as_ else (ev["home"] if hs > as_ else ev["away"])
        v = VERDICT[lang][0] if winner is None else VERDICT[lang][1].format(team=winner)
        lines.append(R["casual_head"].format(home=ev["home"], hs=hs, as_=as_, away=ev["away"], verdict=v))
    else:
        lines.append(R["head"].format(home=ev["home"], hs=hs, as_=as_, away=ev["away"]))
        lines.append(R["stats"].format(hp=f"{th['possession_pct']:.0f}", ap=f"{ta['possession_pct']:.0f}", hsh=th["shots"], ash=ta["shots"],
                                       hxg=f"{th['xg']:.2f}", axg=f"{ta['xg']:.2f}", hpa=f"{th['pass_accuracy_pct']:.0f}", apa=f"{ta['pass_accuracy_pct']:.0f}"))
        fl = ev["dominant_flow"]
        flow = FW["chaos"] if fl == "chaos" else FW["balanced"] if fl == "balanced" else FW["control"].format(team=live.team_name(fl.split(":")[1]))
        lines.append(R["flow"].format(flow=flow))
    if ev["top_players"]:
        p = ev["top_players"][0]
        lines.append(R["potm"].format(name=p["name"], team=live.team_name(p["team"]), goals=p["goals"], xg=f"{p['xg']:.2f}",
                                      passes=p["passes"], pacc=f"{p['pass_accuracy_pct']:.0f}"))
    moments = ev["key_moments"]
    if prof.favorite_club:
        moments = sorted(moments, key=lambda m: -m_score(m, prof))[:4]
        moments.sort(key=lambda m: m["t"])
        lines.append(R["yourteam"].format(team=live.team_name(prof.favorite_club)))
    else:
        lines.append(R["key"])
    for m in moments[: (3 if prof.voice == "casual" else 6)]:
        if m.get("autopsy"):
            txt = live_line({**m, "why": template_autopsy(m["autopsy"], "analyst", lang),
                             "why_short": template_autopsy(m["autopsy"], "casual", lang)}, prof.voice, lang)
        else:
            txt = live_line(m, prof.voice, lang)
        if txt:
            lines.append("• " + txt)
    return "\n".join(lines)


def make_recap(live, prof: Profile, use_llm: bool = True) -> dict:
    ev = recap_evidence(live)
    llm = get_llm() if use_llm else None
    if llm is not None:
        slim = {k: v for k, v in ev.items() if k not in ("H", "A")}
        slim["key_moments"] = [{k: m.get(k) for k in ("clock", "kind", "team_name", "player_name", "evidence")} for m in ev["key_moments"]]
        if ev["best_save"]:
            slim["best_save"] = {"clock": ev["best_save"]["clock"], "keeper": ev["best_save"]["player_name"],
                                 "top_factors": ev["best_save"]["autopsy"]["top_factors"][:2]}
        if prof.favorite_club:
            slim["viewer_favourite_club"] = live.team_name(prof.favorite_club)
        if prof.favorite_player:
            slim["viewer_favourite_player"] = live.state.player_summary(prof.favorite_player)
        try:
            r = llm_text(llm, slim, "analyst" if prof.voice == "analyst" else "casual", prof.language, kind="match recap (4-6 sentences)")
            if r["text"]:
                return {"text": r["text"], "source": r["source"], "grounding": r["grounding"]}
        except Exception:
            pass
    text = template_recap(ev, prof, live)
    return {"text": text, "source": "template", "grounding": check(text, {"ev": ev, "words": list(WORDS)})}


__all__ = ["make_recap", "recap_evidence", "LIVE"]
