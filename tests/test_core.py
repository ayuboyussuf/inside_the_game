import numpy as np
import pandas as pd
import pytest

from events.detect import detect_events
from events.features import FEATURE_NAMES, extract_features, feature_table, mirror_episode
from events.schema import Event
from explain.evidence import Autopsy
from explain.model import OutcomeExplainer
from narrate.grounding import check
from narrate.narrator import template_autopsy
from narrate.templates import AUTOPSY
from sim.chase import RULES, generate


@pytest.fixture(scope="module")
def chases():
    return generate(160, "compound", seed=3)


def test_empty_suite_runs():
    assert True


def test_schema_rejects_unknown_event():
    with pytest.raises(ValueError):
        Event(t=0.0, type="teleport")


def test_features_complete_and_finite(chases):
    df = feature_table(chases)
    assert list(df.columns[:len(FEATURE_NAMES)]) == FEATURE_NAMES
    assert np.isfinite(df[["reaction_delay", "separation_at_commit", "closing_speed"]].values).all()
    assert set(df["outcome"]) == {0, 1}


def test_mirror_invariance(chases):
    for ep in chases[:10]:
        a, b = extract_features(ep), extract_features(mirror_episode(ep))
        for k in a:
            if np.isfinite(a[k]):
                assert abs(a[k] - b[k]) < 1e-9, k


def test_sim_planted_feature_matches_meta(chases):
    for ep in chases[:20]:
        f = extract_features(ep)
        assert abs(f["reaction_delay"] - ep.meta["params"]["reaction"]) < 1e-3


def test_no_leakage_window_ends_before_outcome(chases):
    from events.features import key_times
    for ep in chases[:10]:
        kt = key_times(ep)
        out = next(e for e in ep.events if e.type in ("contact", "abort"))
        assert kt["t_end"] <= out.t - 0.3 + 1e-6


def test_explainer_recovers_single_rule():
    tr, te = generate(300, "reaction", seed=1), generate(60, "reaction", seed=2)
    m = OutcomeExplainer("lgbm").fit(feature_table(tr))
    d = feature_table(te)
    hits = [m.top_factor(d.loc[e.episode_id], e.outcome) == "reaction_delay" for e in te if not e.meta["label_flipped"]]
    assert np.mean(hits) >= 0.8


def test_contributions_are_additive(chases):
    df = feature_table(chases)
    for kind in ("logreg", "lgbm"):
        m = OutcomeExplainer(kind).fit(df)
        phi, base = m.contributions(df.iloc[:5])
        logit = phi.sum(axis=1).values + base
        p = 1 / (1 + np.exp(-logit))
        assert np.allclose(p, m.predict_proba(df.iloc[:5]), atol=1e-6)


def test_evidence_and_templates_are_grounded(chases):
    a = Autopsy("lgbm").fit(chases)
    for ep in chases[:6]:
        ev = a.evidence(ep)
        assert ev["top_factors"] and ev["outcome"] in ("caught", "escaped")
        for lang in AUTOPSY:
            for persona in ("analyst", "casual"):
                txt = template_autopsy(ev, persona, lang)
                g = check(txt, ev)
                assert g["ok"], (lang, persona, txt, g)


def test_grounding_catches_invented_numbers():
    ev = {"value": 0.42, "p": 0.81}
    assert check("reaction 0.42 s, chance 81%", ev)["ok"]
    assert not check("reaction 0.42 s in 4 of 5 clips", ev)["ok"]


def test_detect_events_on_sim_tracks(chases):
    errs = []
    for ep in chases[:20]:
        A, D = ep.track("pred").array(), ep.track("prey").array()
        evs = detect_events(A[:, 0], A[:, 1:], D[:, 1:], "pred", "prey")
        d = next((e for e in evs if e.type == "sprint_start"), None)
        errs.append(abs(d.t - ep.first_event("sprint_start").t) if d else 9)
    assert np.mean(np.array(errs) <= 0.3) >= 0.7


def test_all_rules_generate():
    for r in RULES:
        eps = generate(5, r, seed=0)
        assert len(eps) == 5 and all(e.meta["rule"] == r for e in eps)
