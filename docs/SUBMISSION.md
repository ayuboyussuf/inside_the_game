# Submission: Hunt Autopsy

**Repository:** https://github.com/ayuboyussuf/inside_the_game (AGPL-3.0)
**Demo video:** https://youtu.be/g-nyUr_ukYE
**Tagline:** The *why* behind every moment: live, fact-checked, and made for every fan.
**Try it:** see *Testing instructions* below.

## Elevator pitch (≈ 150 words)

**Hunt Autopsy explains *why* a football moment happened, not just *that* it happened.** It ingests synthetic match events live, turns them into match intelligence, and explains every key moment with evidence:
- **Intelligence:** pass difficulty, xG, shot speed, pressing, momentum, control vs chaos and win probability.
- **Explanation:** why a goal swung the game, and why a keeper could or couldn't reach a shot, down to the decisive tenth of a second and a counterfactual.

A team of agents delivers it as **timed, machine-readable broadcast overlays and narration**:
- ingest, stats, moments, autopsy, personalizer, narrator ⇄ fact-check, overlay, recap;
- personalized for **an analyst and a casual fan watching the same stream**, with favourite club, player focus and 5+ languages.

A fact-check agent blocks any sentence containing a number that isn't in the evidence. Eleven built-in evals prove the explanations are the right ones, using synthetic chases and shots with planted causes. Built with Python, FastAPI, LightGBM, YOLO + TAPNet, and an LLM narrator via Azure OpenAI or GitHub Models. It deploys to Azure Container Apps.

## What we built

| Brief stage | How |
|---|---|
| (a) Ingest live events | Synthetic match generator (passes, shots, tackles, pressures, possession changes, cards, subs, metadata) streamed over SSE; configurable adapter for other feeds |
| (b) Interpret | Live state: possession, pass accuracy, **pass difficulty**, xG, **shot and ball speed**, PPDA, pressing rate, momentum, **control vs chaos**, tempo, Poisson win probability, **player speed/distance tags** at thresholds |
| (c) Explain why it matters | Win-probability swing per goal; flow and pressure shifts with evidence; **shot autopsy**: LightGBM + TreeSHAP factors, similar-shot evidence, decisive moment, counterfactual |
| (d) Render on screen | Timed overlay cues (JSON schema at `/api/overlay-schema`): scorebug, lower thirds, player tags, autopsy cards, momentum and flow meters |
| (e) Personalize | Analyst / casual / player-focus / young-fan modes, favourite club framing, favourite player, language (en, es, fr, pt, de offline; any language via the LLM); two viewers side by side from one stream |
| Narratives and recaps | Live lines, half-time and full-time recaps per persona and language |
| Data innovation | Synthetic match feed with 360-style freeze frames, tracking snippets for every on-target shot, and a **planted keeper model** so explanations can be scored |
| Auto-eventing (video) | YOLO + BoT-SORT re-ID, tiled ball detection + Kalman, pitch homography; TAPNet keeper reaction time; eval 11 measures it on openly licensed match clips |

## Agentic design
Eight single-purpose agents share a blackboard: `ingest → stats → moments → autopsy → personalizer → narrator ⇄ fact-check → overlay`, plus `recap`.
- **Generator/critic loop:** the narrator agent drafts; the fact-check agent rejects any number not found in the evidence JSON and names it back; after two failures, grounded templates are used.
- **Visible decisions:** the agent trace is shown live in the UI, with per-agent timing.
- **Latency-aware:** the live path is deterministic (about 1 ms per event); the LLM handles recaps, deep dives and extra languages.

## Microsoft technology
- **Azure Container Apps** (Bicep: Container Apps env, ACR, Log Analytics, managed identity), `infra/`.
- **Azure OpenAI / Microsoft Foundry** with Entra ID managed identity, or **GitHub Models** (free), `narrate/llm.py`.
- **LightGBM** (Microsoft) for the explainer, with TreeSHAP contributions.
- GitHub Actions CI runs tests and evals.

## Trust: the evals (`python -m evals.run_all`)
| Status | Eval | Result |
|---|---|---|
| PASS | 1. Planted-cause recovery | 91.9% (chases ≈ 100%, shots 69%) |
| PASS | 2. Deletion faithfulness | top factor 21.6× a random one |
| PASS | 3. Outcome prediction | AUC 0.91 (chases), 0.84 (shots) |
| PASS | 4. Stability | 80% of clips keep the same top-2; top-1 92% |
| PASS | 5. Flip invariance | exact |
| PASS | 6. Camera compensation | 0.015 BL/s |
| PASS | 7. Event timing | 95% / 93% within 0.3 s |
| PASS | 9. Grounded narration | 0 unsupported numbers in 1,200 texts |
| FAIL (reported honestly) | 11. Real football video | ball 60%, players 73%; fine-tuning plan in `docs/FOOTBALL_VIDEO_EVAL.md` |

## Testing instructions
```bash
git clone https://github.com/ayuboyussuf/inside_the_game && cd inside_the_game
pip install uv && uv venv && uv pip install -e ".[dev,video]"
uv run uvicorn app.main:app        # open http://localhost:8000
```
- **Live match:** choose two viewers (e.g. Analyst/English and Casual/Español with a favourite club), then press **Kick off**. Ask for a recap at any time.
- **Autopsy lab:** pick a football shot to see why it was saved or scored. Upload your own clip under *Analyse your own clip*.
- **Evals:** the tab shows the latest eval table.
- **Optional LLM narrator:** set `GITHUB_MODELS_TOKEN` (free) or the Azure OpenAI variables in `.env`.

All data is synthetic or openly licensed. Clubs and players are fictional, and no Premier League footage or data is used.
