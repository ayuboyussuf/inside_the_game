# Project page copy (paste into the hackathon form)

## Tagline
**The *why* behind every moment: live, fact-checked, and made for every fan.**

Alternatives (pick one):
- Every goal has a reason. Hunt Autopsy shows it, live.
- Match intelligence that works backwards from the result.

## Keywords
explainable AI · football analytics · multi-agent system · real-time sports · broadcast overlays · fan personalization · multilingual · fact-checking · synthetic data · match recaps · Azure Container Apps · Azure OpenAI · GitHub Models · LightGBM · SHAP · computer vision · YOLO · TAPNet

## Description

🎬 **Demo video (1:58):** https://youtu.be/g-nyUr_ukYE · 💻 **Code:** https://github.com/ayuboyussuf/inside_the_game

### The problem
A scoreline tells you *what* happened. Fans, analysts and broadcasters want to know *why*. Why did that goal swing the game? Why couldn't the keeper reach the shot? Was the team in control, or was it chaos? Today that "why" depends on a pundit's opinion. Stats feeds list numbers without context, and the same graphic goes to a data-hungry analyst and a casual fan watching in a different language.

### What we built
**Hunt Autopsy** is an AI insights engine that **works backwards from the outcome**. It explains the moments and movements that produced it, live, and shapes the story for each viewer. We first proved the method on predator–prey chases (every attack has an anatomy: closing speed, reaction time, separation, screening), then applied the same explainer to football.

It follows the brief's pipeline end to end:
1. **Ingest:** synthetic match events stream in as they happen: passes, shots, tackles, pressures, possession changes, cards, substitutions and match metadata.
2. **Interpret:** a live match state: pass distance, accuracy and a **difficulty rating**; xG; **shot and ball speed**; pressing (PPDA); momentum; **control vs chaos**; game tempo; win probability; and **player speed and distance tags** that fire at thresholds.
3. **Explain:** every key moment comes with its reason and evidence. A goal shows the win-probability swing it caused. A shot gets an **autopsy**: the factors that decided it (keeper reaction time, defenders screening the keeper, shot angle, distance), evidence from similar shots, the decisive tenth of a second, and a counterfactual ("if the keeper had reacted in 0.50 s instead of 0.21 s, the save chance drops from 100% to 85%").
4. **Render:** timed, machine-readable **overlay cues** (scorebug, lower thirds, player tags, autopsy cards, momentum and flow meters) with a published JSON schema, ready for a rendering partner to draw over the live feed.
5. **Personalize:** analyst, casual-fan, player-focus and young-fan modes; favourite club and player; five languages offline and any language through the LLM. In the demo, an analyst in English and a Spanish-speaking Northbridge fan watch the **same stream** as two genuinely different experiences.

It also writes **automated half-time and full-time recaps**, per viewer and per language.

### Agentic design
Eight specialised agents share one blackboard: **ingest → stats → moments → autopsy → personalizer → narrator ⇄ fact-check → overlay**, plus a recap agent.
- The narrator drafts the commentary. The **fact-check agent** blocks any sentence containing a number that is not in the evidence and sends it back to be rewritten.
- The live path stays deterministic and fast (about 1 ms per event). The LLM is used where latency allows: recaps, deep dives and new languages.
- Every agent decision is visible in a live **agent trace**.

### Why you can trust it
Explanations are only useful if they are the *right* ones, so the project tests itself with **11 built-in evals**:
- We generate chases and shots with a **planted cause**, and the explainer must find it: **91.9%** overall.
- Removing the top factor changes the prediction **21.6×** more than removing a random one.
- **0 unsupported numbers** in 1,200 generated narrations.
- We also report what doesn't work yet: on real, openly licensed football footage, our video auto-eventing finds the ball **60%** of the time. The fine-tuning plan to fix that is in the repo.

### Data innovation
A synthetic, football-realistic match generator with fictional clubs and players. Every on-target shot carries a 360-style freeze frame, a tracking snippet and a **planted keeper model**, so the dataset doubles as ground truth for explainability. No real Premier League data or footage is used.

### Technology
Python · FastAPI (server-sent events) · LightGBM + TreeSHAP · YOLO + BoT-SORT · Google DeepMind TAPNet · OpenCV.
LLM narrator via **Azure OpenAI / Microsoft Foundry** (managed identity) or **GitHub Models**. Deploys to **Azure Container Apps** (Bicep: Container Registry, Log Analytics, managed identity). GitHub Actions CI runs the tests and evals.

### Impact and what's next
Broadcasters get explainable, on-screen insight they can trust; streaming platforms get one engine that personalizes for every market; fans get the *why*. Next steps:
- fine-tune the football detector, ball and pitch-keypoint models (a free-GPU notebook is included) so the same explainer runs straight from video;
- add Azure AI Speech for spoken commentary;
- move the orchestration onto Microsoft Agent Framework.

**Repo:** https://github.com/ayuboyussuf/inside_the_game
