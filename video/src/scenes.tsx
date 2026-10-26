import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { Backdrop, Callout, Chapter, Counter, FadeIn, Footage, Mark, Stamp, Words } from "./components";
import { C, INTER, MARK, MONO } from "./theme";

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

/* 1 ── cold open: a chase drawn in tracking dots → "every attack has an anatomy" */
export const ColdOpen: React.FC = () => {
  const frame = useCurrentFrame();
  const path = (t: number, lead: number) => {
    const u = t + lead;
    return { x: 260 + u * 1350, y: 610 - Math.sin(u * 5.2) * 120 - u * 90 };
  };
  const p = Math.min(1, frame / 120);
  const trail = (lead: number, color: string) =>
    new Array(26).fill(0).map((_, i) => {
      const t = Math.max(0, p - i * 0.012);
      const q = path(t, lead);
      return <div key={i} style={{ position: "absolute", left: q.x, top: q.y, width: 14 - i * 0.4, height: 14 - i * 0.4,
        borderRadius: 20, background: color, opacity: 1 - i / 28, boxShadow: `0 0 18px ${color}` }} />;
    });
  return (
    <AbsoluteFill>
      <Backdrop dots={40} />
      <div style={{ opacity: interpolate(frame, [0, 10, 150, 175], [0, 1, 1, 0], clamp) }}>
        {trail(0.0, C.amber)}
        {trail(0.11, C.teal)}
        <div style={{ position: "absolute", left: path(p, 0).x - 40, top: path(p, 0).y - 50, fontFamily: MONO, fontSize: 24, color: C.amber }}>attacker</div>
        <div style={{ position: "absolute", left: path(p, 0.11).x + 20, top: path(p, 0.11).y + 10, fontFamily: MONO, fontSize: 24, color: C.teal }}>defender</div>
      </div>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", marginTop: 380 }}>
          <Words text="Every attack has an *anatomy." delay={40} size={92} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* 2 ── the problem */
export const Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const strike = interpolate(frame, [120, 140], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 60 }}>
        <FadeIn>
          <div style={{ display: "flex", borderRadius: 16, overflow: "hidden", fontFamily: INTER, fontWeight: 800, fontSize: 64, boxShadow: "0 30px 80px rgba(0,0,0,.6)" }}>
            <div style={{ background: "#0d1726", color: C.text, padding: "18px 34px", borderLeft: `10px solid ${C.red}` }}>NORTHBRIDGE</div>
            <div style={{ background: "#f4f7fb", color: "#0b1320", padding: "18px 34px", fontFamily: MONO }}>1 – 0</div>
            <div style={{ background: "#0d1726", color: C.text, padding: "18px 34px", borderRight: `10px solid ${C.blue}` }}>RIVERSIDE</div>
          </div>
        </FadeIn>
        <div style={{ textAlign: "center", position: "relative" }}>
          <Words text="The scoreline tells you what happened." delay={25} size={72} />
          <div style={{ position: "absolute", left: 0, top: 46, height: 6, width: `${strike * 100}%`, background: C.red, opacity: 0.85 }} />
        </div>
        <Words text="Fans want to know ^why." delay={140} size={96} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* 3 ── reveal */
export const Reveal: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 34 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 34, scale: `${interpolate(frame, [0, 20], [0.85, 1], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) })}`,
          opacity: interpolate(frame, [0, 12], [0, 1], clamp) }}>
          <div style={{ width: 130, height: 130, borderRadius: 30, background: `conic-gradient(from 200deg, ${C.teal}, ${C.blue}, ${C.teal})`,
            display: "flex", alignItems: "center", justifyContent: "center", fontFamily: INTER, fontWeight: 800, fontSize: 56, color: C.bg }}>HA</div>
          <div style={{ fontFamily: INTER, fontWeight: 800, fontSize: 140, color: C.text, letterSpacing: -4 }}>Hunt Autopsy</div>
        </div>
        <FadeIn delay={30}>
          <div style={{ fontFamily: INTER, fontSize: 50, color: C.muted, textAlign: "center", maxWidth: 1400, lineHeight: 1.3 }}>
            Works <span style={{ color: C.teal }}>backwards from the outcome</span> to explain the moments that caused it.<br />
            Live. Fact-checked. Made for every fan.
          </div>
        </FadeIn>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* 4 ── live match: ingest + interpret + render */
export const Live: React.FC = () => (
  <AbsoluteFill>
    <Backdrop dots={20} />
    <Footage startAt={MARK.live + 4} keys={[
      { f: 0, x: 960, y: 540, s: 1 }, { f: 80, x: 960, y: 540, s: 1 },
      { f: 140, x: 700, y: 610, s: 1.45 }, { f: 300, x: 520, y: 820, s: 1.9 },
      { f: 400, x: 1515, y: 640, s: 1.6 }, { f: 560, x: 1515, y: 420, s: 2.0 }, { f: 680, x: 960, y: 540, s: 1 }]} />
    <Chapter n="01" title="INGEST" sub="synthetic match events stream in, live" from={10} to={130} />
    <Callout x={1170} y={150} from={20} to={130} text="Passes, shots, tackles, pressures and possession changes, as they happen" />
    <Chapter n="04" title="RENDER" sub="timed, machine-readable overlay cues" from={150} to={330} />
    <Callout x={1150} y={760} from={170} to={330} text="Broadcast graphics a rendering partner can draw over the live feed" />
    <Chapter n="02" title="INTERPRET" sub="football states and statistics" from={345} to={680} />
    <Callout x={130} y={760} from={360} to={520} text="Pass <b>difficulty</b> · xG · shot speed · pressing (PPDA) · momentum" />
    <Callout x={130} y={760} from={540} to={680} accent={C.amber} text="<b>Control vs chaos</b> and live win probability" />
  </AbsoluteFill>
);

/* 5 ── personalize: two viewers, one stream */
export const Fans: React.FC = () => (
  <AbsoluteFill>
    <Backdrop dots={20} />
    <Footage startAt={MARK.feeds} keys={[
      { f: 0, x: 960, y: 560, s: 1 }, { f: 50, x: 700, y: 690, s: 1.55 }, { f: 210, x: 700, y: 690, s: 1.55 },
      { f: 260, x: 1484, y: 690, s: 1.75 }, { f: 360, x: 1484, y: 690, s: 1.75 }]} />
    <Chapter n="05" title="PERSONALIZE" sub="one stream, two genuinely different experiences" from={10} to={230} />
    <Callout x={120} y={850} w={760} from={60} to={230} accent={C.teal} text="<b>Analyst · English</b>: every number, every pattern" />
    <Callout x={1000} y={850} w={800} from={90} to={230} accent={C.red} text="<b>Casual fan · Español · ♥ Northbridge</b>: short, emotional, their club first" />
    <Callout x={120} y={850} w={1100} from={260} to={360} accent={C.blue} text="Eight agents share one blackboard. Every decision is traced live." />
  </AbsoluteFill>
);

/* 6 ── fact-check */
export const FactCheck: React.FC = () => (
  <AbsoluteFill>
    <Backdrop dots={20} />
    <Footage startAt={MARK.feeds + 6} keys={[{ f: 0, x: 700, y: 690, s: 1.55 }, { f: 60, x: 430, y: 620, s: 2.4 }, { f: 180, x: 430, y: 620, s: 2.4 }]} />
    <Stamp from={40} text="✓ GROUNDED" />
    <Callout x={130} y={820} w={1260} from={60} to={200} text="The <b>fact-check agent</b> blocks any sentence with a number that isn't in the evidence. No invented stats." />
  </AbsoluteFill>
);

/* 7 ── recap */
export const Recap: React.FC = () => (
  <AbsoluteFill>
    <Backdrop dots={20} />
    <Footage startAt={MARK.recap + 0.3} keys={[{ f: 0, x: 960, y: 800, s: 1.2 }, { f: 50, x: 640, y: 980, s: 1.8 }, { f: 180, x: 640, y: 980, s: 1.8 }]} />
    <Callout x={130} y={150} w={1100} from={20} to={180} text="Automated <b>match recaps</b>, written for each viewer's language and favourite club" />
  </AbsoluteFill>
);

/* 8 ── explain: the shot autopsy (footage slowed to 0.78x so 13 s of capture covers the narration) */
export const Autopsy: React.FC = () => (
  <AbsoluteFill>
    <Backdrop dots={20} />
    <Footage startAt={MARK.autopsy} rate={0.78} keys={[
      { f: 0, x: 960, y: 540, s: 1 }, { f: 50, x: 895, y: 450, s: 1.3 }, { f: 190, x: 895, y: 450, s: 1.3 },
      { f: 250, x: 1540, y: 330, s: 1.85 }, { f: 350, x: 1540, y: 330, s: 1.85 }, { f: 400, x: 1540, y: 560, s: 2.0 }, { f: 480, x: 1540, y: 560, s: 2.0 }]}>
      <Mark x={994} y={662} w={112} h={116} from={88} to={238} />
      <Mark x={1346} y={404} w={380} h={44} from={260} to={480} color={C.teal} />
      <Mark x={1346} y={580} w={380} h={40} from={400} to={480} color={C.teal} />
    </Footage>
    <Chapter n="03" title="EXPLAIN" sub="why was this shot saved?" from={10} to={250} />
    <Callout x={130} y={830} w={1050} from={88} to={238} accent={C.amber} text="The <b>decisive moment</b>: the save chance jumps right as the ball is struck" />
    <Callout x={130} y={830} w={1050} from={255} to={380} text="Top factor: <b>keeper reaction 0.21 s</b>. Saved in 656 of 721 similar shots, plus a counterfactual" />
    <Callout x={130} y={830} w={1050} from={390} to={480} accent={C.teal} text="Shots carry a planted cause, so we can check the explanation: it <b>matches</b>" />
  </AbsoluteFill>
);

/* 9 ── under the hood: the agent graph */
const AGENTS = ["ingest", "stats", "moments", "autopsy", "personalizer", "narrator", "fact-check", "overlay"];
export const Hood: React.FC = () => {
  const frame = useCurrentFrame();
  const active = Math.floor(interpolate(frame, [30, 200], [0, AGENTS.length], clamp));
  const tech = ["Azure Container Apps", "Azure OpenAI · Foundry", "GitHub Models", "FastAPI · SSE", "LightGBM + TreeSHAP", "YOLO · TAPNet"];
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 120 }}>
        <Words text="Eight agents. One *blackboard." size={84} />
        <div style={{ display: "flex", gap: 10, marginTop: 90, alignItems: "center", maxWidth: 1740 }}>
          {AGENTS.map((a, i) => {
            const on = i < active;
            const loop = a === "narrator" || a === "fact-check";
            return (
              <React.Fragment key={a}>
                <div style={{ fontFamily: MONO, fontSize: 24, fontWeight: 600, padding: "18px 14px", borderRadius: 14,
                  color: on ? C.bg : C.muted, background: on ? (loop ? C.amber : C.teal) : C.panel,
                  border: `2px solid ${on ? (loop ? C.amber : C.teal) : C.line}`, boxShadow: on ? `0 0 30px ${loop ? C.amber : C.teal}66` : "none",
                  scale: `${on ? 1 : 0.96}` }}>{a}</div>
                {i < AGENTS.length - 1 && <div style={{ fontFamily: MONO, fontSize: 28, color: i < active - 1 ? C.teal : C.line }}>{a === "narrator" ? "⇄" : "→"}</div>}
              </React.Fragment>
            );
          })}
        </div>
        <FadeIn delay={120} style={{ marginTop: 60, fontFamily: INTER, fontSize: 38, color: C.muted, textAlign: "center", maxWidth: 1500 }}>
          The live path is deterministic (about 1 ms per event). The LLM writes recaps, deep dives and new languages, and every draft is fact-checked.
        </FadeIn>
        <div style={{ display: "flex", gap: 16, marginTop: 70, flexWrap: "wrap", justifyContent: "center", maxWidth: 1600 }}>
          {tech.map((t, i) => (
            <FadeIn key={t} delay={180 + i * 8}>
              <div style={{ fontFamily: INTER, fontWeight: 600, fontSize: 30, color: C.text, padding: "12px 22px", borderRadius: 999, border: `1px solid ${C.teal}66`, background: "rgba(0,224,164,.07)" }}>{t}</div>
            </FadeIn>
          ))}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* 10 ── proof */
export const Proof: React.FC = () => {
  const stat = (label: string, el: React.ReactNode, delay: number) => (
    <FadeIn delay={delay} style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 24, padding: "34px 40px", width: 760 }}>
      {el}
      <div style={{ fontFamily: INTER, fontSize: 34, color: C.muted, marginTop: 6 }}>{label}</div>
    </FadeIn>
  );
  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 90 }}>
        <Words text="It proves its explanations are the *right ones." size={74} />
        <div style={{ display: "grid", gridTemplateColumns: "760px 760px", gap: 36, marginTop: 70 }}>
          {stat("of planted causes recovered by the explainer", <Counter to={91.9} from={20} decimals={1} suffix="%" />, 15)}
          {stat("bigger impact than a random factor when the top factor is removed", <Counter to={21.6} from={30} decimals={1} suffix="×" />, 25)}
          {stat("unsupported numbers in 1,200 narrations", <Counter to={0} from={40} />, 35)}
          {stat("built-in evals, including the one we still fail", <Counter to={11} from={50} />, 45)}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* 11 ── close */
export const Close: React.FC = () => (
  <AbsoluteFill>
    <Backdrop />
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 40 }}>
      <Words text="The ^why behind every moment." size={110} />
      <FadeIn delay={30}>
        <div style={{ fontFamily: INTER, fontSize: 48, color: C.muted, textAlign: "center" }}>Live · fact-checked · made for every fan</div>
      </FadeIn>
      <FadeIn delay={55}>
        <div style={{ fontFamily: MONO, fontSize: 40, color: C.teal, marginTop: 30 }}>github.com/ayuboyussuf/inside_the_game</div>
      </FadeIn>
      <FadeIn delay={75}>
        <div style={{ fontFamily: INTER, fontSize: 28, color: C.muted, marginTop: 20 }}>Synthetic data only · fictional clubs and players · original soundtrack</div>
      </FadeIn>
    </AbsoluteFill>
  </AbsoluteFill>
);
