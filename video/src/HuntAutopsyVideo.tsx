import React from "react";
import { AbsoluteFill, Sequence, interpolate, staticFile, useVideoConfig } from "remotion";
import { Audio } from "@remotion/media";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { Autopsy, Close, ColdOpen, Fans, FactCheck, Hood, Live, Problem, Proof, Recap, Reveal } from "./scenes";
import { C } from "./theme";
import vo from "./vo.json";

export const T = 15; // transition length (frames)
// minimum visual length per scene; each scene stretches to fit its voice-over line (lead-in + speech + tail)
const BASE = { cold: 7, problem: 8, reveal: 6, live: 24, fans: 12, fact: 6, recap: 6, autopsy: 13, hood: 11, proof: 9, close: 7 };
export const VO_LEAD = 0.6;
const VO_TAIL = 0.9;
type Key = keyof typeof BASE;
export const SCENES = Object.fromEntries(
  (Object.keys(BASE) as Key[]).map((k) => [k, Math.max(BASE[k], Math.ceil((VO_LEAD + vo.seconds[k] + VO_TAIL) * 10) / 10)]),
) as Record<Key, number>;
export const totalFrames = (fps: number) =>
  Object.values(SCENES).reduce((a, s) => a + Math.round(s * fps), 0) - (Object.keys(SCENES).length - 1) * T;

const Voice: React.FC<{ k: Key }> = ({ k }) => {
  const { fps } = useVideoConfig();
  return (
    <Sequence name={`Voice: ${k}`} from={Math.round(VO_LEAD * fps)} layout="none">
      <Audio src={staticFile(`vo/${k}.wav`)} volume={1} />
    </Sequence>
  );
};

export const HuntAutopsyVideo: React.FC = () => {
  const { fps, durationInFrames } = useVideoConfig();
  const t = <TransitionSeries.Transition presentation={fade()} timing={linearTiming({ durationInFrames: T })} />;
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Audio src={staticFile("music.wav")}
        volume={(f) => interpolate(f, [0, 30, durationInFrames - 90, durationInFrames], [0, 0.15, 0.15, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })} />
      <TransitionSeries name="Scenes">
        <TransitionSeries.Sequence name="Cold open" durationInFrames={Math.round(SCENES.cold * fps)}><ColdOpen /><Voice k="cold" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Problem" durationInFrames={Math.round(SCENES.problem * fps)}><Problem /><Voice k="problem" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Reveal" durationInFrames={Math.round(SCENES.reveal * fps)}><Reveal /><Voice k="reveal" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Live match" durationInFrames={Math.round(SCENES.live * fps)}><Live /><Voice k="live" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Two fans" durationInFrames={Math.round(SCENES.fans * fps)}><Fans /><Voice k="fans" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Fact-check" durationInFrames={Math.round(SCENES.fact * fps)}><FactCheck /><Voice k="fact" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Recap" durationInFrames={Math.round(SCENES.recap * fps)}><Recap /><Voice k="recap" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Autopsy" durationInFrames={Math.round(SCENES.autopsy * fps)}><Autopsy /><Voice k="autopsy" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Under the hood" durationInFrames={Math.round(SCENES.hood * fps)}><Hood /><Voice k="hood" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Proof" durationInFrames={Math.round(SCENES.proof * fps)}><Proof /><Voice k="proof" /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Close" durationInFrames={Math.round(SCENES.close * fps)}><Close /><Voice k="close" /></TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
