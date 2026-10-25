import React from "react";
import { AbsoluteFill, interpolate, staticFile, useVideoConfig } from "remotion";
import { Audio } from "@remotion/media";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { Autopsy, Close, ColdOpen, Fans, FactCheck, Hood, Live, Problem, Proof, Recap, Reveal } from "./scenes";
import { C } from "./theme";

export const T = 15; // transition length (frames)
// scene lengths in seconds; total = sum - transitions
export const SCENES = { cold: 7, problem: 8, reveal: 6, live: 24, fans: 12, fact: 6, recap: 6, autopsy: 13, hood: 11, proof: 9, close: 7 };
export const totalFrames = (fps: number) =>
  Object.values(SCENES).reduce((a, s) => a + s * fps, 0) - (Object.keys(SCENES).length - 1) * T;

export const HuntAutopsyVideo: React.FC = () => {
  const { fps, durationInFrames } = useVideoConfig();
  const t = <TransitionSeries.Transition presentation={fade()} timing={linearTiming({ durationInFrames: T })} />;
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Audio src={staticFile("music.wav")}
        volume={(f) => interpolate(f, [0, 30, durationInFrames - 90, durationInFrames], [0, 0.85, 0.85, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })} />
      <TransitionSeries name="Scenes">
        <TransitionSeries.Sequence name="Cold open" durationInFrames={SCENES.cold * fps}><ColdOpen /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Problem" durationInFrames={SCENES.problem * fps}><Problem /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Reveal" durationInFrames={SCENES.reveal * fps}><Reveal /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Live match" durationInFrames={SCENES.live * fps}><Live /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Two fans" durationInFrames={SCENES.fans * fps}><Fans /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Fact-check" durationInFrames={SCENES.fact * fps}><FactCheck /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Recap" durationInFrames={SCENES.recap * fps}><Recap /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Autopsy" durationInFrames={SCENES.autopsy * fps}><Autopsy /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Under the hood" durationInFrames={SCENES.hood * fps}><Hood /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Proof" durationInFrames={SCENES.proof * fps}><Proof /></TransitionSeries.Sequence>
        {t}
        <TransitionSeries.Sequence name="Close" durationInFrames={SCENES.close * fps}><Close /></TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
