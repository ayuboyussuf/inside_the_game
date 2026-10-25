import React from "react";
import { AbsoluteFill, Easing, interpolate, random, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Video } from "@remotion/media";
import { C, INTER, MONO } from "./theme";

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const ease = Easing.bezier(0.16, 1, 0.3, 1);

/* ------------------------------------------------------------ background world */
export const Backdrop: React.FC<{ dots?: number }> = ({ dots = 70 }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: `radial-gradient(circle at 50% 35%, ${C.bg2} 0%, ${C.bg} 70%)` }}>
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(${C.line} 1px, transparent 1px), linear-gradient(90deg, ${C.line} 1px, transparent 1px)`,
          backgroundSize: "80px 80px",
          opacity: 0.35,
          translate: `0px ${(frame * 0.4) % 80}px`,
        }}
      />
      {new Array(dots).fill(0).map((_, i) => {
        const x = random(`x${i}`) * 1920;
        const y = (random(`y${i}`) * 1080 + frame * (0.3 + random(`s${i}`) * 0.8)) % 1080;
        const o = 0.15 + random(`o${i}`) * 0.45;
        return (
          <div key={i} style={{ position: "absolute", left: x, top: y, width: 4, height: 4, borderRadius: 4,
            background: i % 7 === 0 ? C.amber : C.teal, opacity: o, boxShadow: `0 0 10px ${C.teal}` }} />
        );
      })}
      {/* scan line */}
      <div style={{ position: "absolute", left: 0, right: 0, top: (frame * 6) % 1200 - 60, height: 2,
        background: `linear-gradient(90deg, transparent, ${C.teal}55, transparent)` }} />
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------ text */
export const Words: React.FC<{ text: string; delay?: number; size?: number; color?: string; weight?: number; stagger?: number;
  style?: React.CSSProperties }> = ({ text, delay = 0, size = 96, color = C.text, weight = 800, stagger = 4, style }) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ fontFamily: INTER, fontSize: size, fontWeight: weight, color, lineHeight: 1.08, letterSpacing: -1.5, ...style }}>
      {text.split(" ").map((w, i) => {
        const f = frame - delay - i * stagger;
        return (
          <span key={i} style={{ display: "inline-block", marginRight: size * 0.26,
            opacity: interpolate(f, [0, 12], [0, 1], clamp),
            translate: `0px ${interpolate(f, [0, 16], [size * 0.45, 0], { ...clamp, easing: ease })}px` }}>
            {w.startsWith("*") ? <span style={{ color: C.teal }}>{w.slice(1)}</span> :
              w.startsWith("^") ? <span style={{ color: C.amber }}>{w.slice(1)}</span> : w}
          </span>
        );
      })}
    </div>
  );
};

export const FadeIn: React.FC<{ delay?: number; dur?: number; out?: number; children: React.ReactNode; style?: React.CSSProperties }> =
  ({ delay = 0, dur = 14, out, children, style }) => {
    const frame = useCurrentFrame();
    const a = interpolate(frame - delay, [0, dur], [0, 1], { ...clamp, easing: ease });
    const b = out === undefined ? 1 : interpolate(frame, [out, out + 10], [1, 0], clamp);
    return <div style={{ opacity: a * b, translate: `0px ${(1 - a) * 24}px`, ...style }}>{children}</div>;
  };

/* ------------------------------------------------------------ chapter tag (the brief's 5 stages) */
export const Chapter: React.FC<{ n: string; title: string; sub: string; from?: number; to?: number }> = ({ n, title, sub, from = 0, to = 9999 }) => {
  const frame = useCurrentFrame();
  const a = interpolate(frame, [from, from + 14], [0, 1], { ...clamp, easing: ease });
  const b = interpolate(frame, [to, to + 12], [1, 0], clamp);
  return (
    <div style={{ position: "absolute", left: 90, top: 70, opacity: a * b, translate: `${(1 - a) * -40}px 0px`, display: "flex", gap: 22, alignItems: "center",
      background: C.panel, border: `1px solid ${C.line}`, borderRadius: 18, padding: "16px 26px 16px 18px", boxShadow: "0 20px 60px rgba(0,0,0,.5)" }}>
      <div style={{ fontFamily: MONO, fontSize: 30, fontWeight: 600, color: C.bg, background: C.teal, borderRadius: 10, padding: "6px 12px" }}>{n}</div>
      <div>
        <div style={{ fontFamily: INTER, fontSize: 40, fontWeight: 800, color: C.text, letterSpacing: 2 }}>{title}</div>
        <div style={{ fontFamily: INTER, fontSize: 26, fontWeight: 400, color: C.muted }}>{sub}</div>
      </div>
    </div>
  );
};

/* ------------------------------------------------------------ callout */
export const Callout: React.FC<{ x: number; y: number; from: number; to: number; text: string; accent?: string; w?: number }> =
  ({ x, y, from, to, text, accent = C.teal, w = 620 }) => {
    const frame = useCurrentFrame();
    const a = interpolate(frame, [from, from + 12], [0, 1], { ...clamp, easing: ease });
    const b = interpolate(frame, [to, to + 10], [1, 0], clamp);
    return (
      <div style={{ position: "absolute", left: x, top: y, width: w, opacity: a * b, scale: `${0.94 + 0.06 * a}`,
        background: "rgba(4, 10, 18, 0.93)", borderLeft: `6px solid ${accent}`, borderRadius: 14, padding: "18px 24px",
        fontFamily: INTER, fontSize: 34, fontWeight: 600, color: C.text, lineHeight: 1.25, boxShadow: "0 18px 50px rgba(0,0,0,.55)" }}
        dangerouslySetInnerHTML={{ __html: text }} />
    );
  };

/* ------------------------------------------------------------ camera over real footage */
export type Key = { f: number; x: number; y: number; s: number };
export const Footage: React.FC<{ startAt: number; keys: Key[]; children?: React.ReactNode }> = ({ startAt, keys, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const fs = keys.map((k) => k.f);
  const opt = { ...clamp, easing: Easing.bezier(0.45, 0, 0.2, 1) };
  const s = keys.length > 1 ? interpolate(frame, fs, keys.map((k) => k.s), opt) : keys[0].s;
  const cx = keys.length > 1 ? interpolate(frame, fs, keys.map((k) => k.x), opt) : keys[0].x;
  const cy = keys.length > 1 ? interpolate(frame, fs, keys.map((k) => k.y), opt) : keys[0].y;
  const frameIn = interpolate(frame, [0, 18], [0.92, 1], { ...clamp, easing: ease });
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
      <div style={{ width: 1920 * 0.86, height: 1080 * 0.86, borderRadius: 22, overflow: "hidden", position: "relative",
        border: `1px solid ${C.line}`, boxShadow: `0 40px 120px rgba(0,0,0,.7), 0 0 0 1px ${C.teal}22`, scale: `${frameIn}`, background: C.bg }}>
        <div style={{ position: "absolute", left: 0, top: 0, width: 1920, height: 1080, transformOrigin: "0 0",
          scale: `${0.86 * s}`, translate: `${0.86 * (960 - cx * s)}px ${0.86 * (540 - cy * s)}px` }}>
          <Video src={staticFile("footage.mp4")} trimBefore={Math.round(startAt * fps)} muted style={{ width: 1920, height: 1080 }} />
          {children}
        </div>
      </div>
    </AbsoluteFill>
  );
};

/* highlight box drawn in footage coordinates (moves with the camera) */
export const Mark: React.FC<{ x: number; y: number; w: number; h: number; from: number; to: number; color?: string }> =
  ({ x, y, w, h, from, to, color = C.amber }) => {
    const frame = useCurrentFrame();
    const a = interpolate(frame, [from, from + 10], [0, 1], clamp) * interpolate(frame, [to, to + 8], [1, 0], clamp);
    const pulse = 0.6 + 0.4 * Math.sin((frame - from) / 5);
    return <div style={{ position: "absolute", left: x, top: y, width: w, height: h, borderRadius: 10, opacity: a,
      border: `4px solid ${color}`, boxShadow: `0 0 ${18 * pulse}px ${color}` }} />;
  };

/* ------------------------------------------------------------ stamp */
export const Stamp: React.FC<{ from: number; text: string }> = ({ from, text }) => {
  const frame = useCurrentFrame();
  const a = interpolate(frame, [from, from + 8], [0, 1], clamp);
  const sc = interpolate(frame, [from, from + 10], [2.2, 1], { ...clamp, easing: Easing.bezier(0.2, 1.4, 0.4, 1) });
  return (
    <div style={{ position: "absolute", right: 150, top: 160, opacity: a, scale: `${sc}`, rotate: "-8deg",
      border: `6px solid ${C.teal}`, color: C.teal, borderRadius: 18, padding: "16px 30px", fontFamily: MONO, fontSize: 52,
      fontWeight: 600, letterSpacing: 4, background: "rgba(0,224,164,.08)", boxShadow: `0 0 40px ${C.teal}55` }}>{text}</div>
  );
};

/* ------------------------------------------------------------ counter */
export const Counter: React.FC<{ to: number; from: number; dur?: number; decimals?: number; suffix?: string; size?: number; color?: string }> =
  ({ to, from, dur = 40, decimals = 0, suffix = "", size = 120, color = C.teal }) => {
    const frame = useCurrentFrame();
    const v = interpolate(frame, [from, from + dur], [0, to], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) });
    return <span style={{ fontFamily: MONO, fontSize: size, fontWeight: 600, color }}>{v.toFixed(decimals)}{suffix}</span>;
  };
