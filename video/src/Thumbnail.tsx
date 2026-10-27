import React from "react";
import { AbsoluteFill } from "remotion";
import { Backdrop } from "./components";
import { C, INTER, MONO } from "./theme";

// YouTube thumbnail, 1280x720: one idea, readable at phone size. "WHY?" carries it; the card is the proof.
export const Thumbnail: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Backdrop dots={60} />
    {/* chase trail: the "anatomy" motif */}
    {new Array(22).fill(0).map((_, i) => (
      <div key={i} style={{ position: "absolute", left: 640 + i * 26, top: 610 - Math.sin(i / 3.2) * 70 - i * 9, width: 16 - i * 0.5,
        height: 16 - i * 0.5, borderRadius: 20, background: i < 11 ? C.amber : C.teal, opacity: 0.25 + i / 30,
        boxShadow: `0 0 16px ${i < 11 ? C.amber : C.teal}` }} />
    ))}
    {/* headline */}
    <div style={{ position: "absolute", left: 70, top: 70, fontFamily: INTER, fontWeight: 800, color: C.text, lineHeight: 0.92, letterSpacing: -4 }}>
      <div style={{ fontSize: 205, color: C.amber, textShadow: `0 0 60px ${C.amber}66` }}>WHY?</div>
      <div style={{ fontSize: 76, marginTop: 18 }}>AI explains every</div>
      <div style={{ fontSize: 76 }}>football moment</div>
    </div>
    {/* proof card */}
    <div style={{ position: "absolute", right: 60, top: 80, width: 470, background: "rgba(8,14,24,.94)", border: `2px solid ${C.teal}88`,
      borderRadius: 26, padding: "26px 30px", boxShadow: `0 30px 80px rgba(0,0,0,.6), 0 0 40px ${C.teal}33`, rotate: "2deg" }}>
      <div style={{ fontFamily: INTER, fontSize: 30, fontWeight: 800, color: C.text }}>Why was it saved?</div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 18 }}>
        <div style={{ fontFamily: INTER, fontSize: 26, color: C.muted }}>Keeper reaction</div>
        <div style={{ fontFamily: MONO, fontSize: 54, fontWeight: 600, color: C.teal }}>0.21 s</div>
      </div>
      <div style={{ height: 12, borderRadius: 8, background: "#1a2636", marginTop: 12, overflow: "hidden" }}>
        <div style={{ width: "88%", height: "100%", background: `linear-gradient(90deg, ${C.teal}, ${C.blue})` }} />
      </div>
      <div style={{ marginTop: 22, display: "inline-block", border: `4px solid ${C.teal}`, color: C.teal, borderRadius: 12, padding: "6px 16px",
        fontFamily: MONO, fontSize: 30, fontWeight: 600, letterSpacing: 3, rotate: "-4deg" }}>✓ GROUNDED</div>
    </div>
    {/* brand bar */}
    <div style={{ position: "absolute", left: 70, bottom: 54, display: "flex", alignItems: "center", gap: 18 }}>
      <div style={{ width: 62, height: 62, borderRadius: 16, background: `conic-gradient(from 200deg, ${C.teal}, ${C.blue}, ${C.teal})`,
        display: "flex", alignItems: "center", justifyContent: "center", fontFamily: INTER, fontWeight: 800, fontSize: 28, color: C.bg }}>HA</div>
      <div style={{ fontFamily: INTER, fontWeight: 800, fontSize: 46, color: C.text }}>Hunt Autopsy</div>
      <div style={{ fontFamily: INTER, fontSize: 30, color: C.muted, marginLeft: 8 }}>· live · fact-checked · for every fan</div>
    </div>
  </AbsoluteFill>
);
