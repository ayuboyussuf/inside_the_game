import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

export const { fontFamily: INTER } = loadInter("normal", { weights: ["400", "600", "800"], subsets: ["latin"] });
export const { fontFamily: MONO } = loadMono("normal", { weights: ["400", "600"], subsets: ["latin"] });

// The "forensic broadcast room": night navy, evidence teal, decisive amber.
export const C = {
  bg: "#05080d",
  bg2: "#0b1420",
  panel: "rgba(10, 18, 30, 0.88)",
  line: "rgba(120, 160, 200, 0.18)",
  text: "#eef4fb",
  muted: "#8fa3bd",
  teal: "#00e0a4",
  blue: "#5aa9ff",
  amber: "#ffb547",
  red: "#ff5d73",
} as const;

// Footage scene markers (seconds into public/footage.mp4), from scripts/record_footage.py.
// The recorder's clock starts ~4.57 s before the browser video does (markers end at 120.53 s, the file is 115.96 s).
const OFFSET = 120.53 - 115.96;
export const MARK = { live: 22.41 - OFFSET, feeds: 60.08 - OFFSET, recap: 86.89 - OFFSET, autopsy: 100.5 - OFFSET, evals: 113.53 - OFFSET } as const;
