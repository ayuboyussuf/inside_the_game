/* Hunt Autopsy front-end: vanilla JS, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = (p, opt) => fetch(p, opt).then(r => { if (!r.ok) throw new Error(r.status + " " + p); return r.json(); });
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let CFG = null, MATCH = null, ES = null, PAUSED = false, CUR_T = 0;
const ROLE_COL = { attacker: "#ff8a3d", defender: "#e6d3a3", attacker_support: "#ffb47a", defender_support: "#8fb3d9", other: "#999" };

/* ------------------------------------------------------------------ tabs */
$$(".tab").forEach(b => b.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $$(".tabpane").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "autopsy" && !$("#scenarios").children.length) loadLibrary();
  if (b.dataset.tab === "evals") loadEvals();
});

async function health() {
  try {
    const h = await api("/api/health");
    $("#status").classList.toggle("ok", h.models_ready);
    $("#statusText").textContent = (h.models_ready ? "models ready" : "warming models…") + " · narrator: " + (h.llm || "templates (offline)");
    if (!h.models_ready) setTimeout(health, 2500);
  } catch { $("#statusText").textContent = "server offline"; }
}

/* ------------------------------------------------------------------ config */
async function init() {
  CFG = await api("/api/config");
  const clubOpts = CFG.clubs.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
  $("#home").innerHTML = clubOpts; $("#away").innerHTML = clubOpts; $("#away").value = "RVA";
  const per = Object.entries(CFG.personas).map(([k, v]) => `<option value="${k}">${esc(v.split(" — ")[0])}</option>`).join("");
  const langs = Object.entries(CFG.languages).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  $$(".viewer").forEach(v => { $(".persona", v).innerHTML = per; $(".lang", v).innerHTML = langs; });
  $('.viewer[data-v=a] .persona').value = "analyst";
  $('.viewer[data-v=b] .persona').value = "casual"; $('.viewer[data-v=b] .lang').value = "es";
  $("#alang").innerHTML = langs;
  refreshClubSelects();
  ["#home", "#away"].forEach(s => $(s).onchange = refreshClubSelects);
  drawPitch([]);
  health();
}
function refreshClubSelects() {
  const h = $("#home").value, a = $("#away").value;
  $$(".viewer .club").forEach(sel => {
    const cur = sel.value;
    sel.innerHTML = `<option value="">No favourite club</option>` + CFG.clubs.filter(c => c.id === h || c.id === a).map(c => `<option value="${c.id}">${esc(c.short)}</option>`).join("");
    sel.value = [h, a].includes(cur) ? cur : "";
  });
  const hc = CFG.clubs.find(c => c.id === h), ac = CFG.clubs.find(c => c.id === a);
  document.documentElement.style.setProperty("--home", hc.colors[0]);
  document.documentElement.style.setProperty("--away", ac.colors[0]);
}
function profileOf(v) {
  const el = $(`.viewer[data-v=${v}]`);
  return { persona: $(".persona", el).value, language: $(".lang", el).value, favorite_club: $(".club", el).value || null, favorite_player: $(".player", el).value || null };
}

/* ------------------------------------------------------------------ live */
const PITCH = { L: 105, W: 68 };
let trail = [], momentum = [];
$("#kickoff").onclick = kickoff;
$("#pause").onclick = () => { PAUSED = !PAUSED; $("#pause").textContent = PAUSED ? "▶ Resume" : "❚❚ Pause"; if (!PAUSED && ES === null && MATCH) openStream(CUR_T); else if (PAUSED && ES) { ES.close(); ES = null; } };
$("#recapA").onclick = () => recap("a"); $("#recapB").onclick = () => recap("b");
$("#split").onchange = () => { $("#feedB").style.display = $("#split").checked ? "" : "none"; };

async function kickoff() {
  if (ES) { ES.close(); ES = null; }
  const body = { seed: +$("#seed").value || 7, home: $("#home").value, away: $("#away").value };
  if (body.home === body.away) { alert("Pick two different clubs"); return; }
  MATCH = await api("/api/matches", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const players = [...MATCH.lineups[MATCH.home.id], ...MATCH.lineups[MATCH.away.id]];
  $$(".viewer .player").forEach(sel => {
    const cur = sel.value;
    sel.innerHTML = `<option value="">No favourite player</option>` + players.map(p => `<option value="${p.id}">${esc(p.name)} · ${p.team} ${p.position}</option>`).join("");
    if (players.some(p => p.id === cur)) sel.value = cur;
  });
  $("#sbHome").textContent = MATCH.home.short.toUpperCase(); $("#sbAway").textContent = MATCH.away.short.toUpperCase();
  $$(".feed .lines").forEach(l => l.innerHTML = ""); $("#overlayA").innerHTML = "";
  trail = []; momentum = []; CUR_T = 0; PAUSED = false;
  $("#pause").disabled = false; $("#pause").textContent = "❚❚ Pause";
  openStream(0);
}
function openStream(start) {
  const profiles = { a: profileOf("a") };
  if ($("#split").checked) profiles.b = profileOf("b");
  $("#tagA").textContent = describe(profiles.a); $("#tagB").textContent = profiles.b ? describe(profiles.b) : "";
  const q = new URLSearchParams({ profiles: JSON.stringify(profiles), speed: $("#speed").value, start });
  ES = new EventSource(`/api/matches/${encodeURIComponent(MATCH.match_id)}/stream?${q}`);
  ES.onmessage = e => onPacket(JSON.parse(e.data));
  ES.onerror = () => { if (ES) { ES.close(); ES = null; } };
}
function describe(p) {
  const club = p.favorite_club ? " · ♥ " + (CFG.clubs.find(c => c.id === p.favorite_club)?.short || "") : "";
  const pl = p.favorite_player ? " · ★ " + ($(`.viewer .player option[value="${p.favorite_player}"]`)?.textContent.split(" · ")[0] || "") : "";
  return `${p.persona} · ${CFG.languages[p.language] || p.language}${club}${pl}`;
}

function onPacket(m) {
  if (m.done) { $("#ticker").textContent = "Full time — ask for a recap below."; ES && ES.close(); ES = null; if (m.snapshot) updateState(m.snapshot); return; }
  const ev = m.event; CUR_T = ev.timestamp;
  if (ev.location) { trail.push(ev); if (trail.length > 14) trail.shift(); drawPitch(trail); }
  $("#ticker").textContent = tickerText(ev);
  if (m.snapshot) updateState(m.snapshot, m.scorebug);
  for (const [v, out] of Object.entries(m.outputs || {})) {
    const box = $(v === "a" ? "#feedA .lines" : "#feedB .lines");
    for (const ln of out.lines) addLine(box, ln);
    for (const c of out.cues) showCue(c, v);
  }
  if (m.trace) { const box = $("#trace .lines"); for (const t of m.trace) { const d = document.createElement("div"); d.className = "tr"; d.innerHTML = `<b>${t.agent}</b> ${esc(t.msg)} <i>${t.ms} ms</i>`; box.prepend(d); } while (box.children.length > 120) box.lastChild.remove(); }
}
function tickerText(ev) {
  const mm = Math.floor(ev.timestamp / 60), ss = String(Math.floor(ev.timestamp % 60)).padStart(2, "0");
  let s = `${mm}:${ss}  ${ev.type.replace("_", " ").toUpperCase()}`;
  if (ev.player_name) s += ` · ${ev.player_name}`;
  if (ev.type === "pass") s += ` → ${ev.recipient_name || "lost"}  ${ev.length_m} m  diff ${(ev.difficulty * 100).toFixed(0)}%`;
  if (ev.type === "shot") s += `  xG ${ev.xg}  ${ev.shot_speed_kmh} km/h  ${ev.outcome}`;
  if (ev.type === "carry" && ev.speed_kmh) s += `  ${ev.speed_kmh} km/h`;
  return s;
}
function addLine(box, ln) {
  const d = document.createElement("div");
  d.className = "ln " + (ln.framing || "");
  d.innerHTML = `<div class="meta"><span>${esc(ln.kind.replace("_", " "))}</span>${ln.grounded ? '<span class="ok">✓ grounded</span>' : ""}</div><div>${esc(ln.text)}</div>`;
  box.prepend(d);
  while (box.children.length > 80) box.lastChild.remove();
}
function showCue(c, v) {
  if (v === "b" && !$("#split").checked) return;
  const layer = $("#overlayA");
  const d = document.createElement("div");
  d.className = `cue ${c.template} ${v} ${c.framing || ""}`;
  let extra = "";
  if (c.payload.factors) extra = `<div class="facs">${c.payload.factors.map(f => `<span class="fac">${esc(f.label)}: ${esc(f.display)}</span>`).join("")}</div>`;
  d.innerHTML = `<div class="k">${v.toUpperCase()} · ${esc(c.template.replace("_", " "))} · ${esc(c.clock)}</div><div class="t">${esc(c.text)}</div>${extra}`;
  layer.prepend(d);
  while (layer.children.length > 3) layer.lastChild.remove();
  setTimeout(() => d.remove(), Math.min(9, c.duration_s) * 1000);
}
function updateState(s, sb) {
  const H = MATCH.home.id, A = MATCH.away.id;
  $("#sbScore").textContent = `${s.score[H]} – ${s.score[A]}`; $("#sbClock").textContent = s.clock;
  const wp = s.win_probability;
  $(".wp-h").style.width = (100 * wp.home) + "%"; $(".wp-d").style.width = (100 * wp.draw) + "%"; $(".wp-a").style.width = (100 * wp.away) + "%";
  $("#wp").title = `Win probability — ${MATCH.home.short} ${(100 * wp.home).toFixed(0)}% · draw ${(100 * wp.draw).toFixed(0)}% · ${MATCH.away.short} ${(100 * wp.away).toFixed(0)}%`;
  const f = s.flow, ch = f.control[H] || 0, ca = f.control[A] || 0, chaos = f.chaos;
  const tot = ch + ca + chaos || 1;
  $("#flowCtrlH").style.width = (100 * ch / tot) + "%"; $("#flowChaos").style.width = (100 * chaos / tot) + "%"; $("#flowCtrlA").style.width = (100 * ca / tot) + "%";
  $("#flowLabel").textContent = f.label.startsWith("control:") ? `control · ${f.label.split(":")[1]}` : f.label;
  $("#tempo").textContent = `tempo ${f.tempo_per_min}/min · turnovers ${f.turnovers_per_min}/min`;
  momentum.push([s.t, s.momentum, wp.home]); if (momentum.length > 400) momentum.shift();
  drawMomentum();
  const th = s.teams[H], ta = s.teams[A];
  const rows = [["Possession %", "possession_pct"], ["xG", "xg"], ["Shots (on target)", "shots", "on_target"], ["Pass accuracy %", "pass_accuracy_pct"],
    ["Avg pass difficulty", "avg_pass_difficulty"], ["Passes over expected", "passes_over_expected"], ["Pressures", "pressures"], ["PPDA (lower = more pressing)", "ppda"], ["Max shot speed km/h", "max_shot_speed_kmh"]];
  $("#stats").innerHTML = rows.map(([lab, k, k2]) => {
    const a = th[k], b = ta[k], sa = Math.max(0, +a), sbv = Math.max(0, +b), tt = sa + sbv || 1;
    const fa = k2 ? `${a} (${th[k2]})` : a, fb = k2 ? `${b} (${ta[k2]})` : b;
    return `<div class="srow"><div class="lab">${lab}</div><div class="v">${fa}</div><div class="sbar"><div class="h" style="width:${100 * sa / tt}%"></div><div class="a" style="width:${100 * sbv / tt}%"></div></div><div class="v">${fb}</div></div>`;
  }).join("");
}

function drawPitch(tr) {
  const c = $("#pitch"), g = c.getContext("2d"), W = c.width, H = c.height, sx = W / PITCH.L, sy = H / PITCH.W;
  for (let i = 0; i < 10; i++) { g.fillStyle = i % 2 ? "#0f3b2a" : "#124631"; g.fillRect(i * W / 10, 0, W / 10, H); }
  g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = 2;
  g.strokeRect(4, 4, W - 8, H - 8);
  g.beginPath(); g.moveTo(W / 2, 4); g.lineTo(W / 2, H - 4); g.stroke();
  g.beginPath(); g.arc(W / 2, H / 2, 9.15 * sx, 0, Math.PI * 2); g.stroke();
  for (const side of [0, 1]) {
    const x = side ? W - 4 - 16.5 * sx : 4, x2 = side ? W - 4 - 5.5 * sx : 4;
    g.strokeRect(x, H / 2 - 20.16 * sy, 16.5 * sx, 40.32 * sy);
    g.strokeRect(x2, H / 2 - 9.16 * sy, 5.5 * sx, 18.32 * sy);
  }
  tr.forEach((e, i) => {
    const a = (i + 1) / tr.length, home = e.team_id === MATCH?.home.id;
    const col = home ? getComputedStyle(document.documentElement).getPropertyValue("--home") : getComputedStyle(document.documentElement).getPropertyValue("--away");
    const [x, y] = e.location;
    g.globalAlpha = a;
    if (e.end_location) {
      const [x2, y2] = e.end_location;
      g.strokeStyle = e.outcome === "incomplete" ? "#ff5d73" : "#ffffff"; g.lineWidth = 2; g.setLineDash(e.type === "carry" ? [6, 5] : []);
      g.beginPath(); g.moveTo(x * sx, H - y * sy); g.lineTo(x2 * sx, H - y2 * sy); g.stroke(); g.setLineDash([]);
    }
    g.fillStyle = col.trim() || "#fff"; g.beginPath(); g.arc(x * sx, H - y * sy, e.type === "shot" ? 11 : 7, 0, Math.PI * 2); g.fill();
    if (e.type === "shot" || e.type === "goal") { g.strokeStyle = "#ffb547"; g.lineWidth = 3; g.stroke(); }
    if (i === tr.length - 1 && e.player_name) { g.globalAlpha = 1; g.fillStyle = "#fff"; g.font = "600 16px Inter"; g.fillText(e.player_name, x * sx + 12, H - y * sy - 10); }
  });
  g.globalAlpha = 1;
}
function drawMomentum() {
  const c = $("#momentum"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "rgba(255,255,255,.12)"; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
  if (momentum.length < 2) return;
  const t0 = 0, t1 = Math.max(5700, momentum[momentum.length - 1][0]);
  const X = t => (t - t0) / (t1 - t0) * W;
  g.lineWidth = 2; g.strokeStyle = "#00e0a4"; g.beginPath();
  momentum.forEach(([t, m], i) => { const y = H / 2 - m * (H / 2 - 6); i ? g.lineTo(X(t), y) : g.moveTo(X(t), y); }); g.stroke();
  g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 1.5; g.beginPath();
  momentum.forEach(([t, , w], i) => { const y = H - 4 - w * (H - 8); i ? g.lineTo(X(t), y) : g.moveTo(X(t), y); }); g.stroke();
}
async function recap(v) {
  if (!MATCH) return;
  const p = profileOf(v);
  $("#recap").textContent = "Recap agent is writing…";
  const q = new URLSearchParams({ persona: p.persona, language: p.language, upto: CUR_T || 99999 });
  if (p.favorite_club) q.set("favorite_club", p.favorite_club);
  if (p.favorite_player) q.set("favorite_player", p.favorite_player);
  const r = await api(`/api/matches/${encodeURIComponent(MATCH.match_id)}/recap?${q}`);
  $("#recap").textContent = r.text;
  $("#recapMeta").textContent = `source: ${r.source} · grounded: ${r.grounding.ok ? "yes" : "NO — " + r.grounding.unsupported.join(", ")} · viewer ${v.toUpperCase()} (${describe(p)})`;
}

/* ------------------------------------------------------------------ autopsy lab */
let LIB = null, EP = null, EVD = null, CURSEL = null, playing = false, raf = null;
async function loadLibrary() {
  $("#scenarios").innerHTML = '<div class="muted">Training the explainers on synthetic chases and shots…</div>';
  LIB = await api("/api/hunts");
  $("#scenarios").innerHTML = LIB.map(s => `<div class="scn"><h4>${esc(s.title)}</h4><div class="rule">Planted rule: ${esc(s.rule)}</div>
    <div class="chips">${s.episodes.slice(0, 40).map(e => `<span class="chip o${e.outcome === "caught" || e.outcome === "goal" ? 1 : 0}" data-s="${s.scenario}" data-id="${e.id}" title="${e.outcome}">${e.id.split("-").pop()}</span>`).join("")}</div></div>`).join("");
  $$(".chip").forEach(c => c.onclick = () => { $$(".chip").forEach(x => x.classList.remove("active")); c.classList.add("active"); openEpisode(c.dataset.s, c.dataset.id); });
  const first = $(".chip"); if (first) first.click();
}
$("#apersona").onchange = () => CURSEL && openEpisode(...CURSEL);
$("#alang").onchange = () => CURSEL && openEpisode(...CURSEL);
$("#askLLM").onclick = () => CURSEL && openEpisode(CURSEL[0], CURSEL[1], true);
async function openEpisode(s, id, llm = false) {
  CURSEL = [s, id];
  playing = false; cancelAnimationFrame(raf); $("#play").textContent = "▶";
  $("#narration").textContent = llm ? "Narrator agent drafting; fact-check agent verifying every number…" : "…";
  const q = new URLSearchParams({ persona: $("#apersona").value, language: $("#alang").value, llm: llm ? 1 : 0 });
  const r = s === "upload" ? await api(`/api/clips/${encodeURIComponent(id)}?${q}`) : await api(`/api/hunts/${s}/${encodeURIComponent(id)}?${q}`);
  EP = r.episode; EVD = r.evidence;
  $("#epTitle").textContent = id; $("#epOutcome").textContent = EP.outcome;
  $("#epOutcome").className = "pill o" + (EP.outcome === "caught" || EP.outcome === "goal" ? 1 : 0);
  $("#narration").textContent = r.narration.text;
  const g = r.narration.grounding;
  $("#narrSrc").innerHTML = `source: <b>${esc(r.narration.source)}</b> · grounded: ${g.ok ? "✓ every number traced to the evidence" : "✗ " + esc(g.unsupported.join(", "))} · model p(${esc(EVD.outcome)}) = ${esc(EVD.model.p_outcome_display)}`;
  const maxc = Math.max(...EVD.top_factors.map(f => Math.abs(f.contribution)), 0.01);
  $("#factors").innerHTML = EVD.top_factors.map(f => `<div class="fac-row"><div class="nm">${esc(f.label)}</div><div class="val">${esc(f.display)}</div>
    <div class="bar"><div style="width:${100 * Math.max(0, f.contribution) / maxc}%"></div></div>
    <div class="sim">${esc(f.direction)}${f.similar ? " · " + esc(f.similar.display) : ""}</div></div>`).join("");
  $("#cf").innerHTML = EVD.counterfactual ? `<b>Counterfactual.</b> ${esc(EVD.counterfactual.display)}.` : "";
  const tc = r.true_causes;
  if (tc && tc.length) {
    const hit = tc.includes(EVD.top_factors[0].feature);
    $("#truth").innerHTML = `Ground truth (synthetic): planted cause = <b>${esc(tc.join(" / "))}</b> → explainer's top factor <span class="${hit ? "hit" : "miss"}">${hit ? "MATCHES" : "differs"}</span>`;
  } else $("#truth").textContent = "";
  const vid = $("#video");
  if (EP.video) { vid.src = EP.video; vid.hidden = false; } else { vid.hidden = true; vid.removeAttribute("src"); }
  setupArena(); drawCurve();
}
let T0 = 0, T1 = 1, BOUNDS = null;
function setupArena() {
  const xs = EP.tracks.flatMap(t => t.x), ys = EP.tracks.flatMap(t => t.y);
  T0 = Math.min(...EP.tracks.map(t => t.t[0])); T1 = Math.max(...EP.tracks.map(t => t.t[t.t.length - 1]));
  const pad = EP.domain === "football" ? 3 : 2;
  BOUNDS = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
  $("#scrub").value = 0; drawArena(T0);
}
function interp(tr, t) {
  const ts = tr.t; if (t <= ts[0]) return [tr.x[0], tr.y[0]]; if (t >= ts[ts.length - 1]) return [tr.x.at(-1), tr.y.at(-1)];
  let i = ts.findIndex(v => v > t); const a = (t - ts[i - 1]) / (ts[i] - ts[i - 1]);
  return [tr.x[i - 1] + a * (tr.x[i] - tr.x[i - 1]), tr.y[i - 1] + a * (tr.y[i] - tr.y[i - 1])];
}
function drawArena(t) {
  const c = $("#arena"), g = c.getContext("2d"), W = c.width, H = c.height;
  const [x0, x1, y0, y1] = BOUNDS, s = Math.min(W / (x1 - x0), H / (y1 - y0));
  const X = x => (x - x0) * s + (W - (x1 - x0) * s) / 2, Y = y => (y - y0) * s + (H - (y1 - y0) * s) / 2;
  g.fillStyle = EP.domain === "football" ? "#0f3b2a" : "#1b1f16"; g.fillRect(0, 0, W, H);
  g.strokeStyle = "rgba(255,255,255,.05)"; g.lineWidth = 1;
  for (let gx = Math.ceil(x0); gx < x1; gx += 2) { g.beginPath(); g.moveTo(X(gx), 0); g.lineTo(X(gx), H); g.stroke(); }
  for (let gy = Math.ceil(y0); gy < y1; gy += 2) { g.beginPath(); g.moveTo(0, Y(gy)); g.lineTo(W, Y(gy)); g.stroke(); }
  if (EP.domain === "football") { g.strokeStyle = "#fff"; g.lineWidth = 4; g.beginPath(); g.moveTo(X(105), Y(30.34)); g.lineTo(X(105), Y(37.66)); g.stroke(); }
  for (const tr of EP.tracks) {
    const col = ROLE_COL[tr.role] || "#aaa";
    g.strokeStyle = col; g.globalAlpha = .55; g.lineWidth = tr.role === "attacker" || tr.role === "defender" ? 2.5 : 1.2;
    g.beginPath(); let first = true;
    for (let i = 0; i < tr.t.length && tr.t[i] <= t; i++) { const px = X(tr.x[i]), py = Y(tr.y[i]); first ? g.moveTo(px, py) : g.lineTo(px, py); first = false; }
    g.stroke(); g.globalAlpha = 1;
    const [px, py] = interp(tr, t);
    g.fillStyle = col; g.beginPath(); g.arc(X(px), Y(py), tr.role === "attacker" || tr.role === "defender" ? 8 : 5, 0, Math.PI * 2); g.fill();
    if (tr.role === "attacker" || tr.role === "defender") {
      g.fillStyle = "#fff"; g.font = "600 13px Inter";
      const name = EP.domain === "football" ? (tr.role === "attacker" ? "shooter" : "keeper") : (tr.role === "attacker" ? "predator" : "prey");
      g.fillText(name, X(px) + 11, Y(py) - 9);
    }
  }
  const att = EP.tracks.find(x => x.role === "attacker"), def = EP.tracks.find(x => x.role === "defender");
  for (const e of EP.events) {
    if (e.t > t || ["stalk"].includes(e.type)) continue;
    const tr = e.actor === def?.actor ? def : att; if (!tr) continue;
    const [ex, ey] = interp(tr, e.t);
    g.strokeStyle = "#ffb547"; g.lineWidth = 2; g.beginPath(); g.arc(X(ex), Y(ey), 12, 0, Math.PI * 2); g.stroke();
    g.fillStyle = "#ffb547"; g.font = "11px JetBrains Mono"; g.fillText(e.type, X(ex) + 13, Y(ey) + 14);
  }
  const dm = EVD.decisive_moments[0];
  if (dm && Math.abs(t - dm.t) < 0.35) { g.fillStyle = "rgba(255,181,71,.12)"; g.fillRect(0, 0, W, H); g.fillStyle = "#ffb547"; g.font = "700 16px Inter"; g.fillText("DECISIVE MOMENT · " + dm.t_display, 16, 26); }
  $("#tnow").textContent = t.toFixed(2) + " s";
  drawCurve(t);
}
function drawCurve(t) {
  const c = $("#curve"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  if (!EP) return;
  const X = x => (x - T0) / (T1 - T0) * W;
  const dm = EVD.decisive_moments[0];
  if (dm) { g.fillStyle = "rgba(255,181,71,.18)"; g.fillRect(X(dm.t), 0, Math.max(4, X(dm.t + 0.3) - X(dm.t)), H); }
  g.strokeStyle = "rgba(255,255,255,.1)"; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
  for (const e of EP.events) { g.fillStyle = "#8798b0"; g.fillRect(X(e.t) - 1, H - 14, 2, 14); }
  if (EP.curve) {
    const pos = EP.outcome === "caught" || EP.outcome === "goal";
    g.strokeStyle = "#00e0a4"; g.lineWidth = 2.5; g.beginPath();
    EP.curve.t.forEach((tt, i) => { const p = pos ? EP.curve.p[i] : 1 - EP.curve.p[i]; const y = H - 6 - p * (H - 20); i ? g.lineTo(X(tt), y) : g.moveTo(X(tt), y); });
    g.stroke();
  }
  if (t !== undefined) { g.strokeStyle = "#fff"; g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), H); g.stroke(); }
}
$("#scrub").oninput = e => { const t = T0 + (T1 - T0) * e.target.value / 1000; drawArena(t); };
$("#play").onclick = () => {
  playing = !playing; $("#play").textContent = playing ? "❚❚" : "▶";
  if (!playing) { cancelAnimationFrame(raf); return; }
  let t = T0 + (T1 - T0) * $("#scrub").value / 1000; if (t >= T1 - 0.01) t = T0;
  let last = performance.now();
  const tick = now => { if (!playing) return; t += (now - last) / 1000 * 0.5; last = now; if (t > T1) { playing = false; $("#play").textContent = "▶"; return; }
    $("#scrub").value = 1000 * (t - T0) / (T1 - T0); drawArena(t); raf = requestAnimationFrame(tick); };
  raf = requestAnimationFrame(tick);
};

/* ------------------------------------------------------------------ upload your own clip */
const MY = (() => { try { return JSON.parse(localStorage.getItem("myClips") || "[]"); } catch { return []; } })();
function saveMy() { try { localStorage.setItem("myClips", JSON.stringify(MY.slice(-20))); } catch {} }
function renderMy() {
  $("#myClips").innerHTML = MY.map(c => `<span class="chip o${c.outcome === "caught" ? 1 : 0}" data-s="upload" data-id="${c.id}" title="${esc(c.name)}">${esc(c.name.slice(0, 18))}</span>`).join("");
  $$("#myClips .chip").forEach(c => c.onclick = () => { $$(".chip").forEach(x => x.classList.remove("active")); c.classList.add("active"); openEpisode("upload", c.dataset.id); });
}
const drop = $("#drop");
["dragover", "dragenter"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => { if (ev.dataTransfer.files[0]) { $("#clipFile").files = ev.dataTransfer.files; $("#dropText").textContent = ev.dataTransfer.files[0].name; } });
$("#clipFile").onchange = () => { const f = $("#clipFile").files[0]; $("#dropText").textContent = f ? `${f.name} · ${(f.size / 1e6).toFixed(1)} MB` : "Drop a video"; };
$("#uploadForm").onsubmit = async ev => {
  ev.preventDefault();
  const f = $("#clipFile").files[0], st = $("#clipStatus");
  st.classList.remove("err");
  if (!f || !$("#clipOutcome").value) { st.textContent = "Choose a video and its known outcome."; st.classList.add("err"); return; }
  const fd = new FormData();
  fd.append("file", f); fd.append("outcome", $("#clipOutcome").value); fd.append("detector", $("#clipDetector").value);
  if ($("#clipAnimals").value.trim()) fd.append("animals", $("#clipAnimals").value.trim());
  if ($("#clipStart").value) fd.append("start_s", $("#clipStart").value);
  if ($("#clipEnd").value) fd.append("end_s", $("#clipEnd").value);
  $("#clipGo").disabled = true; st.textContent = "Uploading…";
  try {
    const r = await fetch("/api/clips", { method: "POST", body: fd });
    const job = await r.json();
    if (!r.ok) throw new Error(job.detail || r.status);
    MY.push({ id: job.id, name: f.name, outcome: job.outcome }); saveMy(); renderMy();
    pollClip(job.id);
  } catch (e) { st.textContent = "Upload failed: " + e.message; st.classList.add("err"); $("#clipGo").disabled = false; }
};
async function pollClip(id) {
  const st = $("#clipStatus");
  const j = await api(`/api/clips/${id}?persona=${$("#apersona").value}&language=${$("#alang").value}`).catch(e => ({ status: "error", error: e.message }));
  if (j.status === "error") { st.textContent = "Analysis failed: " + j.error; st.classList.add("err"); $("#clipGo").disabled = false; return; }
  if (j.status !== "done") { st.textContent = "Analysing… " + j.step; setTimeout(() => pollClip(id), 1500); return; }
  st.textContent = "Done — " + (j.model_note || "");
  $("#clipGo").disabled = false;
  $$(".chip").forEach(x => x.classList.remove("active"));
  $(`#myClips .chip[data-id="${id}"]`)?.classList.add("active");
  openEpisode("upload", id);
}
renderMy();

/* ------------------------------------------------------------------ evals */
async function loadEvals() {
  const r = await api("/api/evals");
  const tb = $("#evals tbody");
  if (!r.available) { tb.innerHTML = `<tr><td colspan=6 class="muted">No results yet — run <span class="mono">python -m evals.run_all</span></td></tr>`; return; }
  $("#evalWhen").textContent = "run " + r.when + " · " + r.seconds + " s";
  tb.innerHTML = r.results.map(x => `<tr><td>${x.id}</td><td><b>${esc(x.name)}</b><div class="muted small">${esc(x.how)}</div></td><td>${esc(x.layer)}</td><td class="mono">${esc(x.result)}</td><td>${esc(x.bar)}</td><td><span class="badge ${x.status}">${x.status}</span></td></tr>`).join("");
}

init();
