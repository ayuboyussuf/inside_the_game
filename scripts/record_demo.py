"""Record a < 2 minute captioned demo of the running app (Playwright + Chromium).

    python scripts/record_demo.py            # starts the app on :8010, records, writes docs/demo/hunt_autopsy_demo.mp4

Add your own voice-over afterwards if you like; captions carry the story without audio.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
PORT = 8010
W, H = 1440, 900

CAPTION_JS = """(t) => {
  let el = document.getElementById('demo-cap');
  if (!el) { el = document.createElement('div'); el.id = 'demo-cap'; document.body.appendChild(el);
    Object.assign(el.style, {position:'fixed', left:'50%', bottom:'26px', transform:'translateX(-50%)', zIndex:9999,
      background:'rgba(4,10,18,.92)', color:'#fff', font:'600 22px Inter, system-ui', padding:'14px 22px',
      borderRadius:'14px', border:'1px solid rgba(0,224,164,.6)', maxWidth:'1100px', textAlign:'center',
      boxShadow:'0 10px 40px rgba(0,0,0,.5)', transition:'opacity .3s'}); }
  el.innerHTML = t; el.style.opacity = t ? 1 : 0; }"""
TITLE_JS = """([title, sub]) => {
  let el = document.getElementById('demo-title');
  if (!el) { el = document.createElement('div'); el.id = 'demo-title'; document.body.appendChild(el); }
  Object.assign(el.style, {position:'fixed', inset:0, zIndex:10000, display:'flex', flexDirection:'column',
    alignItems:'center', justifyContent:'center', background:'radial-gradient(circle at 50% 40%, #12324a, #05080d 70%)',
    color:'#fff', font:'800 64px Inter, system-ui', textAlign:'center', transition:'opacity .6s'});
  el.innerHTML = `<div>${title}</div><div style="font:500 26px Inter;color:#9fe8d2;margin-top:18px;max-width:1000px">${sub}</div>`;
  el.style.opacity = 1; el.style.display = 'flex'; }"""
HIDE_TITLE_JS = "() => { const el = document.getElementById('demo-title'); if (el) { el.style.opacity = 0; setTimeout(() => el.style.display='none', 600); } }"


def wait_ready(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if b'"models_ready":true' in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=3).read():
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("app did not become ready")


def main():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready()
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT / "raw"),
                                record_video_size={"width": W, "height": H}, color_scheme="dark")
            pg = ctx.new_page()
            cap = lambda t: pg.evaluate(CAPTION_JS, t)
            pg.goto(f"http://127.0.0.1:{PORT}/")
            pg.wait_for_selector("#kickoff")
            # 0 ─ title
            pg.evaluate(TITLE_JS, ["Hunt Autopsy", "Explainable match intelligence: it explains <b>why</b> a moment happened,"
                                   " not just that it happened. Synthetic football events in, personalised insight out."])
            pg.wait_for_timeout(5500)
            pg.evaluate(HIDE_TITLE_JS)
            # 1 ─ set up two viewers
            cap("Two viewers, one live stream: an <b>analyst in English</b> and a <b>casual fan in Spanish</b> who supports Northbridge")
            pg.select_option('.viewer[data-v=a] .persona', 'analyst')
            pg.select_option('.viewer[data-v=a] .lang', 'en')
            pg.select_option('.viewer[data-v=b] .persona', 'casual')
            pg.select_option('.viewer[data-v=b] .lang', 'es')
            pg.select_option('#speed', '120')
            pg.fill('#seed', '7')
            pg.wait_for_timeout(2500)
            pg.select_option('.viewer[data-v=b] .club', 'NBR')
            pg.wait_for_timeout(2500)
            pg.click('#kickoff')
            # 2 ─ live broadcast view
            cap("Synthetic events are ingested live. Eight agents turn them into stats, key moments and timed on-screen overlays")
            pg.wait_for_timeout(9000)
            cap("Every pass gets a <b>difficulty rating</b>; shots get xG and speed; the flow meter shows <b>control vs chaos</b>; win probability updates live")
            pg.wait_for_timeout(9000)
            cap("When a shot is saved or scored, the <b>autopsy agent</b> explains why the keeper could or couldn't reach it")
            pg.wait_for_timeout(8000)
            # 3 ─ personalised feeds + agent trace
            pg.evaluate("document.querySelector('.feeds').scrollIntoView({behavior:'smooth', block:'center'})")
            cap("Same intelligence, two genuinely different experiences: numbers-heavy for the analyst, short and emotional in Spanish for the fan")
            pg.wait_for_timeout(8000)
            cap("The <b>fact-check agent</b> rejects any sentence with a number that is not in the evidence. Every line is marked ✓ grounded")
            pg.wait_for_timeout(7000)
            # 4 ─ recap
            pg.evaluate("document.querySelector('.recapcard').scrollIntoView({behavior:'smooth', block:'center'})")
            pg.click('#recapB')
            pg.wait_for_function("!document.querySelector('#recap').textContent.includes('writing')", timeout=60000)
            cap("One click: an automated <b>match recap</b>, written for this viewer's language and favourite club")
            pg.wait_for_timeout(7000)
            # 5 ─ autopsy lab
            pg.evaluate("window.scrollTo({top:0, behavior:'smooth'})")
            pg.click('.tab[data-tab=autopsy]')
            pg.wait_for_selector('.chip[data-s=football]', timeout=120000)
            # show a saved shot where the explainer's top factor matches the planted cause (and say so only then)
            chips = pg.query_selector_all('.chip[data-s=football][title=saved]') or pg.query_selector_all('.chip[data-s=football]')
            matched = False
            for c in chips[:12]:
                c.click()
                pg.wait_for_function("document.querySelector('#narration').textContent.length > 20 && "
                                     "!document.querySelector('#narration').textContent.startsWith('…')", timeout=60000)
                pg.wait_for_timeout(400)
                if "MATCHES" in (pg.text_content('#truth') or ""):
                    matched = True
                    break
            cap("The <b>Autopsy lab</b>: why was this shot saved? The top factors, their values and the evidence from similar shots")
            pg.wait_for_timeout(3000)
            pg.click('#play')
            pg.wait_for_timeout(6000)
            cap("The decisive moment is highlighted on the timeline, and a counterfactual shows what would have changed the outcome")
            pg.wait_for_timeout(6500)
            cap("Shots are generated with a <b>planted keeper model</b>, so we can check the explanation against the true cause"
                + (": it matches" if matched else ""))
            pg.wait_for_timeout(6500)
            # 6 ─ evals
            pg.click('.tab[data-tab=evals]')
            cap("Eleven built-in evals prove the explanations are the <b>right ones</b>, not just plausible. Real-video limits are reported honestly")
            pg.wait_for_timeout(8000)
            cap("")
            pg.evaluate(TITLE_JS, ["Hunt Autopsy", "Built with FastAPI · LightGBM + TreeSHAP · YOLO + TAPNet · Azure OpenAI / GitHub Models · Azure Container Apps<br>"
                                   "github.com/ayuboyussuf/inside_the_game"])
            pg.wait_for_timeout(5000)
            vid = pg.video.path()
            ctx.close()
            b.close()
        raw = Path(vid)
        mp4 = OUT / "hunt_autopsy_demo.mp4"
        try:
            import imageio_ffmpeg
            ff = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "medium", "-crf", "22", "-movflags", "+faststart", str(mp4)], check=True)
        except Exception as e:
            print("mp4 conversion failed, keeping webm:", e)
            mp4 = raw
        print("demo:", mp4)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
