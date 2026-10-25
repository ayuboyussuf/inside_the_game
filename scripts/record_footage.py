"""Record clean 1920x1080 app footage (no captions) for the Remotion video, with scene markers.

    python scripts/record_footage.py      # -> video/public/footage.mp4 + video/public/markers.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_demo import PORT, ROOT, wait_ready  # noqa: E402

PUB = ROOT / "video" / "public"
W, H = 1920, 1080


def main():
    from playwright.sync_api import sync_playwright
    PUB.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "docs" / "demo" / "raw_footage"
    shutil.rmtree(raw_dir, ignore_errors=True)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    marks = {}
    try:
        wait_ready()
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(raw_dir),
                                record_video_size={"width": W, "height": H}, color_scheme="dark")
            t0 = time.monotonic()
            mark = lambda k: marks.__setitem__(k, round(time.monotonic() - t0, 2))
            pg = ctx.new_page()
            pg.goto(f"http://127.0.0.1:{PORT}/")
            pg.wait_for_selector("#kickoff")
            pg.select_option('.viewer[data-v=a] .persona', 'analyst')
            pg.select_option('.viewer[data-v=a] .lang', 'en')
            pg.select_option('.viewer[data-v=b] .persona', 'casual')
            pg.select_option('.viewer[data-v=b] .lang', 'es')
            pg.select_option('.viewer[data-v=b] .club', 'NBR')
            pg.select_option('#speed', '120')
            pg.fill('#seed', '7')
            pg.wait_for_timeout(800)
            mark("setup")
            pg.wait_for_timeout(2500)
            pg.click('#kickoff')
            mark("live")
            pg.wait_for_timeout(36000)
            pg.evaluate("document.querySelector('.feeds').scrollIntoView({block:'start'})")
            pg.wait_for_timeout(300)
            mark("feeds")
            pg.wait_for_timeout(12000)
            pg.evaluate("document.querySelector('.recapcard').scrollIntoView({block:'center'})")
            pg.click('#recapB')
            pg.wait_for_function("!document.querySelector('#recap').textContent.includes('writing')", timeout=60000)
            mark("recap")
            pg.wait_for_timeout(6000)
            pg.evaluate("window.scrollTo(0,0)")
            pg.click('.tab[data-tab=autopsy]')
            pg.wait_for_selector('.chip[data-s=football]', timeout=120000)
            chips = pg.query_selector_all('.chip[data-s=football][title=saved]')
            for c in chips[:12]:
                c.click()
                pg.wait_for_function("document.querySelector('#narration').textContent.length > 20 && "
                                     "!document.querySelector('#narration').textContent.startsWith('…')", timeout=60000)
                pg.wait_for_timeout(400)
                if "MATCHES" in (pg.text_content('#truth') or ""):
                    break
            pg.evaluate("document.querySelector('.playback').scrollIntoView({block:'start'})")
            pg.wait_for_timeout(500)
            mark("autopsy")
            pg.click('#play')
            pg.wait_for_timeout(12000)
            pg.evaluate("window.scrollTo(0,0)")
            pg.click('.tab[data-tab=evals]')
            pg.wait_for_timeout(500)
            mark("evals")
            pg.wait_for_timeout(7000)
            mark("end")
            vid = pg.video.path()
            ctx.close()
            b.close()
        import imageio_ffmpeg
        ff = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vid), "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(PUB / "footage.mp4")], check=True)
        # the browser video starts later than our clock: shift markers so the last one lands at the file's end
        import cv2
        cap = cv2.VideoCapture(str(PUB / "footage.mp4"))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
        cap.release()
        off = marks["end"] - dur
        marks = {k: round(v - off, 2) for k, v in marks.items()}
        (PUB / "markers.json").write_text(json.dumps(marks, indent=1))
        print("markers", marks)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
