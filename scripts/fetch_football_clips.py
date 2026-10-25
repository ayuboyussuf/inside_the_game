"""Download the openly licensed football clips used by eval 11 (Wikimedia Commons) into data/raw/football/.

    python scripts/fetch_football_clips.py

Clips are not committed to the repo; licences and source pages are written to data/raw/football/SOURCES.json.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "football"
UA = {"User-Agent": "HuntAutopsy/0.1 (hackathon research tool)"}
CLIPS = {
    "fra_ger_open_play.ogv": "Match de football France-Allemagne - 16 octobre 2018 - Phase de jeu (1).ogv",
    "fra_ger_kroos_penalty.ogv": "Match de football France-Allemagne - 16 octobre 2018 - Penalty de Toni Kroos.ogv",
    "keller_saves_henry_pk.ogv": "Keller saves PK attempt by Henry.ogv",
    "valencia_wc2022_goal.webm": "2022 FIFA World Cup's first goal by Enner Valencia of Ecuador against Qatar.webm",
    "latvia_gibraltar.webm": "Latvia-Gibraltar football 2026-03-31.webm",
    "everton_oxford_highlights.webm": "LADIES HIGHLIGHTS- EVERTON 4-0 OXFORD.webm",
    "amateur_shot_topdown.webm": "Amateurfußball Torschuss von oben.webm",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for fn, title in CLIPS.items():
        for attempt in range(6):
            try:
                q = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
                    {"action": "query", "titles": "File:" + title, "prop": "imageinfo", "iiprop": "url|extmetadata|size", "format": "json"})
                ii = list(json.load(urllib.request.urlopen(urllib.request.Request(q, headers=UA)))["query"]["pages"].values())[0]["imageinfo"][0]
                meta[fn] = {"title": title, "url": ii["descriptionurl"], "licence": ii["extmetadata"]["LicenseShortName"]["value"],
                            "size_mb": round(ii["size"] / 1e6, 1)}
                path = OUT / fn
                if not (path.exists() and abs(os.path.getsize(path) - ii["size"]) < 1000):
                    with urllib.request.urlopen(urllib.request.Request(ii["url"].split("?")[0], headers=UA)) as r, open(path, "wb") as fh:
                        fh.write(r.read())
                print(fn, meta[fn]["licence"])
                break
            except Exception as e:  # Commons rate-limits bursts (HTTP 429): back off and retry
                print(fn, "retry", e)
                time.sleep(20 * (attempt + 1))
        time.sleep(5)
    (OUT / "SOURCES.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
