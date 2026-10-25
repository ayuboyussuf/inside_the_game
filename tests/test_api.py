import json
import os

os.environ.setdefault("HUNT_AUTOPSY_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_api_smoke():
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    cfg = c.get("/api/config").json()
    assert cfg["clubs"] and "analyst" in cfg["personas"]
    m = c.post("/api/matches", json={"seed": 5, "home": "NBR", "away": "HLU"}).json()
    mid = m["match_id"]
    assert m["n_events"] > 1000
    # stream at max speed and read a few packets
    profiles = json.dumps({"a": {"persona": "analyst"}, "b": {"persona": "casual", "language": "fr"}})
    with c.stream("GET", f"/api/matches/{mid}/stream", params={"profiles": profiles, "speed": 0}) as r:
        n = 0
        for line in r.iter_lines():
            if line.startswith("data: "):
                pkt = json.loads(line[6:])
                n += 1
                if n > 50:
                    break
        assert n > 50
    rec = c.get(f"/api/matches/{mid}/recap", params={"persona": "casual", "language": "de", "upto": 3000}).json()
    assert rec["grounding"]["ok"] and rec["text"]
    assert "properties" in c.get("/api/overlay-schema").json()
