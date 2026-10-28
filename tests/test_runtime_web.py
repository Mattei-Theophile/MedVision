"""Boucle principale et interface web, avec un capteur scripté (ni matériel ni réseau)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from medvision.config import Config
from medvision.features import HAND_FORMAT, RADAR_FORMAT
from medvision.runtime import Runtime
from medvision.sensors.base import Observation, SensorAdapter
from medvision.store import GestureStore
from medvision.web import WebServer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ScriptedSensor(SensorAdapter):
    kind = "script"
    feature_format = HAND_FORMAT
    presence_name = "Main détectée"

    def __init__(self, cfg, clock, frames=()):
        super().__init__(cfg)
        self.clock, self.frames_left = clock, list(frames)

    def open(self):
        pass

    def read(self):
        self.clock.t += 1 / 15
        if not self.frames_left:
            return Observation(features=None, present=False, live={"hands": []})
        f = self.frames_left.pop(0)
        self.frames += 1
        return Observation(features=f, present=f is not None, live={"hands": []})

    def close(self):
        pass


class FakeHA:
    def __init__(self):
        self.calls = []
        self.labels = {}

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


@pytest.fixture
def setup(tmp_path, gesture):
    store = GestureStore(tmp_path)
    for kind in ("poing", "vague", "repos"):
        store.save(kind, [gesture(kind, frames=22, seed=s) for s in range(4)], label=kind.title())
    # Geste radar : ne doit pas être chargé avec un capteur « main ».
    store.save("approche", [[[0, -1, -0.5]] * 8] * 2, feature_format=RADAR_FORMAT)
    cfg = Config(gestures_dir=str(tmp_path))
    clock = Clock()
    sensor = ScriptedSensor(cfg, clock)
    ha = FakeHA()
    rt = Runtime(cfg, sensor=sensor, store=store, publisher=ha, clock=clock)
    rt.open()
    return rt, sensor, ha, store


def test_runtime_loads_only_gestures_of_the_sensor_format(setup):
    rt, _, ha, _ = setup
    assert rt.lib.names == ["poing", "repos", "vague"]
    assert rt.command_names() == ["poing", "vague"]
    assert ha.calls[0][0] == "start" and ha.calls[0][1][0] == ["poing", "vague"]
    by_name = {g["name"]: g for g in rt.gestures()}
    assert by_name["poing"]["compatible"] and by_name["poing"]["quality"] == "bon"
    assert not by_name["approche"]["compatible"] and by_name["approche"]["quality"] is None
    assert by_name["repos"]["rest"]


def test_runtime_recognises_and_publishes(setup, gesture):
    rt, sensor, ha, _ = setup
    sensor.frames_left = [None] * 5 + list(gesture("vague", frames=22, seed=99)) + [None] * 20
    for _ in range(len(sensor.frames_left)):
        rt.step()
    snap = rt.snapshot()
    assert [h["name"] for h in snap["history"]] == ["vague"]
    assert snap["history"][0]["label"] == "Vague"
    assert any(c[0] == "publish_gesture" for c in ha.calls)
    assert json.dumps(snap)  # sérialisable pour l'API


def test_recording_through_runtime_saves_and_reloads(setup, gesture):
    rt, sensor, ha, store = setup
    rt.start_recording(name="pouce", label="Pouce levé", samples=2, duration=1.0, countdown=0.5)
    with pytest.raises(ValueError, match="déjà en cours"):
        rt.start_recording(name="autre")
    sensor.frames_left = list(gesture("pouce", frames=50, seed=5)) + [None] * 30
    events_before = len(rt.history)
    for _ in range(80):
        rt.step()
    rec = rt.snapshot()["recording"]
    assert rec["phase"] == "done", rec
    assert "pouce" in rt.lib.names and store.labels()["pouce"] == "Pouce levé"
    assert len(rt.history) == events_before  # pas de reconnaissance pendant l'enregistrement
    assert any(c[0] == "update_gestures" for c in ha.calls)


def test_recording_refuses_name_used_by_other_sensor(setup):
    rt, *_ = setup
    with pytest.raises(ValueError, match="autre capteur"):
        rt.start_recording(name="approche")
    rt.start_recording(name="approche", replace=True)  # remplacement explicite accepté


def test_delete_triggers_reload(setup):
    rt, *_ = setup
    assert rt.delete_gesture("poing")
    rt.step()
    assert "poing" not in rt.lib.names
    assert not rt.delete_gesture("poing")


# ------------------------------------------------------------------------- web
@pytest.fixture
def web(setup):
    rt = setup[0]
    server = WebServer(rt, "127.0.0.1", 0).start()
    host, port = server.address
    yield rt, f"http://{host}:{port}/"
    server.stop()


def _req(url, method="GET", body=None, ctype="application/json"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if ctype and body is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if "json" in r.headers["Content-Type"] else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_web_serves_page_and_state(web):
    rt, base = web
    status, page = _req(base)
    assert status == 200 and b"MedVision" in page and b"api/state" in page
    status, state = _req(base + "api/state")
    assert status == 200 and state["sensor"]["kind"] == "script" and state["version"]
    status, data = _req(base + "api/gestures")
    assert {g["name"] for g in data["gestures"]} >= {"poing", "approche"}


def test_web_record_validation_and_cancel(web):
    rt, base = web
    status, err = _req(base + "api/record", "POST", {"name": "Mauvais nom!"})
    assert status == 400 and "invalide" in err["error"]
    status, rec = _req(base + "api/record", "POST", {"name": "lumiere", "samples": 3})
    assert status == 202 and rec["phase"] == "countdown" and rec["total"] == 3
    assert _req(base + "api/record/cancel", "POST", {})[0] == 200
    assert rt.recorder.phase.value == "cancelled"


def test_web_requires_json_for_changes(web):
    rt, base = web
    status, _ = _req(base + "api/gestures/poing", "DELETE", {}, ctype="text/plain")
    assert status == 415
    assert "poing" in GestureStore(rt.cfg.gestures_dir).list()
    assert _req(base + "api/gestures/poing", "DELETE", {})[0] == 200
    assert _req(base + "api/gestures/poing", "DELETE", {})[0] == 404
    assert _req(base + "api/gestures/..%2Fetc", "DELETE", {})[0] == 400
    assert _req(base + "api/inconnu")[0] == 404


def test_run_loop_stops(setup):
    rt, *_ = setup
    stop = threading.Event()
    t = threading.Thread(target=rt.run, args=(stop,))
    t.start()
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive()
