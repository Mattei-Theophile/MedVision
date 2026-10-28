import json

import numpy as np
import pytest

from medvision.config import Config
from medvision.features import FORMATS, HAND_FORMAT, RADAR_FORMAT
from medvision.sensors import (
    Observation,
    SensorAdapter,
    available_sensors,
    create_sensor,
    register_sensor,
    sensor_class,
)
from medvision.sensors.camera import CameraAdapter
from medvision.sensors.simulation import SimulatedAdapter
from medvision.store import GestureStore
from medvision.vision import HandObservation


def test_factory_builds_each_builtin_adapter():
    assert {"camera", "ld2450", "simulation"} <= set(available_sensors())
    assert isinstance(create_sensor(Config(sensor_type="camera")), CameraAdapter)
    assert create_sensor(Config(sensor_type="LD2450")).feature_format == RADAR_FORMAT
    with pytest.raises(ValueError, match="Capteur inconnu"):
        sensor_class("lidar")


def test_every_adapter_declares_a_known_format():
    for kind in ("camera", "ld2450"):
        cls = sensor_class(kind)
        assert cls.feature_format in FORMATS and cls.kind == kind and cls.presence_name


def test_register_third_party_adapter():
    @register_sensor("bouton-test", "Capteur factice")
    class Dummy(SensorAdapter):
        feature_format = HAND_FORMAT

        def open(self):
            pass

        def read(self):
            return Observation(features=None, present=False)

        def close(self):
            pass

    assert available_sensors()["bouton-test"] == "Capteur factice"
    with create_sensor(Config(sensor_type="bouton-test")) as s:
        assert s.read().present is False
    with pytest.raises(TypeError):
        register_sensor("x")(object)


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        pass


class FakeTracker:
    def __init__(self, results):
        self.results = list(results)

    def detect(self, frame):
        return self.results.pop(0)

    def close(self):
        pass


def test_camera_adapter_wraps_legacy_classes(gesture):
    feats = gesture("poing", frames=2)
    raw = np.zeros((21, 3), dtype=np.float32)
    hand = HandObservation(features=feats[0], raw=raw, handedness="Right")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cam = CameraAdapter(
        Config(motion_gating=False),
        camera=FakeCamera([frame, frame, None]),
        tracker=FakeTracker([[hand], []]),
    )
    o1, o2, o3 = cam.read(), cam.read(), cam.read()
    assert o1.present and np.array_equal(o1.features, feats[0])
    assert len(o1.live["hands"][0]) == 21
    assert not o2.present and o2.features is None and o2.live == {"hands": []}
    assert o3 is None  # échec passager de lecture
    assert cam.render(o1, "ok").shape == frame.shape
    assert (cam.frames, cam.analysed) == (2, 2)


def test_simulation_replays_recorded_gestures(tmp_path, gesture):
    store = GestureStore(tmp_path)
    store.save("vague", [gesture("vague", seed=s) for s in range(2)])
    sim = SimulatedAdapter(Config(gestures_dir=str(tmp_path), fps=15), realtime=False, seed=1)
    with sim:
        obs = [sim.read() for _ in range(80)]
    assert sim.feature_format == HAND_FORMAT and sim.current == "vague"
    assert all(o.features.shape == (FORMATS[HAND_FORMAT],) for o in obs)
    assert len(obs[0].live["hands"][0]) == 21
    json.dumps(obs[0].live)  # sérialisable pour l'interface web


def test_simulation_follows_radar_gestures(tmp_path):
    store = GestureStore(tmp_path)
    seq = np.linspace([0, 0, 0], [1, 0, 0.5], 12, dtype=np.float32)
    store.save("pas_droite", [seq, seq], feature_format=RADAR_FORMAT)
    sim = SimulatedAdapter(Config(gestures_dir=str(tmp_path)), realtime=False)
    with sim:
        o = sim.read()
    assert sim.feature_format == RADAR_FORMAT and o.features.shape == (3,)
    assert o.live["targets"][0]["focus"]
