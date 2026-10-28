import json

import numpy as np
import pytest

from medvision.config import Config
from medvision.store import GestureStore


def test_store_roundtrip(tmp_path, gesture):
    store = GestureStore(tmp_path / "g")
    store.save("lumiere", [gesture("poing", seed=s) for s in range(3)], label="Lumière salon")
    store.save("lumiere", [gesture("poing", seed=9)])  # ajout d'un exemple
    assert store.list() == ["lumiere"]
    data = store.load_one("lumiere")
    assert len(data["samples"]) == 4 and data["label"] == "Lumière salon"
    lib = store.build_library()
    assert lib.classify(gesture("poing", seed=42)).name == "lumiere"
    assert store.delete("lumiere") and store.list() == []


def test_store_rejects_bad_names_and_shapes(tmp_path):
    store = GestureStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("../evil", [np.zeros((5, 63))])
    with pytest.raises(ValueError):
        store.save("ok", [np.zeros((5, 10))])


def test_config_file_env_and_addon_json(tmp_path):
    p = tmp_path / "options.json"  # format d'un add-on Home Assistant
    p.write_text(json.dumps({"camera_source": "rtsp://cam/stream", "fps": 10}))
    cfg = Config.load(p, env={"MEDVISION_MQTT_ENABLED": "false", "MEDVISION_FPS": "12"})
    assert cfg.camera_index_or_url == "rtsp://cam/stream"
    assert cfg.fps == 12 and cfg.mqtt_enabled is False
    assert Config().camera_index_or_url == 0


def test_config_rejects_unknown_and_invalid(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("typo_option: 1\n")
    with pytest.raises(ValueError):
        Config.load(p, env={})
    p.write_text("num_hands: 3\n")
    with pytest.raises(ValueError):
        Config.load(p, env={})
