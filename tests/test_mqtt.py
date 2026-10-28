import json

from medvision.config import Config
from medvision.mqtt import HomeAssistantMqtt
from medvision.sensors.camera import CameraAdapter
from medvision.sensors.ld2450 import LD2450Adapter


class RecordingClient:
    def __init__(self):
        self.published = {}

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published[topic] = payload


def _discovery(sensor_cls):
    ha = HomeAssistantMqtt(Config(device_id="salon"), sensor=sensor_cls)
    ha.client = RecordingClient()
    ha.gesture_names = ["lumiere"]
    ha._publish_discovery()
    return ha, {t: json.loads(p) for t, p in ha.client.published.items()}


def test_camera_keeps_v01_entities():
    _, cfgs = _discovery(CameraAdapter)
    presence = cfgs["homeassistant/binary_sensor/medvision_salon/hand/config"]
    assert presence["unique_id"] == "medvision_salon_main"
    assert presence["state_topic"] == "medvision/salon/presence"
    assert not any("/sensor/medvision_salon/targets" in t for t in cfgs)
    assert cfgs["homeassistant/event/medvision_salon/gesture/config"]["event_types"] == ["lumiere"]


def test_radar_adds_presence_and_telemetry():
    ha, cfgs = _discovery(LD2450Adapter)
    assert cfgs["homeassistant/binary_sensor/medvision_salon/presence/config"]["name"] == (
        "Présence (radar)"
    )
    dist = cfgs["homeassistant/sensor/medvision_salon/distance/config"]
    assert dist["unit_of_measurement"] == "m" and dist["device_class"] == "distance"
    assert dist["value_template"] == "{{ value_json.distance }}"
    assert "ld2450" in dist["device"]["model"]
    ha.publish_telemetry({"targets": 1, "distance": None})
    assert json.loads(ha.client.published["medvision/salon/telemetry"])["distance"] is None
