"""Publication vers Home Assistant via MQTT, avec découverte automatique.

Entités créées automatiquement pour chaque capteur (`device_id`) :
  - event.<device>_geste           : un événement par geste reconnu (déclencheur d'automatisation)
  - sensor.<device>_dernier_geste  : nom du dernier geste
  - binary_sensor.<device>_main    : une main est visible (caméra)
    ou binary_sensor.<device>_presence : quelqu'un est présent (radar)
  - binary_sensor.<device>_attente : le geste d'attention a été fait, commande attendue
  - button.<device>_recharger      : relit le dossier des gestes sans redémarrer
  - sensor.<device>_<mesure>       : mesures propres au capteur (radar : nombre de
                                     personnes, distance)

Les entités dépendant du capteur sont décrites par l'adapter (`SensorAdapter`) : ce module
ne connaît aucun capteur en particulier.

Topics : <base>/<device>/{status,gesture,last_gesture,presence,armed,telemetry,cmd}
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import paho.mqtt.client as mqtt

from medvision import __version__
from medvision.config import Config
from medvision.engine import GestureEvent, State
from medvision.sensors.base import SensorAdapter

log = logging.getLogger(__name__)


class HomeAssistantMqtt:
    def __init__(
        self,
        cfg: Config,
        on_command: Callable[[str], None] | None = None,
        sensor: SensorAdapter | type[SensorAdapter] | None = None,
    ):
        self.cfg = cfg
        self.sensor = sensor
        self.presence_key = getattr(sensor, "presence_key", "main")
        self.presence_name = getattr(sensor, "presence_name", "Main détectée")
        self.telemetry_entities = tuple(getattr(sensor, "telemetry_entities", ()))
        self.on_command = on_command or (lambda c: None)
        self.uid = f"medvision_{cfg.device_id}"
        self.base = f"{cfg.mqtt_base_topic}/{cfg.device_id}"
        self.gesture_names: list[str] = []
        self.labels: dict[str, str] = {}

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=self.uid
        )
        if cfg.mqtt_username:
            self.client.username_pw_set(cfg.mqtt_username, cfg.mqtt_password or None)
        self.client.will_set(f"{self.base}/status", "offline", qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    # ---------------------------------------------------------------- cycle de vie
    def start(self, gesture_names: list[str], labels: dict[str, str]) -> None:
        self.gesture_names, self.labels = gesture_names, labels
        log.info("Connexion MQTT à %s:%s", self.cfg.mqtt_host, self.cfg.mqtt_port)
        self.client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.publish(f"{self.base}/status", "offline", qos=1, retain=True)
        self.client.loop_stop()
        self.client.disconnect()

    def update_gestures(self, gesture_names: list[str], labels: dict[str, str]) -> None:
        self.gesture_names, self.labels = gesture_names, labels
        if self.client.is_connected():
            self._publish_discovery()

    # ---------------------------------------------------------------- publications
    def publish_gesture(self, event: GestureEvent) -> None:
        payload = {
            "event_type": event.name,
            "label": self.labels.get(event.name, event.name),
            "confidence": round(event.confidence, 2),
        }
        self.client.publish(f"{self.base}/gesture", json.dumps(payload, ensure_ascii=False), qos=1)
        self.client.publish(f"{self.base}/last_gesture", event.name, qos=1, retain=True)

    def publish_state(self, state: State) -> None:
        self.client.publish(
            f"{self.base}/armed", "ON" if state == State.ARMED else "OFF", qos=1, retain=True
        )

    def publish_presence(self, present: bool) -> None:
        self.client.publish(f"{self.base}/presence", "ON" if present else "OFF", qos=0, retain=True)

    def publish_telemetry(self, values: dict) -> None:
        if self.telemetry_entities:
            self.client.publish(
                f"{self.base}/telemetry", json.dumps(values, ensure_ascii=False), qos=0
            )

    # ---------------------------------------------------------------- interne
    def _device(self) -> dict:
        return {
            "identifiers": [self.uid],
            "name": self.cfg.device_name,
            "manufacturer": "MedVision (open source)",
            "model": f"Gesture Engine ({getattr(self.sensor, 'kind', '') or 'camera'})",
            "sw_version": __version__,
        }

    def _common(self, domain: str, key: str, name: str) -> dict:
        return {
            "name": name,
            "unique_id": f"{self.uid}_{key}",
            # Identifiant d'entité suggéré (HA récents) ; sinon HA le déduit des noms.
            "default_entity_id": f"{domain}.{self.uid}_{key}",
            "availability_topic": f"{self.base}/status",
            "device": self._device(),
        }

    def _publish_discovery(self) -> None:
        p = self.cfg.mqtt_discovery_prefix
        configs = {
            f"{p}/event/{self.uid}/gesture/config": {
                **self._common("event", "geste", "Geste"),
                "state_topic": f"{self.base}/gesture",
                "event_types": self.gesture_names or ["aucun"],
                "icon": "mdi:hand-wave",
            },
            f"{p}/sensor/{self.uid}/last_gesture/config": {
                **self._common("sensor", "dernier_geste", "Dernier geste"),
                "state_topic": f"{self.base}/last_gesture",
                "icon": "mdi:gesture",
            },
            # « hand » : topic de découverte de la v0.1, gardé pour ne pas dupliquer l'entité.
            f"{p}/binary_sensor/{self.uid}/"
            f"{'hand' if self.presence_key == 'main' else self.presence_key}/config": {
                **self._common("binary_sensor", self.presence_key, self.presence_name),
                "state_topic": f"{self.base}/presence",
                "device_class": "occupancy",
            },
            f"{p}/binary_sensor/{self.uid}/armed/config": {
                **self._common("binary_sensor", "attente", "En attente d'une commande"),
                "state_topic": f"{self.base}/armed",
                "icon": "mdi:hand-back-right",
            },
            f"{p}/button/{self.uid}/reload/config": {
                **self._common("button", "recharger", "Recharger les gestes"),
                "command_topic": f"{self.base}/cmd",
                "payload_press": "reload",
                "entity_category": "config",
            },
        }
        for ent in self.telemetry_entities:
            cfg = {
                **self._common("sensor", ent.key, ent.name),
                "state_topic": f"{self.base}/telemetry",
                # Une valeur absente est publiée « None » : Home Assistant affiche « inconnu ».
                "value_template": f"{{{{ value_json.{ent.key} }}}}",
                "state_class": "measurement",
            }
            for k in ("unit", "device_class", "icon"):
                if getattr(ent, k):
                    cfg["unit_of_measurement" if k == "unit" else k] = getattr(ent, k)
            configs[f"{p}/sensor/{self.uid}/{ent.key}/config"] = cfg
        for topic, payload in configs.items():
            self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=True)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("Connexion MQTT refusée : %s", reason_code)
            return
        log.info("MQTT connecté.")
        self._publish_discovery()
        client.publish(f"{self.base}/status", "online", qos=1, retain=True)
        client.publish(f"{self.base}/armed", "OFF", qos=1, retain=True)
        client.subscribe(f"{self.base}/cmd", qos=1)
        # Quand Home Assistant redémarre, il demande de republier la découverte.
        client.subscribe(f"{self.cfg.mqtt_discovery_prefix}/status", qos=1)

    def _on_message(self, client, userdata, msg):
        text = msg.payload.decode("utf-8", "replace").strip()
        if msg.topic == f"{self.cfg.mqtt_discovery_prefix}/status" and text == "online":
            self._publish_discovery()
            client.publish(f"{self.base}/status", "online", qos=1, retain=True)
        elif msg.topic == f"{self.base}/cmd":
            self.on_command(text)
