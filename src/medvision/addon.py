"""Point d'entrée de l'add-on Home Assistant.

- lit /data/options.json (options saisies dans l'interface de l'add-on) ;
- si aucun serveur MQTT n'est renseigné, récupère automatiquement celui de Home Assistant
  (add-on Mosquitto) auprès du Supervisor ;
- range les gestes dans /share/medvision/gestures (accessible via Samba / File editor)
  et le modèle dans /data.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from medvision.cli import _setup_logging, build_parser, cmd_run
from medvision.config import Config

OPTIONS = Path("/data/options.json")
log = logging.getLogger("medvision.addon")


def supervisor_mqtt() -> dict | None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return None
    req = urllib.request.Request(
        "http://supervisor/services/mqtt", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 (URL interne fixe)
            return json.load(resp).get("data")
    except OSError as exc:
        log.warning("Service MQTT du Supervisor indisponible : %s", exc)
        return None


def main() -> None:
    cfg = Config.load(OPTIONS if OPTIONS.exists() else None)
    _setup_logging(cfg.log_level)
    cfg.gestures_dir = "/share/medvision/gestures"
    cfg.model_path = "/data/hand_landmarker.task"
    Path(cfg.gestures_dir).mkdir(parents=True, exist_ok=True)

    if not cfg.mqtt_host or cfg.mqtt_host == "auto":
        svc = supervisor_mqtt()
        if svc:
            cfg.mqtt_host, cfg.mqtt_port = svc["host"], int(svc["port"])
            cfg.mqtt_username, cfg.mqtt_password = svc.get("username", ""), svc.get("password", "")
            log.info("MQTT de Home Assistant détecté : %s:%s", cfg.mqtt_host, cfg.mqtt_port)
        else:
            cfg.mqtt_host = "core-mosquitto"

    args = build_parser().parse_args(["run"])
    raise SystemExit(cmd_run(cfg, args))


if __name__ == "__main__":
    main()
