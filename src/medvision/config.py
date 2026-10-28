"""Configuration du moteur.

La configuration est volontairement « plate » (pas de sections imbriquées) pour être
identique au fichier d'options d'un add-on Home Assistant (/data/options.json). Un
fichier JSON étant du YAML valide, le même chargeur lit les deux formats.

Ordre de priorité : valeurs par défaut < fichier < variables d'environnement MEDVISION_*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- Caméra ---------------------------------------------------------------
    # Index de webcam ("0") ou URL RTSP/HTTP ("rtsp://user:pass@192.168.50.10/stream2").
    camera_source: str = "0"
    camera_width: int = 640
    camera_height: int = 480
    # 15 images/s suffisent pour des gestes et divisent par deux la charge CPU.
    fps: int = 15

    # --- Économie de calcul (« motion gating ») -------------------------------
    # MediaPipe ne tourne que si du mouvement est détecté ou si une main a été vue récemment.
    motion_gating: bool = True
    motion_threshold: float = 0.01  # part des pixels qui changent (0-1)
    active_timeout: float = 4.0  # secondes de suivi après le dernier mouvement/main vus

    # --- Détection des mains --------------------------------------------------
    model_path: str = "models/hand_landmarker.task"
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    # Symétrise la main gauche pour qu'un geste appris d'une main marche avec l'autre.
    ambidextrous: bool = True

    # --- Reconnaissance -------------------------------------------------------
    gestures_dir: str = "gestures"
    window_frames: int = 24  # fenêtre glissante (~1,6 s à 15 fps, un peu plus qu'un exemple)
    min_hand_ratio: float = 0.7  # part minimale d'images avec une main dans la fenêtre
    default_threshold: float = 0.35  # distance DTW max si aucun seuil n'a été calibré
    evaluate_every: int = 1  # classifier toutes les N images (2 sur une machine très lente)
    # Nb de classifications identiques consécutives requises (2+ pour des poses statiques
    # très sensibles ; 1 convient aux gestes en mouvement).
    stable_hits: int = 1
    cooldown: float = 1.5  # secondes d'ignorance après un geste reconnu
    # Gestes « neutres » (séparés par des virgules) : reconnus mais jamais émis. Ils servent
    # à absorber les mouvements ordinaires et réduisent fortement les faux déclenchements.
    rest_gestures: str = "repos"

    # --- Sécurité : geste d'attention -----------------------------------------
    # Si renseigné, il faut d'abord faire ce geste, puis la commande dans `armed_timeout`.
    # Laisser vide pour déclencher directement (déconseillé hors tests).
    attention_gesture: str = ""
    armed_timeout: float = 5.0

    # --- MQTT / Home Assistant ------------------------------------------------
    mqtt_enabled: bool = True
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_discovery_prefix: str = "homeassistant"
    mqtt_base_topic: str = "medvision"
    device_id: str = "salon"
    device_name: str = "MedVision Salon"

    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
        data: dict[str, Any] = {}
        if path:
            p = Path(path)
            if p.exists():
                loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(f"{p} doit contenir un dictionnaire clé/valeur")
                data.update(loaded)
            else:
                raise FileNotFoundError(f"Fichier de configuration introuvable : {p}")

        env = os.environ if env is None else env
        for f in fields(cls):
            key = f"MEDVISION_{f.name.upper()}"
            if key in env:
                data[f.name] = env[key]

        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Options inconnues : {', '.join(sorted(unknown))}")

        kwargs = {
            name: _coerce(value, type(getattr(cls, name)))
            for name, value in data.items()
            if value is not None  # option facultative laissée vide : valeur par défaut
        }
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps doit être > 0")
        if self.window_frames < 5:
            raise ValueError("window_frames doit être >= 5")
        if not 0 < self.min_hand_ratio <= 1:
            raise ValueError("min_hand_ratio doit être dans ]0, 1]")
        if self.num_hands not in (1, 2):
            raise ValueError("num_hands doit valoir 1 ou 2")

    @property
    def rest_set(self) -> set[str]:
        return {n.strip() for n in self.rest_gestures.split(",") if n.strip()}

    @property
    def camera_index_or_url(self) -> int | str:
        src = str(self.camera_source).strip()
        return int(src) if src.isdigit() else src


def _coerce(value: Any, target: type) -> Any:
    if target is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "oui"}
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    if target is str:
        return "" if value is None else str(value)
    return value
