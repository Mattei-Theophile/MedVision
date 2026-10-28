"""Capteurs pris en charge, derrière une interface commune (patron *Adapter*).

    from medvision.sensors import create_sensor
    with create_sensor(cfg) as sensor:          # cfg.sensor_type = "camera" | "ld2450" | …
        obs = sensor.read()

Les adapters intégrés sont chargés à la demande : un poste qui n'utilise que le radar
n'importe jamais OpenCV ni MediaPipe.
"""

from __future__ import annotations

import importlib

from medvision.config import Config
from medvision.sensors.base import Observation, SensorAdapter, TelemetryEntity

__all__ = [
    "Observation",
    "SensorAdapter",
    "TelemetryEntity",
    "available_sensors",
    "create_sensor",
    "register_sensor",
    "sensor_class",
]

# type -> "module:Classe" (chargement paresseux) ou classe déjà importée.
_REGISTRY: dict[str, str | type[SensorAdapter]] = {
    "camera": "medvision.sensors.camera:CameraAdapter",
    "ld2450": "medvision.sensors.ld2450:LD2450Adapter",
    "simulation": "medvision.sensors.simulation:SimulatedAdapter",
}

# Descriptions statiques, pour lister les capteurs sans importer leurs dépendances.
DESCRIPTIONS = {
    "camera": "Webcam USB ou caméra IP (RTSP) : points de la main via MediaPipe",
    "ld2450": "Radar mmWave HLK-LD2450 (UART ou passerelle série-TCP) : présence et "
    "gestes amples, fonctionne dans le noir, sans image",
    "simulation": "Capteur simulé qui rejoue les gestes enregistrés (tests, démonstration)",
}


def register_sensor(kind: str, description: str = ""):
    """Décorateur pour déclarer un adapter supplémentaire (extension, plugin…)."""

    def deco(cls: type[SensorAdapter]) -> type[SensorAdapter]:
        if not issubclass(cls, SensorAdapter):
            raise TypeError(f"{cls.__name__} doit hériter de SensorAdapter")
        cls.kind = kind
        _REGISTRY[kind] = cls
        DESCRIPTIONS[kind] = description or cls.description or kind
        return cls

    return deco


def available_sensors() -> dict[str, str]:
    return {k: DESCRIPTIONS.get(k, k) for k in sorted(_REGISTRY)}


def sensor_class(kind: str) -> type[SensorAdapter]:
    kind = kind.strip().lower()
    try:
        target = _REGISTRY[kind]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Capteur inconnu {kind!r} (disponibles : {known})") from None
    if isinstance(target, str):
        module, _, name = target.partition(":")
        target = getattr(importlib.import_module(module), name)
        _REGISTRY[kind] = target
    return target


def create_sensor(cfg: Config) -> SensorAdapter:
    """Fabrique : instancie l'adapter correspondant à `cfg.sensor_type` (non ouvert)."""
    return sensor_class(cfg.sensor_type)(cfg)
