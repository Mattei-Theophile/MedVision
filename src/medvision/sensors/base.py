"""Interface commune des capteurs (patron de conception *Adapter*).

Le moteur de reconnaissance (`GestureEngine`), l'enregistreur, MQTT et l'interface web ne
connaissent qu'une seule interface : `SensorAdapter`. Chaque capteur réel (webcam via
MediaPipe, radar mmWave HLK-LD2450, …) a sa propre API ; un *adapter* la traduit en une suite
d'`Observation` homogènes :

    capteur réel ──► Adapter ──► Observation(features, present, live, telemetry)
                                        │
                     ┌──────────────────┼───────────────┐
                     ▼                  ▼               ▼
               GestureEngine        Recorder       MQTT / web

Ajouter un capteur = écrire une classe qui hérite de `SensorAdapter` et la déclarer avec
`@register_sensor("nom")` : rien d'autre ne change dans le reste du code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from medvision.config import Config


@dataclass
class Observation:
    """Une image « logique » du capteur, déjà réduite à ce dont le reste du code a besoin."""

    # Vecteur de features (dimension = FORMATS[adapter.feature_format]) ou None si rien
    # d'exploitable n'est vu (pas de main, pas de cible…).
    features: np.ndarray | None
    # Quelqu'un (ou une main) est détecté. Peut être vrai même si `features` est None.
    present: bool
    timestamp: float = field(default_factory=time.monotonic)
    # Données légères pour la vue en direct de l'interface web (JSON) : jamais d'image.
    live: dict[str, Any] = field(default_factory=dict)
    # Mesures publiées vers Home Assistant (voir `SensorAdapter.telemetry_entities`).
    telemetry: dict[str, Any] = field(default_factory=dict)
    # Donnée brute propre à l'adapter (image caméra pour l'aperçu local…). Ne quitte jamais
    # le processus.
    raw: Any = None


@dataclass(frozen=True)
class TelemetryEntity:
    """Mesure exposée comme `sensor` dans Home Assistant."""

    key: str
    name: str
    unit: str | None = None
    device_class: str | None = None
    icon: str | None = None


class SensorAdapter(ABC):
    """Contrat que doit respecter tout capteur."""

    #: identifiant utilisé dans la configuration (`sensor_type`)
    kind: ClassVar[str] = ""
    #: description courte, affichée par `medvision sensors`
    description: ClassVar[str] = ""
    #: format des vecteurs produits (clé de `medvision.features.FORMATS`)
    feature_format: ClassVar[str] = ""
    #: entité de présence dans Home Assistant (clé, nom affiché)
    presence_key: ClassVar[str] = "presence"
    presence_name: ClassVar[str] = "Présence"
    #: mesures supplémentaires publiées vers Home Assistant
    telemetry_entities: ClassVar[tuple[TelemetryEntity, ...]] = ()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.frames = 0  # observations produites
        self.analysed = 0  # observations réellement analysées (hors veille)

    # ----------------------------------------------------------------- à implémenter
    @abstractmethod
    def open(self) -> None:
        """Ouvre le matériel. Peut lever RuntimeError avec un message lisible."""

    @abstractmethod
    def read(self) -> Observation | None:
        """Bloque jusqu'à la prochaine observation (au rythme du capteur).

        Retourne None en cas d'échec passager (reconnexion en cours, trame illisible…).
        """

    @abstractmethod
    def close(self) -> None:
        """Libère le matériel. Doit pouvoir être appelé plusieurs fois."""

    # ----------------------------------------------------------------- optionnel
    @property
    def rate(self) -> float:
        """Observations par seconde attendues (sert au dimensionnement de la fenêtre)."""
        return float(self.cfg.fps)

    def render(self, obs: Observation, status: str) -> np.ndarray | None:
        """Image BGR d'aperçu local (`--preview`), ou None si le capteur n'en propose pas."""
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "feature_format": self.feature_format,
            "presence_name": self.presence_name,
            "rate": self.rate,
        }

    # ----------------------------------------------------------------- gestion de contexte
    def __enter__(self) -> SensorAdapter:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
