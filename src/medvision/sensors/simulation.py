"""Capteur simulé : rejoue en boucle les gestes enregistrés, entrecoupés de repos.

Sert à tester toute la chaîne (moteur, MQTT, automatisations Home Assistant, interface web)
sans caméra ni radar :

    MEDVISION_SENSOR_TYPE=simulation uv run medvision run --dry-run
"""

from __future__ import annotations

import logging
import time
from collections import Counter

import numpy as np

from medvision.config import Config
from medvision.features import FORMATS, HAND_FORMAT, RADAR_FORMAT, normalize_hand
from medvision.sensors.base import Observation, SensorAdapter

log = logging.getLogger(__name__)


def synthetic_hand() -> np.ndarray:
    """Main ouverte immobile (21 points, coordonnées image) : sert de « repos »."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = (0.5, 0.8, 0.0)
    for finger in range(5):
        for joint in range(4):
            pts[1 + finger * 4 + joint] = (0.4 + finger * 0.05, 0.7 - joint * 0.05, 0.0)
    return pts


class SimulatedAdapter(SensorAdapter):
    kind = "simulation"
    description = "Capteur simulé qui rejoue les gestes enregistrés"
    presence_key = "presence"
    presence_name = "Présence (simulée)"

    def __init__(self, cfg: Config, realtime: bool = True, seed: int | None = None):
        super().__init__(cfg)
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.samples: dict[str, list[np.ndarray]] = {}
        self.feature_format = HAND_FORMAT  # précisé à l'ouverture selon les gestes trouvés
        self._queue: list[np.ndarray] = []
        self._next_tick = 0.0
        self.current = "repos"

    def open(self) -> None:
        from medvision.store import GestureStore

        store = GestureStore(self.cfg.gestures_dir)
        wanted = {n.strip() for n in self.cfg.simulation_gestures.split(",") if n.strip()}
        by_format: dict[str, dict[str, list[np.ndarray]]] = {}
        for name in store.list():
            if wanted and name not in wanted:
                continue
            data = store.load_one(name)
            seqs = [np.asarray(s, dtype=np.float32) for s in data["samples"]]
            by_format.setdefault(data["format"], {})[name] = seqs
        if by_format:
            fmt = Counter({f: len(g) for f, g in by_format.items()}).most_common(1)[0][0]
            self.feature_format, self.samples = fmt, by_format[fmt]
        log.info(
            "Simulation (%s) : %s",
            self.feature_format,
            ", ".join(sorted(self.samples)) or "repos uniquement (aucun geste enregistré)",
        )
        self._next_tick = time.monotonic()

    def _rest(self, frames: int) -> list[np.ndarray]:
        if self.feature_format == HAND_FORMAT:
            base = synthetic_hand()
            return [
                normalize_hand(base + self.rng.normal(0, 0.002, base.shape).astype(np.float32))
                for _ in range(frames)
            ]
        dim = FORMATS[self.feature_format]
        return [self.rng.normal(0, 0.01, dim).astype(np.float32) for _ in range(frames)]

    def _refill(self) -> None:
        rest = self._rest(int(2 * self.rate))
        candidates = [n for n in self.samples if n not in self.cfg.rest_set]
        if not candidates:
            self._queue, self.current = rest, "repos"
            return
        name = candidates[int(self.rng.integers(len(candidates)))]
        seqs = self.samples[name]
        seq = seqs[int(self.rng.integers(len(seqs)))]
        self._queue = rest + [np.asarray(f) for f in seq]
        self.current = name
        log.debug("Simulation : prochain geste %s", name)

    def read(self) -> Observation | None:
        if self.realtime:
            self._next_tick += 1.0 / self.rate
            delay = self._next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_tick = time.monotonic()
        if not self._queue:
            self._refill()
        features = self._queue.pop(0)
        self.frames += 1
        self.analysed += 1
        return Observation(features=features, present=True, live=self._live(features))

    def _live(self, f: np.ndarray) -> dict:
        if self.feature_format == HAND_FORMAT:
            pts = f.reshape(21, 3)
            return {
                "hands": [
                    [
                        [round(0.5 + 0.1 * float(x), 3), round(0.75 + 0.1 * float(y), 3)]
                        for x, y, _ in pts
                    ]
                ]
            }
        if self.feature_format == RADAR_FORMAT:
            return {
                "max_distance": self.cfg.radar_max_distance,
                "targets": [
                    {
                        "x": round(0.5 * float(f[0]), 3),
                        "y": round(2.0 + 0.5 * float(f[1]), 3),
                        "speed": round(float(f[2]), 2),
                        "focus": True,
                        "in_zone": True,
                    }
                ],
            }
        return {}

    def close(self) -> None:
        self._queue = []
