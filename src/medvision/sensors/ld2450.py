"""Adapter radar mmWave HLK-LD2450 (24 GHz, suivi de 3 cibles en X/Y).

Pourquoi un radar à côté de la caméra :
  - aucune image n'existe, même en mémoire : c'est le capteur le plus respectueux de la vie
    privée (chambre, salle de bain) ;
  - il fonctionne dans le noir complet et à travers un léger obstacle (abat-jour, tissu) ;
  - il détecte une personne immobile (assise, allongée), ce que la caméra fait mal.
Limite : sa résolution (~15 cm) ne permet pas de voir les doigts. On reconnaît des **gestes
amples** du corps ou du bras (s'approcher, reculer, pas de côté, balayage), avec la même
méthode DTW que pour la caméra.

Protocole série (256 000 bauds, 8N1), une trame de 30 octets ~10 fois par seconde :

    AA FF 03 00 | cible 1 (8 o) | cible 2 (8 o) | cible 3 (8 o) | 55 CC

Chaque cible : x, y (mm), vitesse (cm/s) sur 16 bits little-endian où le bit de poids fort
vaut 1 pour une valeur positive (valeur = brut - 0x8000) et 0 pour une valeur négative
(valeur = -brut), puis la résolution de distance (uint16, mm). Une cible absente vaut 0.

Le port peut être local (`/dev/ttyUSB0`) ou une passerelle série-TCP (`socket://ip:port`,
par ex. un ESP32 avec ESPHome *stream server* collé au radar).
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from medvision.config import Config
from medvision.features import RADAR_FORMAT
from medvision.sensors.base import Observation, SensorAdapter, TelemetryEntity

log = logging.getLogger(__name__)

HEADER = b"\xaa\xff\x03\x00"
FOOTER = b"\x55\xcc"
FRAME_LEN = 30
TARGET_LEN = 8
MAX_TARGETS = 3
DEFAULT_BAUDRATE = 256000
NOMINAL_RATE = 10.0  # trames/s annoncées par le fabricant

# Échelles de normalisation des features (ordre de grandeur d'un geste ample).
POSITION_SCALE_MM = 500.0
SPEED_SCALE_CMS = 100.0


# --------------------------------------------------------------------------- protocole
@dataclass(frozen=True)
class RadarTarget:
    x_mm: int  # latéral : négatif à gauche du radar, positif à droite
    y_mm: int  # distance devant le radar (toujours >= 0)
    speed_cms: int  # radiale : positive = s'éloigne
    resolution_mm: int

    @property
    def distance_mm(self) -> float:
        return math.hypot(self.x_mm, self.y_mm)


def _signed(lo: int, hi: int) -> int:
    raw = lo | (hi << 8)
    return raw - 0x8000 if raw & 0x8000 else -raw


def decode_frame(frame: bytes) -> list[RadarTarget]:
    """Décode une trame complète de 30 octets ; lève ValueError si elle est invalide."""
    if len(frame) != FRAME_LEN or not frame.startswith(HEADER) or not frame.endswith(FOOTER):
        raise ValueError("trame LD2450 invalide")
    targets = []
    for i in range(MAX_TARGETS):
        b = frame[4 + i * TARGET_LEN : 4 + (i + 1) * TARGET_LEN]
        if not any(b):
            continue  # emplacement vide
        targets.append(
            RadarTarget(
                x_mm=_signed(b[0], b[1]),
                y_mm=_signed(b[2], b[3]),
                speed_cms=_signed(b[4], b[5]),
                resolution_mm=b[6] | (b[7] << 8),
            )
        )
    return targets


def encode_frame(targets: list[tuple[int, int, int]] | list[RadarTarget]) -> bytes:
    """Construit une trame (utile pour les tests et la simulation)."""

    def enc(v: int) -> bytes:
        raw = (v + 0x8000) if v >= 0 else (-v)
        return int(raw & 0xFFFF).to_bytes(2, "little")

    body = b""
    for t in list(targets)[:MAX_TARGETS]:
        if isinstance(t, RadarTarget):
            x, y, s, r = t.x_mm, t.y_mm, t.speed_cms, t.resolution_mm
        else:
            (x, y, s), r = t, 360
        body += enc(x) + enc(y) + enc(s) + int(r).to_bytes(2, "little")
    body += b"\x00" * (TARGET_LEN * MAX_TARGETS - len(body))
    return HEADER + body + FOOTER


class LD2450Parser:
    """Découpe un flux d'octets en trames, en se resynchronisant sur l'en-tête.

    Les trames d'acquittement de commandes (FD FC FB FA …) et les octets parasites sont
    ignorés.
    """

    def __init__(self, max_buffer: int = 4096):
        self._buf = bytearray()
        self._max = max_buffer
        self.errors = 0

    def feed(self, data: bytes) -> list[list[RadarTarget]]:
        self._buf.extend(data)
        frames: list[list[RadarTarget]] = []
        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                # Garder la fin : l'en-tête peut être à cheval sur deux lectures.
                del self._buf[: max(0, len(self._buf) - (len(HEADER) - 1))]
                break
            if start:
                del self._buf[:start]
            if len(self._buf) < FRAME_LEN:
                break
            chunk = bytes(self._buf[:FRAME_LEN])
            try:
                frames.append(decode_frame(chunk))
                del self._buf[:FRAME_LEN]
            except ValueError:
                self.errors += 1
                del self._buf[: len(HEADER)]  # faux en-tête : on cherche le suivant
        if len(self._buf) > self._max:
            del self._buf[: -self._max]
        return frames


# --------------------------------------------------------------------------- features
class TargetFeaturizer:
    """Transforme la cible suivie en vecteur (dx, dy, vitesse) invariant à la position.

    Le geste est mesuré **par rapport à l'endroit où se tient la personne** : une ancre suit
    lentement sa position (moyenne glissante exponentielle de `anchor_time` secondes). Un pas
    de côté ou un balayage du bras écarte brièvement la cible de l'ancre ; c'est cet écart,
    et non la position absolue dans la pièce, qui est comparé aux exemples.
    """

    def __init__(self, max_distance_mm: float, anchor_time: float, rate: float = NOMINAL_RATE):
        self.max_distance_mm = max_distance_mm
        self.alpha = min(1.0, 1.0 / max(1e-3, anchor_time * rate))
        self.focus: RadarTarget | None = None
        self._anchor: np.ndarray | None = None
        self._lost = 0
        self._lost_limit = max(1, int(rate))  # ~1 s sans cible : on oublie l'ancre

    def in_zone(self, targets: list[RadarTarget]) -> list[RadarTarget]:
        return [t for t in targets if 0 < t.distance_mm <= self.max_distance_mm]

    def _pick(self, targets: list[RadarTarget]) -> RadarTarget | None:
        if not targets:
            return None
        if self.focus is not None:  # continuité : la cible la plus proche de la précédente
            fx, fy = self.focus.x_mm, self.focus.y_mm
            return min(targets, key=lambda t: math.hypot(t.x_mm - fx, t.y_mm - fy))
        return min(targets, key=lambda t: t.distance_mm)  # sinon la plus proche du radar

    def update(self, targets: list[RadarTarget]) -> np.ndarray | None:
        target = self._pick(self.in_zone(targets))
        if target is None:
            self._lost += 1
            if self._lost >= self._lost_limit:
                self.focus, self._anchor = None, None
            return None
        self._lost = 0
        self.focus = target
        pos = np.array([target.x_mm, target.y_mm], dtype=np.float64)
        if self._anchor is None:
            self._anchor = pos.copy()
        delta = (pos - self._anchor) / POSITION_SCALE_MM
        self._anchor += self.alpha * (pos - self._anchor)
        return np.array([delta[0], delta[1], target.speed_cms / SPEED_SCALE_CMS], dtype=np.float32)


# --------------------------------------------------------------------------- adapter
class LD2450Adapter(SensorAdapter):
    kind = "ld2450"
    description = "Radar mmWave HLK-LD2450 : présence et gestes amples, sans image"
    feature_format = RADAR_FORMAT
    presence_key = "presence"
    presence_name = "Présence (radar)"
    telemetry_entities = (
        TelemetryEntity("targets", "Personnes détectées", icon="mdi:account-multiple"),
        TelemetryEntity("distance", "Distance", unit="m", device_class="distance"),
    )

    def __init__(self, cfg: Config, stream=None, clock=time.monotonic):
        super().__init__(cfg)
        self._stream = stream  # injectable : tout objet avec read(n) -> bytes
        self._injected = stream is not None
        self._clock = clock
        self.parser = LD2450Parser()
        self.featurizer = TargetFeaturizer(cfg.radar_max_distance * 1000.0, cfg.radar_anchor_time)
        self._pending: deque[list[RadarTarget]] = deque(maxlen=5)  # 0,5 s de retard max
        self._last_data = clock()

    @property
    def rate(self) -> float:
        return NOMINAL_RATE

    def open(self) -> None:
        if self._stream is not None:
            return
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover - dépendance déclarée
            raise RuntimeError("pyserial est requis pour le radar : lancez `uv sync`") from exc
        try:
            self._stream = serial.serial_for_url(
                self.cfg.radar_port, baudrate=self.cfg.radar_baudrate, timeout=0.2
            )
        except (serial.SerialException, ValueError, OSError) as exc:
            msg = f"Impossible d'ouvrir le radar {self.cfg.radar_port!r} : {exc}"
            raise RuntimeError(msg) from exc
        self._last_data = self._clock()
        log.info("Radar LD2450 ouvert : %s", self.cfg.radar_port)

    def read(self) -> Observation | None:
        if self._stream is None:
            self.open()
        while not self._pending:
            data = self._read_bytes()
            if not data:  # rien reçu pendant le délai de lecture : on rend la main
                return None
            self._pending.extend(self.parser.feed(data))
        targets = self._pending.popleft()
        self.frames += 1
        self.analysed += 1
        features = self.featurizer.update(targets)
        zone = self.featurizer.in_zone(targets)
        focus = self.featurizer.focus if features is not None else None
        live = {
            "max_distance": self.cfg.radar_max_distance,
            "targets": [
                {
                    "x": round(t.x_mm / 1000, 3),
                    "y": round(t.y_mm / 1000, 3),
                    "speed": round(t.speed_cms / 100, 2),
                    "focus": t is focus,
                    "in_zone": t in zone,
                }
                for t in targets
            ],
        }
        telemetry = {
            "targets": len(zone),
            "distance": round(focus.distance_mm / 1000, 2) if focus else None,
        }
        return Observation(
            features=features, present=bool(zone), live=live, telemetry=telemetry, raw=targets
        )

    def _read_bytes(self) -> bytes | None:
        try:
            data = self._stream.read(256)
        except Exception as exc:  # noqa: BLE001 - pyserial lève des types variés
            log.warning("Lecture radar échouée (%s), reconnexion dans 2 s…", exc)
            self._reconnect()
            return None
        if data:
            self._last_data = self._clock()
            return data
        if self._clock() - self._last_data > 3.0:
            log.warning("Aucune donnée du radar depuis 3 s (câblage, baudrate ?)")
            self._last_data = self._clock()
            if not self._injected:
                self._reconnect()
            return None
        return b""

    def _reconnect(self) -> None:
        if self._injected:
            return
        self.close()
        time.sleep(2)
        try:
            self.open()
        except RuntimeError as exc:
            log.warning("%s", exc)

    def render(self, obs: Observation, status: str):
        import cv2

        w, h, scale = 480, 360, 320 / max(1.0, self.cfg.radar_max_distance)  # px par mètre
        img = np.full((h, w, 3), 24, dtype=np.uint8)
        ox, oy = w // 2, h - 10
        for r in range(1, int(self.cfg.radar_max_distance) + 1):
            cv2.ellipse(img, (ox, oy), (int(r * scale),) * 2, 0, 210, 330, (70, 70, 70), 1)
        for t in obs.live.get("targets", []):
            p = (int(ox + t["x"] * scale), int(oy - t["y"] * scale))
            color = (0, 200, 255) if t["focus"] else (160, 160, 160)
            cv2.circle(img, p, 9 if t["focus"] else 6, color, -1)
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def close(self) -> None:
        if self._stream is not None and not self._injected:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
