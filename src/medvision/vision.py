"""Acquisition vidéo et extraction des points clés de la main (MediaPipe Hand Landmarker).

Les images restent en mémoire le temps d'être analysées puis sont jetées : seuls les
points clés sortent de ce module.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from medvision.config import Config
from medvision.features import normalize_hand

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Connexions entre points pour l'aperçu à l'écran (os de la main).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]  # fmt: skip


def download_model(path: str | Path, url: str = MODEL_URL) -> Path:
    path = Path(path)
    if path.exists():
        log.info("Modèle déjà présent : %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Téléchargement du modèle MediaPipe (~8 Mo) vers %s", path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (URL fixe)
    tmp.replace(path)
    return path


@dataclass
class HandObservation:
    features: np.ndarray  # vecteur normalisé (63,)
    raw: np.ndarray  # points bruts (21, 3), coordonnées image normalisées, pour l'aperçu
    handedness: str


class Camera:
    """Webcam ou flux RTSP, avec limitation du nombre d'images/s et reconnexion."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.source = cfg.camera_index_or_url
        self.cap: cv2.VideoCapture | None = None
        self._period = 1.0 / cfg.fps
        self._last = 0.0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        # Tampon minimal : on veut l'image la plus récente, pas un retard accumulé.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {self.source!r}")
        log.info("Caméra ouverte : %s", self.source)

    def read(self) -> np.ndarray | None:
        """Retourne une image BGR au rythme `fps`, ou None en cas d'échec de lecture."""
        if self.cap is None:
            self.open()
        while (wait := self._period - (time.monotonic() - self._last)) > 0:
            if isinstance(self.source, int):
                time.sleep(wait)
                break
            # Flux réseau : on jette les images en trop plutôt que de les laisser s'accumuler.
            if not self.cap.grab():
                break
        ok, frame = self.cap.read()
        self._last = time.monotonic()
        if not ok:
            log.warning("Lecture caméra échouée, reconnexion dans 2 s…")
            self.close()
            time.sleep(2)
            return None
        h, w = frame.shape[:2]
        if w > self.cfg.camera_width:  # réduction : moins de calcul, même précision utile
            scale = self.cfg.camera_width / w
            frame = cv2.resize(frame, (self.cfg.camera_width, int(h * scale)))
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class MotionGate:
    """Détecteur de mouvement très léger (différence d'images en basse résolution).

    Évite de faire tourner le réseau de neurones quand la pièce est vide : c'est ce qui
    permettra plus tard d'utiliser une puce caméra bon marché.
    """

    def __init__(self, threshold: float, active_timeout: float):
        self.threshold = threshold
        self.active_timeout = active_timeout
        self._prev: np.ndarray | None = None
        self._active_until = 0.0

    def keep_alive(self) -> None:
        self._active_until = time.monotonic() + self.active_timeout

    def should_process(self, frame: np.ndarray) -> bool:
        small = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        moving = False
        if self._prev is not None:
            diff = cv2.absdiff(small, self._prev)
            moving = float((diff > 18).mean()) > self.threshold
        self._prev = small
        if moving:
            self.keep_alive()
        return time.monotonic() < self._active_until


class HandTracker:
    def __init__(self, cfg: Config):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = Path(cfg.model_path)
        if not model.exists():
            download_model(model)
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._ambi = cfg.ambidextrous
        self._last_ts = 0

    def detect(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))  # strictement croissant
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness, strict=False):
            raw = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
            label = handed[0].category_name if handed else "Right"
            hands.append(HandObservation(normalize_hand(raw, label, self._ambi), raw, label))
        return hands

    def close(self) -> None:
        self._landmarker.close()


def draw_preview(frame: np.ndarray, hands: list[HandObservation], status: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.raw]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (0, 200, 255), 2)
        for p in pts:
            cv2.circle(out, p, 3, (255, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(out, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out
