"""Adapter caméra : webcam / RTSP + détection de mouvement + MediaPipe Hand Landmarker.

Les classes historiques de `medvision.vision` (`Camera`, `MotionGate`, `HandTracker`) sont
les *adaptees* : elles gardent leur API propre et cet adapter les assemble pour produire des
`Observation`. Les images restent dans ce module (aperçu local uniquement).
"""

from __future__ import annotations

from medvision.config import Config
from medvision.features import HAND_FORMAT
from medvision.sensors.base import Observation, SensorAdapter


class CameraAdapter(SensorAdapter):
    kind = "camera"
    description = "Webcam USB ou caméra IP (RTSP) : points de la main via MediaPipe"
    feature_format = HAND_FORMAT
    # Clé historique : l'entité binary_sensor.medvision_<id>_main de la v0.1 est conservée.
    presence_key = "main"
    presence_name = "Main détectée"

    def __init__(self, cfg: Config, camera=None, tracker=None, gate=None):
        super().__init__(cfg)
        # Injection possible pour les tests (objets factices sans matériel ni MediaPipe).
        self._camera, self._tracker, self._gate = camera, tracker, gate
        self._injected_gate = gate is not None

    def open(self) -> None:
        from medvision.vision import Camera, HandTracker, MotionGate

        if self._camera is None:
            self._camera = Camera(self.cfg)
        if hasattr(self._camera, "open"):
            self._camera.open()
        if self._tracker is None:
            self._tracker = HandTracker(self.cfg)
        if self._gate is None and not self._injected_gate and self.cfg.motion_gating:
            self._gate = MotionGate(self.cfg.motion_threshold, self.cfg.active_timeout)

    def read(self) -> Observation | None:
        if self._camera is None:
            self.open()
        frame = self._camera.read()
        if frame is None:
            return None
        self.frames += 1
        hands = []
        if self._gate is None or self._gate.should_process(frame):
            self.analysed += 1
            hands = self._tracker.detect(frame)
            if hands and self._gate is not None:
                self._gate.keep_alive()
        live = {
            "hands": [
                [[round(float(x), 3), round(float(y), 3)] for x, y, _ in h.raw] for h in hands
            ]
        }
        return Observation(
            features=hands[0].features if hands else None,
            present=bool(hands),
            live=live,
            raw=(frame, hands),
        )

    def render(self, obs: Observation, status: str):
        from medvision.vision import draw_preview

        frame, hands = obs.raw
        return draw_preview(frame, hands, status)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
