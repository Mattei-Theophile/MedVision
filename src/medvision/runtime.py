"""Boucle principale : capteur ─► (enregistreur | moteur) ─► Home Assistant / interface web.

`Runtime` ne connaît que l'interface `SensorAdapter` : la même boucle fait tourner une
caméra, un radar LD2450 ou le capteur simulé. Le capteur n'est lu que par le fil principal ;
l'interface web (autres fils) passe par les méthodes publiques protégées par un verrou.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from medvision import __version__
from medvision.classifier import GestureLibrary
from medvision.config import Config
from medvision.engine import GestureEngine, GestureEvent, State
from medvision.features import FORMAT_LABELS
from medvision.recorder import Phase, Recorder, RecordingRequest
from medvision.sensors import SensorAdapter, create_sensor
from medvision.sensors.base import Observation
from medvision.store import GestureStore, validate_name

log = logging.getLogger(__name__)


class Runtime:
    def __init__(
        self,
        cfg: Config,
        sensor: SensorAdapter | None = None,
        store: GestureStore | None = None,
        publisher=None,
        clock: Callable[[], float] = time.monotonic,
        echo: bool = False,
    ):
        self.cfg = cfg
        self.sensor = sensor or create_sensor(cfg)
        self.store = store or GestureStore(cfg.gestures_dir)
        self.ha = publisher  # HomeAssistantMqtt ou None (mode --dry-run)
        self.clock = clock
        self.echo = echo
        self._lock = threading.RLock()
        self._reload = threading.Event()
        self._opened = False

        self.lib = GestureLibrary(default_threshold=cfg.default_threshold)
        self.labels: dict[str, str] = {}
        self.engine: GestureEngine | None = None
        self.recorder: Recorder | None = None
        self.status_text = "Veille"
        self.state = State.IDLE
        self.present = False
        self.live: dict[str, Any] = {}
        self.telemetry: dict[str, Any] = {}
        self.history: deque[dict] = deque(maxlen=15)
        self.started_at = time.time()
        self._telemetry_sent: tuple[float, dict] = (0.0, {})
        self._perf = {"since": clock(), "frames": 0, "analysed": 0, "fps": 0.0, "ratio": 0.0}

    # ================================================================ cycle de vie
    def open(self) -> None:
        self.sensor.open()
        self._opened = True
        self._load_library()
        log.info(
            "Capteur %s prêt (format %s). Gestes : %s",
            self.sensor.kind,
            self.sensor.feature_format,
            ", ".join(self.lib.names) or "aucun",
        )
        if self.ha:
            self.ha.start(self.command_names(), self.store.labels(self.sensor.feature_format))

    def close(self) -> None:
        if self._opened:
            self.sensor.close()
            self._opened = False
        if self.ha:
            self.ha.stop()

    def run(self, stop: threading.Event, preview=None) -> None:
        """Boucle jusqu'à `stop` (ou fermeture de l'aperçu)."""
        while not stop.is_set():
            obs = self.step()
            if preview is not None and obs is not None:
                image = self.sensor.render(obs, self.status_text)
                if image is not None and not preview.show(image):
                    break

    # ================================================================ une itération
    def step(self) -> Observation | None:
        if self._reload.is_set():
            self._reload.clear()
            self._load_library()
            if self.ha:
                self.ha.update_gestures(
                    self.command_names(), self.store.labels(self.sensor.feature_format)
                )
            log.info("Gestes rechargés : %s", ", ".join(self.lib.names) or "aucun")

        obs = self.sensor.read()
        if obs is None:
            return None
        now = self.clock()
        with self._lock:
            self.live, self.telemetry = obs.live, obs.telemetry
            rec = self.recorder if self.recorder and not self.recorder.finished else None
        if rec is not None:
            # Sous verrou : l'interface web ne voit jamais un état intermédiaire.
            with self._lock:
                rec.push(obs, now)
                self.status_text = rec.status_line()
                if rec.finished:
                    self._finish_recording(rec)
        else:
            self.engine.push(obs.features)
        self._publish_telemetry(obs.telemetry, now)
        self._update_perf(now)
        return obs

    # ================================================================ API (interface web)
    def command_names(self) -> list[str]:
        return [n for n in self.lib.names if n not in self.cfg.rest_set]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rec = self.recorder.status() if self.recorder else None
            return {
                "version": __version__,
                "device": {"id": self.cfg.device_id, "name": self.cfg.device_name},
                "sensor": {
                    **self.sensor.describe(),
                    "format_label": FORMAT_LABELS.get(self.sensor.feature_format, ""),
                },
                "state": self.state.value,
                "status": self.status_text,
                "present": self.present,
                "attention_gesture": self.engine.attention if self.engine else "",
                "history": list(self.history),
                "recording": rec,
                "live": self.live,
                "telemetry": self.telemetry,
                "performance": {k: self._perf[k] for k in ("fps", "ratio")},
                "mqtt": bool(self.ha),
            }

    def gestures(self) -> list[dict[str, Any]]:
        fmt = self.sensor.feature_format
        with self._lock:
            lib = self.lib
        out = []
        for name in self.store.list():
            try:
                data = self.store.load_one(name)
            except (OSError, ValueError) as exc:
                out.append({"name": name, "error": str(exc)})
                continue
            compatible = data["format"] == fmt and name in lib.templates
            out.append(
                {
                    "name": name,
                    "label": data.get("label", name),
                    "format": data["format"],
                    "format_label": FORMAT_LABELS.get(data["format"], data["format"]),
                    "samples": len(data["samples"]),
                    "compatible": compatible,
                    "threshold": round(lib.thresholds[name], 3) if compatible else None,
                    "quality": lib.quality(name) if compatible else None,
                    "rest": name in self.cfg.rest_set,
                    "attention": name == self.cfg.attention_gesture,
                }
            )
        return out

    def start_recording(self, **kwargs) -> dict:
        req = RecordingRequest(**kwargs)
        existing = self.store.format_of(req.name) if req.name in self.store.list() else None
        if existing and existing != self.sensor.feature_format and not req.replace:
            raise ValueError(
                f"« {req.name} » existe déjà pour un autre capteur ({FORMAT_LABELS.get(existing)})."
                " Choisissez un autre nom ou cochez « remplacer »."
            )
        with self._lock:
            if self.recorder and not self.recorder.finished:
                raise ValueError("Un enregistrement est déjà en cours.")
            self.recorder = Recorder(req, min_ratio=self.cfg.min_hand_ratio, start=self.clock())
            self.status_text = self.recorder.status_line()
        log.info("Enregistrement du geste %s (%d exemples)", req.name, req.samples)
        return self.recorder.status()

    def cancel_recording(self) -> None:
        with self._lock:
            if self.recorder:
                self.recorder.cancel()
                self.status_text = "Veille"

    def delete_gesture(self, name: str) -> bool:
        ok = self.store.delete(validate_name(name))
        if ok:
            log.info("Geste supprimé : %s", name)
            self.request_reload()
        return ok

    def request_reload(self) -> None:
        self._reload.set()

    # ================================================================ interne
    def _load_library(self) -> None:
        lib = self.store.build_library(self.cfg.default_threshold, self.sensor.feature_format)
        engine = GestureEngine(
            lib,
            self.cfg,
            on_gesture=self._on_gesture,
            on_state=self._on_state,
            on_presence=self._on_presence,
            clock=self.clock,
        )
        labels = self.store.labels(self.sensor.feature_format)
        with self._lock:
            self.lib, self.engine, self.labels = lib, engine, labels
            self.state = State.IDLE

    def _finish_recording(self, rec: Recorder) -> None:
        if rec.phase == Phase.DONE:
            req = rec.request
            try:
                self.store.save(
                    req.name,
                    rec.captured,
                    label=req.label,
                    append=not req.replace,
                    feature_format=self.sensor.feature_format,
                )
            except (OSError, ValueError) as exc:
                rec.phase, rec.message = Phase.FAILED, f"Échec de l'enregistrement : {exc}"
            else:
                self._load_library()
                # La personne finit souvent son geste : on évite un déclenchement immédiat.
                self.engine.pause(max(2.0, self.cfg.cooldown))
                if self.ha:
                    self.ha.update_gestures(
                        self.command_names(), self.store.labels(self.sensor.feature_format)
                    )
                quality = self.lib.quality(req.name) if req.name in self.lib.templates else "?"
                rec.message = (
                    f"« {req.name} » enregistré ({len(self.lib.templates.get(req.name, []))} "
                    f"exemples au total, qualité : {quality})."
                )
        log.info("Enregistrement terminé : %s", rec.message)
        self.status_text = "Veille"

    def _on_gesture(self, ev: GestureEvent) -> None:
        entry = {
            "name": ev.name,
            "label": self.labels.get(ev.name, ev.name),
            "confidence": round(ev.confidence, 2),
            "at": ev.timestamp,
        }
        with self._lock:
            self.history.appendleft(entry)
            self.status_text = f"Geste : {ev.name} ({ev.confidence:.0%})"
        if self.ha:
            self.ha.publish_gesture(ev)
        if self.echo:
            print(f">>> GESTE : {ev.name}  (confiance {ev.confidence:.0%})", flush=True)

    def _on_state(self, st: State) -> None:
        with self._lock:
            self.state = st
            self.status_text = "En attente d'une commande..." if st == State.ARMED else "Veille"
        if self.ha:
            self.ha.publish_state(st)

    def _on_presence(self, present: bool) -> None:
        with self._lock:
            self.present = present
        if self.ha:
            self.ha.publish_presence(present)

    def _publish_telemetry(self, values: dict, now: float) -> None:
        if not self.ha or not values:
            return
        last_t, last_v = self._telemetry_sent
        # Au plus une fois par seconde si ça change, et toutes les 30 s sinon.
        if (values != last_v and now - last_t >= 1.0) or now - last_t >= 30.0:
            self.ha.publish_telemetry(values)
            self._telemetry_sent = (now, dict(values))

    def _update_perf(self, now: float) -> None:
        p = self._perf
        dt = now - p["since"]
        if dt < 60:
            return
        frames = self.sensor.frames - p["frames"]
        analysed = self.sensor.analysed - p["analysed"]
        p.update(
            fps=round(frames / dt, 1),
            ratio=round(analysed / max(1, frames), 2),
            since=now,
            frames=self.sensor.frames,
            analysed=self.sensor.analysed,
        )
        log.info(
            "Performance : %.1f obs/s, %d %% analysées (%s)",
            p["fps"],
            100 * p["ratio"],
            self.sensor.kind,
        )
