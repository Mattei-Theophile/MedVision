"""Enregistrement d'un geste, piloté par les observations du capteur (donc testable).

Le même enregistreur sert à la ligne de commande (`medvision record`) et à l'interface web.
Il ne dépend que du temps et des `Observation` : il marche avec n'importe quel capteur.

Déroulé pour chaque exemple :
    COUNTDOWN (compte à rebours) ─► CAPTURE (durée fixe) ─► exemple accepté ou recommencé
puis DONE quand tous les exemples sont acquis (ou FAILED après trop d'échecs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from medvision.sensors.base import Observation
from medvision.store import validate_name


class Phase(str, Enum):
    COUNTDOWN = "countdown"
    CAPTURE = "capture"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RecordingRequest:
    name: str
    label: str = ""
    samples: int = 5
    duration: float = 1.5
    countdown: float = 3.0
    replace: bool = False

    def __post_init__(self):
        self.name = validate_name(self.name)
        self.label = (self.label or "").strip()[:80]
        if not 1 <= int(self.samples) <= 20:
            raise ValueError("Nombre d'exemples : entre 1 et 20.")
        if not 0.5 <= float(self.duration) <= 6:
            raise ValueError("Durée d'un exemple : entre 0,5 et 6 secondes.")
        if not 0 <= float(self.countdown) <= 10:
            raise ValueError("Compte à rebours : entre 0 et 10 secondes.")
        self.samples, self.duration = int(self.samples), float(self.duration)
        self.countdown = float(self.countdown)


@dataclass
class Recorder:
    request: RecordingRequest
    min_ratio: float = 0.7
    start: float = 0.0
    max_failures: int | None = None

    phase: Phase = Phase.COUNTDOWN
    captured: list[np.ndarray] = field(default_factory=list)
    message: str = ""
    failures: int = 0
    _phase_start: float = 0.0
    _seq: list[np.ndarray] = field(default_factory=list)
    _seen: int = 0
    _now: float = 0.0

    def __post_init__(self):
        self._phase_start = self._now = self.start
        if self.max_failures is None:
            self.max_failures = 3 * self.request.samples
        self.message = f"Exemple 1/{self.request.samples} : préparez-vous…"

    # ------------------------------------------------------------------ API
    @property
    def finished(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED, Phase.CANCELLED)

    def cancel(self) -> None:
        if not self.finished:
            self.phase, self.message = Phase.CANCELLED, "Enregistrement annulé."

    def push(self, obs: Observation | None, now: float) -> Phase:
        self._now = now
        if self.finished:
            return self.phase
        elapsed = now - self._phase_start
        if self.phase == Phase.COUNTDOWN:
            if elapsed >= self.request.countdown:
                self._set(Phase.CAPTURE, now, f"{self._label()} : faites le geste !")
                self._seq, self._seen = [], 0
                self._capture(obs)  # cette observation compte déjà
            return self.phase
        # CAPTURE
        if elapsed < self.request.duration:
            self._capture(obs)
            return self.phase
        self._finish_sample(now)
        return self.phase

    def status(self) -> dict:
        req = self.request
        elapsed = self._now - self._phase_start
        if self.phase == Phase.COUNTDOWN:
            progress, left = 0.0, max(0.0, req.countdown - elapsed)
        elif self.phase == Phase.CAPTURE:
            progress, left = min(1.0, elapsed / req.duration), max(0.0, req.duration - elapsed)
        else:
            progress, left = 1.0, 0.0
        return {
            "name": req.name,
            "label": req.label,
            "phase": self.phase.value,
            "sample": min(len(self.captured) + 1, req.samples),
            "total": req.samples,
            "captured": len(self.captured),
            "progress": round(progress, 3),
            "seconds_left": round(left, 1),
            "message": self.message,
        }

    def status_line(self) -> str:
        s = self.status()
        if self.phase == Phase.COUNTDOWN:
            return f"Exemple {s['sample']}/{s['total']} - pret... {int(s['seconds_left']) + 1}"
        if self.phase == Phase.CAPTURE:
            return f"Exemple {s['sample']}/{s['total']} - enregistrement {s['progress']:.0%}"
        return self.message

    # ------------------------------------------------------------------ interne
    def _label(self) -> str:
        return f"Exemple {len(self.captured) + 1}/{self.request.samples}"

    def _capture(self, obs: Observation | None) -> None:
        self._seen += 1
        if obs is not None and obs.features is not None:
            self._seq.append(np.asarray(obs.features, dtype=np.float32))

    def _finish_sample(self, now: float) -> None:
        seen, ok = max(1, self._seen), len(self._seq)
        if ok < max(5, self.min_ratio * seen):
            self.failures += 1
            if self.failures >= self.max_failures:
                self._set(Phase.FAILED, now, "Trop d'essais sans détection : vérifiez le capteur.")
                return
            self._set(
                Phase.COUNTDOWN,
                now,
                f"Détecté sur {ok}/{seen} images seulement : on recommence l'exemple "
                f"{len(self.captured) + 1}.",
            )
            return
        self.captured.append(np.stack(self._seq))
        if len(self.captured) >= self.request.samples:
            self._set(Phase.DONE, now, f"{len(self.captured)} exemples enregistrés.")
        else:
            self._set(Phase.COUNTDOWN, now, f"{self._label()} : préparez-vous…")

    def _set(self, phase: Phase, now: float, message: str) -> None:
        self.phase, self._phase_start, self.message = phase, now, message
