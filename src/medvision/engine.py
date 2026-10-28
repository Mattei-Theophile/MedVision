"""Machine à états de reconnaissance, indépendante de la caméra (donc testable).

Le moteur reçoit une image « logique » à la fois : soit un vecteur de main normalisé,
soit None (aucune main). Il garde une fenêtre glissante, classe périodiquement, et émet :
  - on_gesture(GestureEvent)  : une commande validée ;
  - on_state(State)          : IDLE / ARMED (pour un retour visuel : lampe qui clignote...) ;
  - on_presence(bool)        : une main est visible ou non.

Sécurité contre les faux positifs :
  - stabilité : le même geste doit sortir `stable_hits` fois de suite ;
  - cooldown  : pause après chaque reconnaissance ;
  - attention : si `attention_gesture` est défini, une commande n'est acceptée que dans
                les `armed_timeout` secondes qui suivent ce geste.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from medvision.classifier import GestureLibrary, Match
from medvision.config import Config

log = logging.getLogger(__name__)

SUFFIX_RATIOS = (0.5, 0.75, 1.0)


class State(str, Enum):
    IDLE = "idle"
    ARMED = "armed"


@dataclass
class GestureEvent:
    name: str
    distance: float
    threshold: float
    timestamp: float

    @property
    def confidence(self) -> float:
        """Score indicatif 0-1 (1 = identique aux exemples)."""
        return float(max(0.0, 1.0 - self.distance / (2 * self.threshold)))


class GestureEngine:
    def __init__(
        self,
        library: GestureLibrary,
        config: Config,
        on_gesture: Callable[[GestureEvent], None] | None = None,
        on_state: Callable[[State], None] | None = None,
        on_presence: Callable[[bool], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.lib = library
        self.cfg = config
        self.on_gesture = on_gesture or (lambda e: None)
        self.on_state = on_state or (lambda s: None)
        self.on_presence = on_presence or (lambda p: None)
        self.clock = clock

        self.window: deque[np.ndarray | None] = deque(maxlen=config.window_frames)
        self.state = State.IDLE
        self._armed_at = 0.0
        self._cooldown_until = 0.0
        self._frame_count = 0
        self._last_candidate: str | None = None
        self._hits = 0
        self._present = False

        att = config.attention_gesture.strip()
        if att and att not in library.names:
            log.warning("Geste d'attention %r absent de la bibliothèque : mode direct.", att)
            att = ""
        self.attention = att
        self.rest = config.rest_set

    # ------------------------------------------------------------------ API
    def push(self, features: np.ndarray | None) -> GestureEvent | None:
        now = self.clock()
        self.window.append(features)
        self._frame_count += 1
        self._update_presence(features is not None)
        self._check_timeout(now)

        if now < self._cooldown_until:
            return None
        if self._frame_count % max(1, self.cfg.evaluate_every):
            return None

        match = self._classify_window()
        if match is None or not match.accepted or match.name in self.rest:
            self._last_candidate, self._hits = None, 0
            return None

        if match.name == self._last_candidate:
            self._hits += 1
        else:
            self._last_candidate, self._hits = match.name, 1
        if self._hits < self.cfg.stable_hits:
            return None

        return self._handle(match, now)

    def reset(self) -> None:
        self.window.clear()
        self._last_candidate, self._hits = None, 0
        self._set_state(State.IDLE)

    # ------------------------------------------------------------ internes
    def _classify_window(self) -> Match | None:
        if len(self.window) < self.window.maxlen:
            return None
        # Le geste réel n'est jamais aligné sur la fenêtre : on teste plusieurs « fins de
        # fenêtre » (la moitié, les trois quarts, la totalité) et on garde la meilleure.
        window = list(self.window)
        best: Match | None = None
        for ratio in SUFFIX_RATIOS:
            segment = window[len(window) - max(5, round(ratio * len(window))) :]
            frames = [f for f in segment if f is not None]
            if len(frames) < max(5, self.cfg.min_hand_ratio * len(segment)):
                continue
            m = self.lib.classify(np.stack(frames))
            if m and (best is None or m.distance / m.threshold < best.distance / best.threshold):
                best = m
        return best

    def _handle(self, match: Match, now: float) -> GestureEvent | None:
        self._cooldown_until = now + self.cfg.cooldown
        self.window.clear()
        self._last_candidate, self._hits = None, 0

        if self.attention:
            if match.name == self.attention:
                self._armed_at = now
                self._set_state(State.ARMED)
                log.info("Geste d'attention reconnu : en attente d'une commande.")
                return None
            if self.state != State.ARMED:
                log.debug("Commande %s ignorée (pas de geste d'attention).", match.name)
                return None
            self._set_state(State.IDLE)

        event = GestureEvent(match.name, match.distance, match.threshold, time.time())
        log.info(
            "Geste reconnu : %s (distance %.3f / seuil %.3f)",
            event.name,
            event.distance,
            event.threshold,
        )
        self.on_gesture(event)
        return event

    def _check_timeout(self, now: float) -> None:
        if self.state == State.ARMED and now - self._armed_at > self.cfg.armed_timeout:
            log.info("Délai dépassé : retour en veille.")
            self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        if state != self.state:
            self.state = state
            self.on_state(state)

    def _update_presence(self, present: bool) -> None:
        # Hystérésis simple : présence = au moins une main dans les 5 dernières images.
        recent = list(self.window)[-5:]
        now_present = any(f is not None for f in recent) if recent else present
        if now_present != self._present:
            self._present = now_present
            self.on_presence(now_present)
