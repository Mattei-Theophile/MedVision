"""Reconnaissance de gestes par comparaison à des exemples (k plus proches voisins + DTW).

Pourquoi ce choix pour la v0.1 :
- 5 à 10 exemples suffisent par geste, sans entraînement ni GPU ;
- DTW (Dynamic Time Warping) tolère un geste fait plus vite ou plus lentement ;
- un geste statique (pose de la main) et un geste dynamique (mouvement) sont traités pareil.

Chaque geste reçoit un seuil calibré automatiquement à partir de la dispersion de ses
propres exemples : un geste mal enregistré (exemples très différents) aura un seuil large,
signalé par `quality()` pour inviter à le réenregistrer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medvision.features import resample

RESAMPLE_LEN = 16


def dtw_batch(query: np.ndarray, templates: np.ndarray, band: int | None = 4) -> np.ndarray:
    """Distances DTW normalisées entre une séquence (n, D) et K exemples (K, m, D).

    La programmation dynamique est vectorisée sur les K exemples : le coût en Python ne
    dépend que de n x m (16 x 16), pas du nombre de gestes enregistrés.
    """
    q = np.asarray(query, dtype=np.float32)
    t = np.asarray(templates, dtype=np.float32)
    cost = np.linalg.norm(q[None, :, None, :] - t[:, None, :, :], axis=3)  # (K, n, m)
    k, n, m = cost.shape
    acc = np.full((k, n + 1, m + 1), np.inf, dtype=np.float64)
    acc[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = (1, m) if band is None else (max(1, i - band), min(m, i + band))
        for j in range(lo, hi + 1):
            prev = np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
            acc[:, i, j] = cost[:, i - 1, j - 1] + prev
    return acc[:, n, m] / (n + m)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None = 4) -> float:
    """Distance DTW normalisée entre deux séquences (T, D)."""
    return float(dtw_batch(a, np.asarray(b)[None], band)[0])


@dataclass
class Match:
    name: str
    distance: float
    threshold: float

    @property
    def accepted(self) -> bool:
        return self.distance <= self.threshold


@dataclass
class GestureLibrary:
    default_threshold: float = 0.35
    margin: float = 1.6  # seuil = marge x distance typique entre exemples d'un même geste
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    _stack: np.ndarray | None = field(default=None, repr=False)
    _labels: list[str] = field(default_factory=list, repr=False)

    def add(self, name: str, sequences: list[np.ndarray]) -> None:
        prepared = [resample(np.asarray(s, dtype=np.float32), RESAMPLE_LEN) for s in sequences]
        self.templates.setdefault(name, []).extend(prepared)
        self._stack = None
        self._calibrate(name)

    def remove(self, name: str) -> None:
        self.templates.pop(name, None)
        self.thresholds.pop(name, None)
        self._stack = None

    @property
    def names(self) -> list[str]:
        return sorted(self.templates)

    def _intra_distances(self, name: str) -> list[float]:
        temps = self.templates[name]
        out = []
        if len(temps) < 2:
            return out
        stack = np.stack(temps)
        for i, t in enumerate(temps):
            d = dtw_batch(t, stack)
            d[i] = np.inf
            out.append(float(d.min()))
        return out

    def _calibrate(self, name: str) -> None:
        d = self._intra_distances(name)
        if len(d) < 2:
            self.thresholds[name] = self.default_threshold
            return
        typical = float(np.mean(d) + np.std(d))
        # Seuil borné : jamais plus strict que 40 % ni plus large que 250 % du défaut.
        lo, hi = 0.4 * self.default_threshold, 2.5 * self.default_threshold
        self.thresholds[name] = float(np.clip(self.margin * typical, lo, hi))

    def quality(self, name: str) -> str:
        """Indique si les exemples d'un geste sont cohérents entre eux."""
        d = self._intra_distances(name)
        if len(d) < 2:
            return "insuffisant (au moins 3 exemples conseillés)"
        if self.margin * float(np.mean(d) + np.std(d)) > 2.5 * self.default_threshold:
            return "incohérent (réenregistrer le geste plus régulièrement)"
        return "bon"

    def classify(self, sequence: np.ndarray) -> Match | None:
        """Geste le plus proche (accepté ou non), ou None si la bibliothèque est vide."""
        if not self.templates:
            return None
        query = resample(np.asarray(sequence, dtype=np.float32), RESAMPLE_LEN)
        if self._stack is None:
            self._labels = [n for n, temps in self.templates.items() for _ in temps]
            self._stack = np.stack([t for temps in self.templates.values() for t in temps])
        dists = dtw_batch(query, self._stack)
        per_name: dict[str, float] = {}
        for label, d in zip(self._labels, dists, strict=True):
            per_name[label] = min(per_name.get(label, np.inf), float(d))
        best: Match | None = None
        for name, dist in per_name.items():
            thr = self.thresholds.get(name, self.default_threshold)
            # On compare les distances relativement au seuil propre à chaque geste.
            if best is None or dist / thr < best.distance / best.threshold:
                best = Match(name=name, distance=dist, threshold=thr)
        return best
