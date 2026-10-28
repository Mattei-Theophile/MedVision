"""Générateurs de mains synthétiques : les tests n'ont besoin ni de caméra ni de MediaPipe."""

from __future__ import annotations

import numpy as np
import pytest

from medvision.features import FEATURE_DIM, normalize_hand


def _base_hand(rng: np.random.Generator) -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float32)
    # Poignet en (0.5, 0.8), doigts répartis vers le haut.
    pts[0] = (0.5, 0.8, 0.0)
    for finger in range(5):
        for joint in range(4):
            idx = 1 + finger * 4 + joint
            pts[idx] = (
                0.4 + finger * 0.05,
                0.7 - joint * 0.05,
                0.0,
            )
    return pts + rng.normal(0, 0.002, pts.shape).astype(np.float32)


def make_gesture(kind: str, frames: int = 20, seed: int = 0, speed: float = 1.0) -> np.ndarray:
    """Séquence (frames, 63) pour quelques gestes types."""
    rng = np.random.default_rng(seed)
    seq = []
    for t in range(frames):
        phase = min(1.0, speed * t / max(1, frames - 1))
        pts = _base_hand(rng)
        fingertips = [i for i in range(5, 21) if (i - 1) % 4 != 0]  # hors bases des doigts
        if kind == "poing":  # doigts repliés vers la paume
            pts[fingertips, 1] += 0.12 * phase
        elif kind == "vague":  # oscillation latérale de la main entière
            pts[:, 0] += 0.08 * np.sin(phase * 2 * np.pi * 2)
            pts[1:, 0] += 0.05 * np.sin(phase * 2 * np.pi * 2)
        elif kind == "pouce":  # pouce qui s'écarte
            pts[1:5, 0] -= 0.15 * phase
        elif kind == "repos":
            pass
        else:
            raise ValueError(kind)
        seq.append(normalize_hand(pts))
    arr = np.stack(seq)
    assert arr.shape[1] == FEATURE_DIM
    return arr


@pytest.fixture
def gesture():
    return make_gesture
