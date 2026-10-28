"""Transformation des points clés MediaPipe en vecteurs comparables.

Une main = 21 points (x, y, z). On rend le vecteur indépendant de la position de la main
dans l'image et de sa distance à la caméra :
  1. translation : le poignet (point 0) devient l'origine ;
  2. échelle : division par la taille de la paume (distance moyenne poignet -> bases
     des doigts 5, 9, 13, 17), stable même quand la main se ferme ;
  3. option ambidextre : la main gauche est symétrisée (x -> -x).
Seules ces coordonnées sont manipulées : aucune image n'est jamais conservée.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
PALM_POINTS = (5, 9, 13, 17)  # bases de l'index, du majeur, de l'annulaire, de l'auriculaire
FEATURE_DIM = NUM_LANDMARKS * 3

# Chaque capteur produit des vecteurs d'un format donné. Un geste enregistré avec un capteur
# n'est comparable qu'à des observations du même format : le format est stocké avec le geste.
HAND_FORMAT = "mediapipe-hand-21x3-normalized"
RADAR_FORMAT = "ld2450-target-v1"
FORMATS: dict[str, int] = {HAND_FORMAT: FEATURE_DIM, RADAR_FORMAT: 3}
FORMAT_LABELS = {HAND_FORMAT: "main (caméra)", RADAR_FORMAT: "cible (radar LD2450)"}


def register_format(name: str, dim: int, label: str = "") -> None:
    """Déclare un nouveau format de vecteur (pour un adapter de capteur tiers)."""
    if FORMATS.get(name, dim) != dim:
        raise ValueError(f"format {name!r} déjà déclaré avec une autre dimension")
    FORMATS[name] = dim
    FORMAT_LABELS.setdefault(name, label or name)


def normalize_hand(
    landmarks: np.ndarray | list, handedness: str | None = None, ambidextrous: bool = True
) -> np.ndarray:
    """Retourne un vecteur (63,) normalisé à partir de 21 points (x, y, z)."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[WRIST]
    scale = float(np.linalg.norm(pts[list(PALM_POINTS), :2], axis=1).mean())
    if scale < 1e-6:
        scale = float(np.linalg.norm(pts[:, :2], axis=1).max()) or 1.0
    pts /= scale
    if ambidextrous and handedness and handedness.lower().startswith("l"):
        pts[:, 0] *= -1.0
    return pts.reshape(FEATURE_DIM)


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    """Ré-échantillonne une séquence (T, D) en (length, D) par interpolation linéaire."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("la séquence doit être de forme (T, D)")
    t = seq.shape[0]
    if t == 0:
        raise ValueError("séquence vide")
    if t == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(dst, src, seq[:, d]) for d in range(seq.shape[1])], axis=1).astype(
        np.float32
    )
