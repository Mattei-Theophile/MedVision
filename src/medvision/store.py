"""Stockage des gestes enregistrés : un fichier JSON par geste, lisible et partageable.

Format :
{
  "name": "lumiere",
  "label": "Allumer la lumière",          # texte libre affiché dans Home Assistant
  "format": "mediapipe-hand-21x3-normalized",
  "samples": [[[63 floats], ...], ...]      # liste de séquences d'images
}
Seuls des points clés normalisés sont stockés : aucune image, aucun visage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from medvision.classifier import GestureLibrary
from medvision.features import FEATURE_DIM

FORMAT = "mediapipe-hand-21x3-normalized"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


def validate_name(name: str) -> str:
    name = name.strip().lower()
    if not _NAME_RE.match(name):
        raise ValueError(
            "Nom de geste invalide : minuscules, chiffres, '-' ou '_' uniquement (40 car. max)."
        )
    return name


class GestureStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _path(self, name: str) -> Path:
        return self.dir / f"{validate_name(name)}.json"

    def list(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def load_one(self, name: str) -> dict:
        data = json.loads(self._path(name).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{name}: format inconnu {data.get('format')!r}")
        return data

    def save(self, name: str, samples: list[np.ndarray], label: str = "", append: bool = True):
        name = validate_name(name)
        self.dir.mkdir(parents=True, exist_ok=True)
        existing: list = []
        path = self._path(name)
        if append and path.exists():
            data = self.load_one(name)
            existing = data["samples"]
            label = label or data.get("label", "")
        for s in samples:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
                raise ValueError(f"échantillon de forme {arr.shape}, attendu (T, {FEATURE_DIM})")
        payload = {
            "name": name,
            "label": label or name,
            "format": FORMAT,
            "samples": existing + [np.round(np.asarray(s), 4).tolist() for s in samples],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def labels(self) -> dict[str, str]:
        return {n: self.load_one(n).get("label", n) for n in self.list()}

    def build_library(self, default_threshold: float = 0.35) -> GestureLibrary:
        lib = GestureLibrary(default_threshold=default_threshold)
        for name in self.list():
            data = self.load_one(name)
            lib.add(name, [np.asarray(s, dtype=np.float32) for s in data["samples"]])
        return lib
