"""Stockage des gestes enregistrés : un fichier JSON par geste, lisible et partageable.

Format :
{
  "name": "lumiere",
  "label": "Allumer la lumière",          # texte libre affiché dans Home Assistant
  "format": "mediapipe-hand-21x3-normalized",   # ou "ld2450-target-v1" (radar)…
  "samples": [[[D floats], ...], ...]       # liste de séquences d'images
}
Le format dépend du capteur qui a servi à l'enregistrement : un geste fait devant la caméra
n'est reconnu que par une caméra, un geste radar que par un radar. Les fichiers de la v0.1
(format main) restent valides tels quels.
Seuls des points clés normalisés sont stockés : aucune image, aucun visage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from medvision.classifier import GestureLibrary
from medvision.features import FORMATS, HAND_FORMAT

FORMAT = HAND_FORMAT  # compatibilité v0.1
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

    def list(self, feature_format: str | None = None) -> list[str]:
        """Noms des gestes, éventuellement limités à un format de capteur."""
        if not self.dir.exists():
            return []
        names = sorted(p.stem for p in self.dir.glob("*.json"))
        if feature_format is None:
            return names
        return [n for n in names if self.format_of(n) == feature_format]

    def format_of(self, name: str) -> str | None:
        try:
            return self.load_one(name)["format"]
        except (OSError, ValueError, KeyError):
            return None

    def load_one(self, name: str) -> dict:
        data = json.loads(self._path(name).read_text(encoding="utf-8"))
        if data.get("format") not in FORMATS:
            raise ValueError(f"{name}: format inconnu {data.get('format')!r}")
        return data

    def save(
        self,
        name: str,
        samples: list[np.ndarray],
        label: str = "",
        append: bool = True,
        feature_format: str = HAND_FORMAT,
    ):
        name = validate_name(name)
        if feature_format not in FORMATS:
            raise ValueError(f"format inconnu {feature_format!r}")
        dim = FORMATS[feature_format]
        self.dir.mkdir(parents=True, exist_ok=True)
        existing: list = []
        path = self._path(name)
        if append and path.exists():
            data = self.load_one(name)
            if data["format"] != feature_format:
                raise ValueError(
                    f"Le geste « {name} » a été enregistré avec un autre capteur "
                    f"({data['format']}). Choisissez un autre nom ou remplacez-le."
                )
            existing = data["samples"]
            label = label or data.get("label", "")
        for s in samples:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != dim:
                raise ValueError(f"échantillon de forme {arr.shape}, attendu (T, {dim})")
        payload = {
            "name": name,
            "label": label or name,
            "format": feature_format,
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

    def labels(self, feature_format: str | None = None) -> dict[str, str]:
        return {n: self.load_one(n).get("label", n) for n in self.list(feature_format)}

    def build_library(
        self, default_threshold: float = 0.35, feature_format: str | None = None
    ) -> GestureLibrary:
        """Bibliothèque des gestes d'un format (par défaut : format main, comme en v0.1)."""
        fmt = feature_format or HAND_FORMAT
        lib = GestureLibrary(default_threshold=default_threshold)
        for name in self.list():
            data = self.load_one(name)
            if data["format"] != fmt:
                continue
            lib.add(name, [np.asarray(s, dtype=np.float32) for s in data["samples"]])
        return lib
