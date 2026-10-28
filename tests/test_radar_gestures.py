"""Chaîne complète radar : trames LD2450 ─► adapter ─► moteur DTW (sans matériel)."""

import numpy as np

from medvision.classifier import GestureLibrary
from medvision.config import Config
from medvision.engine import GestureEngine
from medvision.sensors.ld2450 import LD2450Adapter, encode_frame


def trajectory(kind, x0, y0, rng, frames=15):
    """Positions (mm) et vitesse radiale (cm/s) d'une personne faisant un geste ample."""
    out = []
    for t in range(frames):
        p = t / (frames - 1)
        x, y = x0, y0
        if kind == "pas_droite":
            x += 450 * min(1.0, 2 * p)
        elif kind == "approche":
            y -= 600 * p
        elif kind == "balayage":
            x += 250 * np.sin(2 * np.pi * p)
        speed = -60 if kind == "approche" else 0
        out.append((int(x + rng.normal(0, 15)), int(y + rng.normal(0, 15)), speed))
    return out


class Stream:
    def __init__(self, data):
        self.data = data

    def read(self, n):
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk


def observe(points, cfg):
    blob = b"".join(encode_frame([p]) for p in points)
    sensor = LD2450Adapter(cfg, stream=Stream(blob))
    feats = []
    while (o := sensor.read()) is not None:
        feats.append(o.features)
    return feats


def test_radar_gestures_are_recognised_anywhere_in_the_room():
    cfg = Config(sensor_type="ld2450")
    rng = np.random.default_rng(0)
    lib = GestureLibrary()
    for kind in ("pas_droite", "approche", "balayage", "repos"):
        samples = []
        for _ in range(4):  # appris à 1,5 m, face au radar
            pts = [(0, 1500, 0)] * 10 + trajectory(kind, 0, 1500, rng)
            samples.append(np.stack(observe(pts, cfg)[10:]))
        lib.add(kind, samples)

    for kind in ("pas_droite", "approche", "balayage"):
        events = []
        eng = GestureEngine(lib, Config(window_frames=15), on_gesture=events.append)
        # Rejoué ailleurs dans la pièce (décalé de 1 m sur le côté, à 2,5 m).
        pts = [(-1000, 2500, 0)] * 12 + trajectory(kind, -1000, 2500, rng)
        pts += [pts[-1][:2] + (0,)] * 15
        for f in observe(pts, cfg):
            eng.push(f)
        assert [e.name for e in events] == [kind], (kind, events)
