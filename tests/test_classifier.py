import numpy as np

from medvision.classifier import GestureLibrary, dtw_distance


def test_dtw_identity_and_symmetry(gesture):
    a, b = gesture("vague", seed=1), gesture("poing", seed=2)
    assert dtw_distance(a, a) == 0.0
    assert np.isclose(dtw_distance(a, b), dtw_distance(b, a))
    assert dtw_distance(a, b) > 0


def _library(gesture):
    lib = GestureLibrary()
    for kind in ("poing", "vague", "pouce"):
        lib.add(kind, [gesture(kind, seed=s) for s in range(5)])
    return lib


def test_recognises_each_gesture(gesture):
    lib = _library(gesture)
    for kind in ("poing", "vague", "pouce"):
        m = lib.classify(gesture(kind, seed=100))
        assert m.name == kind and m.accepted, m


def test_tolerates_speed_and_length_changes(gesture):
    lib = _library(gesture)
    m = lib.classify(gesture("poing", frames=32, seed=7, speed=1.3))
    assert m.name == "poing" and m.accepted


def test_rejects_unknown_movement(gesture):
    lib = _library(gesture)
    m = lib.classify(gesture("repos", seed=3))
    assert not m.accepted, m


def test_empty_library_and_quality(gesture):
    assert GestureLibrary().classify(gesture("poing")) is None
    lib = _library(gesture)
    assert lib.quality("poing") == "bon"
    lib.add("seul", [gesture("vague")])
    assert lib.quality("seul").startswith("insuffisant")
