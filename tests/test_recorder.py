import numpy as np
import pytest

from medvision.recorder import Phase, Recorder, RecordingRequest
from medvision.sensors.base import Observation

DT = 1 / 15


def _obs(f):
    return Observation(features=f, present=f is not None)


def _run(rec, frames, t0=0.0):
    t = t0
    for f in frames:
        t += DT
        rec.push(_obs(f), t)
    return t


def test_request_validation():
    with pytest.raises(ValueError):
        RecordingRequest(name="../x")
    with pytest.raises(ValueError):
        RecordingRequest(name="ok", samples=0)
    with pytest.raises(ValueError):
        RecordingRequest(name="ok", duration=20)
    assert RecordingRequest(name=" Lumiere ").name == "lumiere"


def test_records_all_samples_whatever_the_sensor_rate(gesture):
    req = RecordingRequest(name="vague", samples=3, duration=1.0, countdown=1.0)
    rec = Recorder(req)
    hand = gesture("vague", frames=200)
    _run(rec, list(hand))
    assert rec.phase == Phase.DONE and len(rec.captured) == 3
    # ~1 s à 15 obs/s : 15 images par exemple (le compte à rebours n'est pas capturé).
    assert all(14 <= len(s) <= 16 for s in rec.captured)
    assert rec.status()["progress"] == 1.0


def test_retries_when_nothing_is_detected(gesture):
    req = RecordingRequest(name="poing", samples=1, duration=1.0, countdown=0.5)
    rec = Recorder(req)
    t = _run(rec, [None] * 25)  # compte à rebours + capture sans main
    assert rec.phase == Phase.COUNTDOWN and rec.failures == 1
    assert "recommence" in rec.message
    _run(rec, list(gesture("poing", frames=40)), t)
    assert rec.phase == Phase.DONE


def test_gives_up_and_cancel():
    rec = Recorder(RecordingRequest(name="x", samples=1, duration=0.5, countdown=0), max_failures=2)
    _run(rec, [None] * 40)
    assert rec.phase == Phase.FAILED and rec.finished
    rec2 = Recorder(RecordingRequest(name="y"))
    rec2.cancel()
    assert rec2.phase == Phase.CANCELLED
    assert rec2.push(_obs(np.zeros(63)), 10) == Phase.CANCELLED


def test_status_during_countdown_and_capture(gesture):
    rec = Recorder(RecordingRequest(name="z", samples=2, duration=1.0, countdown=2.0))
    rec.push(_obs(None), 0.5)
    s = rec.status()
    assert s["phase"] == "countdown" and s["seconds_left"] == 1.5 and s["sample"] == 1
    assert "pret" in rec.status_line()
    rec.push(_obs(gesture("poing")[0]), 2.1)
    rec.push(_obs(gesture("poing")[1]), 2.6)
    s = rec.status()
    assert s["phase"] == "capture" and 0.4 < s["progress"] < 0.6
