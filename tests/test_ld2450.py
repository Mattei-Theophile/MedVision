import numpy as np
import pytest

from medvision.config import Config
from medvision.sensors.ld2450 import (
    FRAME_LEN,
    LD2450Adapter,
    LD2450Parser,
    RadarTarget,
    TargetFeaturizer,
    decode_frame,
    encode_frame,
)

# Trame d'exemple de la documentation HLK : une cible à x = -782 mm, y = 1713 mm,
# vitesse -16 cm/s, résolution 320 mm.
DOC_FRAME = bytes.fromhex("AAFF0300" + "0E03B1861000400100000000000000000000000000000000" + "55CC")


def test_decode_documented_frame():
    (t,) = decode_frame(DOC_FRAME)
    assert (t.x_mm, t.y_mm, t.speed_cms, t.resolution_mm) == (-782, 1713, -16, 320)
    assert t.distance_mm == pytest.approx(1883, abs=1)


def test_encode_decode_roundtrip_and_signs():
    targets = [(-1200, 2500, 35), (800, 900, -60), (0, 400, 0)]
    frame = encode_frame(targets)
    assert len(frame) == FRAME_LEN
    decoded = decode_frame(frame)
    assert [(t.x_mm, t.y_mm, t.speed_cms) for t in decoded] == targets


def test_empty_slots_are_skipped():
    assert decode_frame(encode_frame([])) == []
    assert len(decode_frame(encode_frame([(100, 1000, 0)]))) == 1
    with pytest.raises(ValueError):
        decode_frame(DOC_FRAME[:-1] + b"\x00")


def test_parser_resyncs_and_handles_split_frames():
    parser = LD2450Parser()
    f1, f2 = encode_frame([(100, 1000, 0)]), encode_frame([(-300, 2000, 10)])
    ack = bytes.fromhex("FDFCFBFA0400FF010000" + "04030201")  # acquittement de commande
    stream = b"\x01\x02garbage" + ack + f1 + b"\xaa\xff" + f2  # faux début d'en-tête
    frames = []
    for i in range(0, len(stream), 7):  # lecture par petits morceaux
        frames += parser.feed(stream[i : i + 7])
    assert [fr[0].x_mm for fr in frames] == [100, -300]


def test_featurizer_is_relative_to_where_the_person_stands():
    feat = TargetFeaturizer(max_distance_mm=4000, anchor_time=3.0, rate=10)

    # Même pas de côté (+40 cm), fait à deux endroits différents de la pièce.
    def step_aside(x0, y0):
        f = TargetFeaturizer(max_distance_mm=4000, anchor_time=3.0, rate=10)
        out = [f.update([RadarTarget(x0, y0, 0, 300)]) for _ in range(5)]
        out += [f.update([RadarTarget(x0 + 400, y0, 0, 300)]) for _ in range(3)]
        return np.stack(out)

    np.testing.assert_allclose(step_aside(-500, 1500), step_aside(900, 2800), atol=1e-6)
    assert feat.update([RadarTarget(0, 5000, 0, 300)]) is None  # hors zone


def test_featurizer_follows_the_same_target():
    f = TargetFeaturizer(max_distance_mm=4000, anchor_time=3.0, rate=10)
    a, b = RadarTarget(0, 1000, 0, 300), RadarTarget(1500, 3000, 0, 300)
    f.update([a])
    f.update([RadarTarget(1400, 2900, 0, 300), RadarTarget(50, 1050, 0, 300)])
    assert f.focus.x_mm == 50  # la plus proche de la cible précédente, pas la première
    f.update([b])
    assert f.focus == b


class FakeSerial:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def read(self, n):
        return self.chunks.pop(0) if self.chunks else b""


def test_adapter_produces_observations():
    frames = [encode_frame([(0, 1500, 0)]) for _ in range(3)] + [encode_frame([(0, 6000, 0)])]
    blob = b"".join(frames)
    sensor = LD2450Adapter(Config(sensor_type="ld2450"), stream=FakeSerial([blob[:40], blob[40:]]))
    with sensor:
        obs = [sensor.read() for _ in range(4)]
        assert sensor.read() is None  # plus de données
    assert all(o.present for o in obs[:3]) and not obs[3].present
    assert obs[0].features.shape == (3,) and obs[3].features is None
    assert obs[0].telemetry == {"targets": 1, "distance": 1.5}
    assert obs[3].telemetry == {"targets": 0, "distance": None}
    assert obs[0].live["targets"][0]["focus"] is True
    assert sensor.render(obs[0], "test").shape == (360, 480, 3)
