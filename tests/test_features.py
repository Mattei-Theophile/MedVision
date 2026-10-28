import numpy as np
import pytest

from medvision.features import FEATURE_DIM, normalize_hand, resample


def _hand():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, (21, 3)).astype(np.float32)


def test_translation_and_scale_invariance():
    pts = _hand()
    moved = pts.copy()
    moved[:, :2] = moved[:, :2] * 2.0 + 0.3  # main plus proche et ailleurs dans l'image
    moved[:, 2] *= 2.0
    np.testing.assert_allclose(normalize_hand(pts), normalize_hand(moved), atol=1e-5)


def test_wrist_is_origin_and_shape():
    v = normalize_hand(_hand())
    assert v.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(v[:3], 0.0, atol=1e-7)


def test_ambidextrous_mirrors_left_hand():
    pts = _hand()
    right = normalize_hand(pts, "Right")
    left = normalize_hand(pts, "Left")
    np.testing.assert_allclose(left.reshape(21, 3)[:, 0], -right.reshape(21, 3)[:, 0])
    np.testing.assert_allclose(normalize_hand(pts, "Left", ambidextrous=False), right)


def test_resample_lengths():
    seq = np.arange(10, dtype=np.float32).reshape(5, 2)
    out = resample(seq, 9)
    assert out.shape == (9, 2)
    np.testing.assert_allclose(out[0], seq[0])
    np.testing.assert_allclose(out[-1], seq[-1])
    assert resample(seq[:1], 4).shape == (4, 2)
    with pytest.raises(ValueError):
        resample(np.zeros((0, 2)), 4)
