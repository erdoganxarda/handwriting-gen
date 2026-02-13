from __future__ import annotations

import numpy as np

from src.datasets.emnist_to_strokes import convert_images_to_stroke_cache, image_to_stroke_sequence


def _toy_letter_image() -> np.ndarray:
    img = np.ones((28, 28), dtype=np.float32)
    img[4:24, 14] = 0.0
    img[4, 10:19] = 0.0
    img[14, 10:19] = 0.0
    return img


def test_image_to_stroke_sequence_non_empty_and_bounded() -> None:
    image = _toy_letter_image()
    seq, mask = image_to_stroke_sequence(image, max_len=160)

    assert seq.shape == (160, 3)
    assert mask.shape == (160,)
    assert mask.sum() > 0

    valid = mask.astype(bool)
    assert np.all(seq[valid, :2] <= 1.0)
    assert np.all(seq[valid, :2] >= -1.0)
    assert np.all((seq[valid, 2] == 0.0) | (seq[valid, 2] == 1.0))


def test_image_to_stroke_sequence_is_deterministic() -> None:
    image = _toy_letter_image()
    seq1, mask1 = image_to_stroke_sequence(image, max_len=120)
    seq2, mask2 = image_to_stroke_sequence(image, max_len=120)

    assert np.allclose(seq1, seq2)
    assert np.allclose(mask1, mask2)


def test_convert_images_to_stroke_cache_deterministic_splits(tmp_path) -> None:
    base = _toy_letter_image()
    images = np.stack(
        [
            base,
            np.rot90(base),
            np.flipud(base),
            np.fliplr(base),
            np.roll(base, shift=2, axis=0),
            np.roll(base, shift=2, axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

    out1 = tmp_path / "cache1.npz"
    out2 = tmp_path / "cache2.npz"

    convert_images_to_stroke_cache(images, labels, out1, max_len=64, seed=7)
    convert_images_to_stroke_cache(images, labels, out2, max_len=64, seed=7)

    p1 = np.load(out1)
    p2 = np.load(out2)

    assert np.allclose(p1["sequences"], p2["sequences"])
    assert np.allclose(p1["masks"], p2["masks"])
    assert np.array_equal(p1["train_idx"], p2["train_idx"])
    assert np.array_equal(p1["val_idx"], p2["val_idx"])
    assert np.array_equal(p1["test_idx"], p2["test_idx"])
