"""Unit tests for the preprocessing and prediction helpers in model.py."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import UnidentifiedImageError

import model

EXPECTED_SHAPE = (1, 224, 224, 3)


# --- preprocess_img: normal cases --------------------------------------------

def test_preprocess_returns_model_input_shape(make_image):
    """An RGB image becomes a single-image batch of 224x224x3."""
    result = model.preprocess_img(make_image(size=(64, 64)))
    assert result.shape == EXPECTED_SHAPE


@pytest.mark.parametrize("size", [(1, 1), (31, 77), (224, 224), (1000, 600)])
def test_preprocess_resizes_any_input_size(make_image, size):
    """Tiny, odd, exact and large images are all resized to 224x224."""
    assert model.preprocess_img(make_image(size=size)).shape == EXPECTED_SHAPE


def test_preprocess_scales_pixels_to_unit_range(make_image):
    """Pixel values are scaled from 0-255 to 0.0-1.0."""
    white = model.preprocess_img(make_image(color=(255, 255, 255)))
    black = model.preprocess_img(make_image(color=(0, 0, 0)))
    assert np.allclose(white, 1.0)
    assert np.allclose(black, 0.0)
    assert white.dtype == np.float32


def test_preprocess_accepts_a_file_path(tmp_path, make_image):
    """preprocess_img also works with a path on disk, like the sample images."""
    image_path = tmp_path / "sample.jpeg"
    image_path.write_bytes(make_image(fmt="JPEG").getvalue())
    assert model.preprocess_img(image_path).shape == EXPECTED_SHAPE


# --- preprocess_img: edge cases ----------------------------------------------

@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "LA"])
def test_preprocess_converts_non_rgb_images(make_image, mode):
    """Grayscale, transparent and palette images are converted to 3 channels."""
    assert model.preprocess_img(make_image(mode=mode)).shape == EXPECTED_SHAPE


def test_preprocess_keeps_colour_values_of_rgba_images(make_image):
    """Converting RGBA to RGB drops the alpha channel but keeps the colour."""
    result = model.preprocess_img(make_image(mode="RGBA", color=(255, 0, 0, 255)))
    assert np.allclose(result[0, 0, 0], [1.0, 0.0, 0.0])


def test_preprocess_rejects_non_image_data():
    """Data that is not an image raises PIL's UnidentifiedImageError."""
    with pytest.raises(UnidentifiedImageError):
        model.preprocess_img(io.BytesIO(b"definitely not an image"))


def test_preprocess_rejects_empty_file():
    """An empty upload cannot be opened as an image."""
    with pytest.raises(UnidentifiedImageError):
        model.preprocess_img(io.BytesIO(b""))


# --- predict_result ------------------------------------------------------------

@pytest.mark.parametrize("digit", [0, 3, 9])
def test_predict_result_returns_index_of_highest_probability(monkeypatch, digit):
    """predict_result returns the class with the highest probability."""
    probabilities = np.full((1, 10), 0.01, dtype="float32")
    probabilities[0, digit] = 0.91
    fake_model = MagicMock()
    fake_model.predict.return_value = probabilities
    monkeypatch.setattr(model, "get_model", lambda: fake_model)

    assert model.predict_result(np.zeros(EXPECTED_SHAPE)) == digit


def test_predict_result_returns_plain_int(fake_model):
    """The prediction is a Python int (not numpy.int64) so it renders cleanly."""
    result = model.predict_result(np.zeros(EXPECTED_SHAPE))
    assert isinstance(result, int)
    fake_model.predict.assert_called_once()


# --- get_model -----------------------------------------------------------------

def test_get_model_loads_model_once_and_caches_it(monkeypatch):
    """The .h5 file is loaded only once, from the path next to model.py."""
    loader = MagicMock(return_value="loaded-model")
    monkeypatch.setattr(model, "load_model", loader)
    model.get_model.cache_clear()
    try:
        assert model.get_model() == "loaded-model"
        assert model.get_model() == "loaded-model"
    finally:
        model.get_model.cache_clear()

    loader.assert_called_once_with(str(model.MODEL_PATH), compile=False)


def test_model_file_exists_next_to_module():
    """The trained model file ships with the repository."""
    assert model.MODEL_PATH.is_file()
    assert model.MODEL_PATH.name == "digit_model.h5"
