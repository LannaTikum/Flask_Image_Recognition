"""End-to-end tests that use the real trained model.

These tests need TensorFlow. They are skipped automatically if it is not
installed, and they run in CI where requirements.txt is installed.
"""

from pathlib import Path

import numpy as np
import pytest

import model

pytest.importorskip("tensorflow")

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "test_images"


def sample_image(digit):
    """Return the first sample image provided for a digit."""
    return sorted((SAMPLE_DIR / str(digit)).glob("*.jpeg"))[0]


def test_real_model_outputs_probabilities_for_ten_digits():
    """The loaded model returns 10 softmax probabilities for one image."""
    probabilities = model.get_model().predict(model.preprocess_img(sample_image(3)))
    assert probabilities.shape == (1, 10)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-3)


@pytest.mark.parametrize("digit", range(10))
def test_real_model_predicts_a_valid_digit(digit):
    """Every sample image gets a prediction between 0 and 9."""
    prediction = model.predict_result(model.preprocess_img(sample_image(digit)))
    assert 0 <= prediction <= 9


def test_upload_sample_image_end_to_end(client):
    """Uploading a sample image through the web form returns a digit."""
    image = sample_image(5)
    with image.open("rb") as handle:
        data = {"file": (handle, image.name)}
        response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert any(f">{digit}<".encode() in response.data for digit in range(10))
