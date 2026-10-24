"""Shared pytest fixtures for the hand-sign digit recognition app."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import model
from app import app as flask_app

PREDICTED_DIGIT = 7


def make_image_bytes(mode="RGB", size=(64, 64), fmt="PNG", color=None):
    """Return an in-memory image file with the given mode, size and format."""
    buffer = io.BytesIO()
    Image.new(mode, size, color=color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


@pytest.fixture(name="make_image")
def fixture_make_image():
    """Factory fixture: build an in-memory image file for a test."""
    return make_image_bytes


@pytest.fixture(name="client")
def fixture_client():
    """Flask test client with testing mode enabled."""
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        yield client


@pytest.fixture(name="fake_model")
def fixture_fake_model(monkeypatch):
    """Replace the Keras model with a mock that always predicts PREDICTED_DIGIT.

    The rest of the pipeline (upload handling, preprocessing, argmax) still
    runs for real, but no TensorFlow model has to be loaded.
    """
    probabilities = np.zeros((1, 10), dtype="float32")
    probabilities[0, PREDICTED_DIGIT] = 1.0
    mock_model = MagicMock()
    mock_model.predict.return_value = probabilities
    monkeypatch.setattr(model, "get_model", lambda: mock_model)
    return mock_model
