"""Model loading, image preprocessing and prediction for the hand-sign digit app."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# Resolve the model relative to this file so the app works from any directory.
MODEL_PATH = Path(__file__).resolve().parent / "digit_model.h5"
IMAGE_SIZE = (224, 224)


@lru_cache(maxsize=1)
def get_model():
    """Load the trained Keras model once and reuse it for every prediction.

    ``compile=False`` is used because the model is only used for inference,
    so the optimizer state saved with the legacy model is not needed.
    """
    return load_model(str(MODEL_PATH), compile=False)


def preprocess_img(img_path):
    """Convert an image file (path or file-like object) into model input.

    Returns a float array of shape (1, 224, 224, 3) scaled to the range [0, 1].
    Grayscale, palette and transparent images are converted to RGB first,
    because the model expects exactly three colour channels.
    """
    op_img = Image.open(img_path).convert("RGB")
    img_resize = op_img.resize(IMAGE_SIZE)
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return img_reshape


def predict_result(img_array):
    """Return the predicted digit (0-9) for a preprocessed image array."""
    pred = get_model().predict(img_array)
    return int(np.argmax(pred[0], axis=-1))
