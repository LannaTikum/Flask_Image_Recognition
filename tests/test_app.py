"""Tests for the Flask routes in app.py (the model is mocked)."""

import io

from conftest import PREDICTED_DIGIT

NO_FILE_MESSAGE = b"No file was uploaded."
CANNOT_PROCESS_MESSAGE = b"File cannot be processed."


# --- Home page ---------------------------------------------------------------

def test_home_page_loads(client):
    """GET / returns 200 and the page title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data


def test_home_page_contains_upload_form(client):
    """The home page posts a multipart form with a 'file' input to /prediction."""
    html = client.get("/").data
    assert b'action="/prediction"' in html
    assert b'enctype="multipart/form-data"' in html
    assert b'name="file"' in html


def test_unknown_route_returns_404(client):
    """Requests for pages that do not exist return 404."""
    assert client.get("/does-not-exist").status_code == 404


# --- Valid predictions --------------------------------------------------------

def test_prediction_with_valid_png_returns_digit(client, fake_model, make_image):
    """A valid PNG upload renders the result page with the predicted digit."""
    data = {"file": (make_image(fmt="PNG"), "hand.png")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Prediction" in response.data
    assert f">{PREDICTED_DIGIT}<".encode() in response.data
    fake_model.predict.assert_called_once()


def test_prediction_with_valid_jpeg_returns_digit(client, fake_model, make_image):
    """JPEG images (the format of the sample images) are accepted too."""
    data = {"file": (make_image(fmt="JPEG", size=(400, 400)), "sign.jpeg")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert f">{PREDICTED_DIGIT}<".encode() in response.data
    fake_model.predict.assert_called_once()


def test_prediction_sends_correctly_shaped_input_to_model(client, fake_model, make_image):
    """The route passes a (1, 224, 224, 3) array to the model."""
    data = {"file": (make_image(size=(31, 77)), "odd_size.png")}
    client.post("/prediction", data=data, content_type="multipart/form-data")
    model_input = fake_model.predict.call_args.args[0]
    assert model_input.shape == (1, 224, 224, 3)


def test_prediction_with_grayscale_image(client, fake_model, make_image):
    """Edge case: a grayscale upload is still predicted instead of failing."""
    data = {"file": (make_image(mode="L"), "gray.png")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert f">{PREDICTED_DIGIT}<".encode() in response.data
    fake_model.predict.assert_called_once()


# --- Missing and invalid input ------------------------------------------------

def test_prediction_without_file_field_returns_400(client, fake_model):
    """A POST with no 'file' field is a bad request and says no file was sent."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert NO_FILE_MESSAGE in response.data
    fake_model.predict.assert_not_called()


def test_prediction_with_empty_filename_returns_400(client, fake_model):
    """Submitting the form without choosing a file is a bad request."""
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert NO_FILE_MESSAGE in response.data
    fake_model.predict.assert_not_called()


def test_prediction_with_text_file_returns_400(client, fake_model):
    """A file that is not an image is rejected with an error message."""
    data = {"file": (io.BytesIO(b"this is not an image"), "notes.txt")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert CANNOT_PROCESS_MESSAGE in response.data
    fake_model.predict.assert_not_called()


def test_prediction_with_corrupted_image_returns_400(client, fake_model, make_image):
    """A truncated image file is rejected with an error message."""
    truncated = io.BytesIO(make_image(fmt="PNG").getvalue()[:40])
    data = {"file": (truncated, "broken.png")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert CANNOT_PROCESS_MESSAGE in response.data
    fake_model.predict.assert_not_called()


def test_prediction_rejects_get_requests(client):
    """/prediction only accepts POST; GET returns 405 Method Not Allowed."""
    assert client.get("/prediction").status_code == 405
