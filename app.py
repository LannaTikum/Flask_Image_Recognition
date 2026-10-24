"""Flask web application that predicts hand-sign digits from uploaded images."""

from flask import Flask, render_template, request
from PIL import UnidentifiedImageError

from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


NO_FILE_ERROR = "No file was uploaded."
INVALID_FILE_ERROR = "File cannot be processed."


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Predict the digit in the uploaded image and render the result page.

    The route only accepts POST, so Flask itself rejects other methods and
    no explicit ``request.method`` check is needed. Bad uploads are client
    errors, so they render the error message with HTTP 400.
    """
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return render_template("result.html", err=NO_FILE_ERROR), 400

    try:
        img = preprocess_img(uploaded.stream)
    except (UnidentifiedImageError, OSError, ValueError):
        # UnidentifiedImageError: not an image; OSError: truncated/corrupted image.
        return render_template("result.html", err=INVALID_FILE_ERROR), 400

    pred = predict_result(img)
    return render_template("result.html", predictions=str(pred))


if __name__ == "__main__":
    app.run(port=9000, debug=True)
