"""Flask web application that predicts hand-sign digits from uploaded images."""

from flask import Flask, render_template, request
from PIL import UnidentifiedImageError

from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Predict the digit in the uploaded image and render the result page.

    The route only accepts POST, so Flask itself rejects other methods and
    no explicit ``request.method`` check is needed.
    """
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except (KeyError, UnidentifiedImageError, OSError, ValueError):
        # KeyError: no "file" field in the form.
        # UnidentifiedImageError/OSError: the upload is not a readable image.
        # ValueError: the image could not be reshaped for the model.
        error = "File cannot be processed."
        return render_template("result.html", err=error)


if __name__ == "__main__":
    app.run(port=9000, debug=True)
