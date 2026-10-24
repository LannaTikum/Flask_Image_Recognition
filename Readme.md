# Hand Digit Image Recognition

This web app is made using Flask framework and is all about predicting the hand sign digit images based on the deep learning model trained using transfer learning.

![Python](https://img.shields.io/badge/python-8338ec)
![Keras](https://img.shields.io/badge/keras-219ebc)
![Numpy](https://img.shields.io/badge/numpy-ffafcc)
![Pillow](https://img.shields.io/badge/pillow-00bbf9)
![Flask](https://img.shields.io/badge/flask-f72585)
![Bootstrap](https://img.shields.io/badge/bootstrap-57cc99)

# Glimpse

https://user-images.githubusercontent.com/72191416/201943098-c8f5fd8b-ec7d-4e5d-883d-8b69109b946f.mp4

# Dependencies

- Flask
- TensorFlow (Keras)
- Numpy
- Pillow (PIL)

## UI Framework

- Bootstrap

# Get Started

Requires **Python 3.11** (TensorFlow 2.15 supports Python 3.9 to 3.11).

```commandline
git clone <your-repository-url>
cd Flask_Image_Recognition
python -m venv venv
```

Activate the virtual environment:

```commandline
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

# Install Dependencies

```commandline
pip install -r requirements.txt        # run the app
pip install -r requirements-dev.txt    # app + pylint, pytest, pre-commit
```

# Run

```commandline
flask --app app.py run --debug
```

Open http://127.0.0.1:5000, choose an image from the `test_images` folder, and click **Submit**.

# Code Quality and Tests

```commandline
pylint app.py model.py tests      # static analysis (.pylintrc)
pytest                            # unit tests + real-model integration tests
pre-commit install                # run checks automatically on commit/push
pre-commit run --all-files        # run every hook on the whole repository
```

Unit tests mock the Keras model, so they run in under a second. Tests in
`tests/test_integration.py` load the real model and are skipped if TensorFlow
is not installed.

# Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request: it checks out
the code, sets up Python 3.11, installs `requirements-dev.txt`, runs Pylint and
pytest, and writes a pass/fail summary to the run page.
