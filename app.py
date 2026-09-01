"""
app.py
------
Flask deployment of the Handwritten OCR project (for Render, Railway,
PythonAnywhere, Fly.io, Docker, or any plain Python host).

For Streamlit Community Cloud deployment, use streamlit_app.py instead —
Streamlit Cloud runs `streamlit run <file>.py`, not a Flask app.

Both front-ends share the same model/preprocessing code in ocr_utils.py.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://127.0.0.1:5000

Production:
    gunicorn -w 2 -b 0.0.0.0:8000 app:app
"""

import io
import base64
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

import ocr_utils

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = app.logger

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

# Warm up models once at startup (fails fast if digit_model.pkl is missing)
ocr_utils.load_digit_model()
_crnn = ocr_utils.load_crnn_model()
CRNN_AVAILABLE = _crnn is not None
TESSERACT_AVAILABLE = ocr_utils.tesseract_available()


def _decode_image(file_storage=None, b64_data=None) -> np.ndarray:
    if file_storage is not None:
        img = Image.open(file_storage.stream).convert("L")
    elif b64_data is not None:
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("L")
    else:
        raise ValueError("No image data provided")
    return np.array(img)


@app.route("/")
def index():
    return render_template(
        "index.html",
        crnn_available=CRNN_AVAILABLE,
        tesseract_available=TESSERACT_AVAILABLE,
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "digit_model_loaded": True,
        "crnn_model_loaded": CRNN_AVAILABLE,
        "tesseract_available": TESSERACT_AVAILABLE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    mode = request.form.get("mode", "digit")

    try:
        if "file" in request.files and request.files["file"].filename:
            gray = _decode_image(file_storage=request.files["file"])
        elif request.form.get("image_data"):
            gray = _decode_image(b64_data=request.form.get("image_data"))
        else:
            return jsonify({"error": "No image provided"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    try:
        if mode == "digit":
            result = ocr_utils.predict_digit(gray)
            result["mode"] = "digit"
            return jsonify(result)
        elif mode == "text":
            result = ocr_utils.predict_text(gray)
            result["mode"] = "text"
            return jsonify(result)
        else:
            return jsonify({"error": f"Unknown mode '{mode}'. Use 'digit' or 'text'."}), 400
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
