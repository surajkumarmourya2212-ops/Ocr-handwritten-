"""
ocr_utils.py
------------
Framework-agnostic model loading, pre-processing and prediction logic,
shared by both deployment front-ends in this repo:

  - app.py           (Flask, for Render/Railway/PythonAnywhere/Docker etc.)
  - streamlit_app.py (Streamlit, for Streamlit Community Cloud)

Keeping this logic in one place means both UIs stay in sync and there is
exactly one place to fix bugs or swap in the trained CRNN model.
"""

import os
import logging
import numpy as np
import cv2
import joblib

log = logging.getLogger("ocr_utils")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DIGIT_MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.pkl")
CRNN_MODEL_PATH = os.path.join(MODEL_DIR, "crnn_model.h5")


# --------------------------------------------------------------------------
# MODEL LOADING (cached — call once, reuse everywhere)
# --------------------------------------------------------------------------
_digit_model = None
_crnn_model = None
_crnn_ready = False
_tesseract_ready = False


def load_digit_model():
    global _digit_model
    if _digit_model is None:
        if not os.path.isfile(DIGIT_MODEL_PATH):
            raise RuntimeError(
                f"Digit model not found at {DIGIT_MODEL_PATH}. Run `python train_model.py` first."
            )
        _digit_model = joblib.load(DIGIT_MODEL_PATH)
        log.info("Loaded digit recognition model from %s", DIGIT_MODEL_PATH)
    return _digit_model


def load_crnn_model():
    """Returns (inference_model_or_None, ctc_greedy_decode_fn_or_None, preprocess_fn_or_None)."""
    global _crnn_model, _crnn_ready
    if _crnn_ready:
        return _crnn_model
    _crnn_ready = True  # only try once per process
    if os.path.isfile(CRNN_MODEL_PATH):
        try:
            import tensorflow as tf  # noqa: F401
            from handwritten_ocr import (
                build_crnn_model, build_inference_model,
                preprocess_image, ctc_greedy_decode,
            )
            training_model = build_crnn_model(max_label_len=32)
            training_model.load_weights(CRNN_MODEL_PATH)
            inf_model = build_inference_model(training_model)
            _crnn_model = {
                "model": inf_model,
                "preprocess": preprocess_image,
                "decode": ctc_greedy_decode,
            }
            log.info("Loaded CRNN word-level model from %s", CRNN_MODEL_PATH)
        except Exception as e:
            log.warning("Could not load CRNN model (%s). Falling back to Tesseract.", e)
            _crnn_model = None
    else:
        log.info("No CRNN weights at %s — text mode will use Tesseract fallback.", CRNN_MODEL_PATH)
    return _crnn_model


def tesseract_available():
    global _tesseract_ready
    try:
        import pytesseract  # noqa: F401
        _tesseract_ready = True
    except ImportError:
        _tesseract_ready = False
    return _tesseract_ready


# --------------------------------------------------------------------------
# PRE-PROCESSING
# --------------------------------------------------------------------------
def prepare_digit_input(gray_img: np.ndarray) -> np.ndarray:
    """
    Reproduces sklearn's `load_digits` preprocessing so a live drawn/uploaded
    image matches the training distribution: crop to ink bounding box,
    resize to 8x8, normalise to the 0-16 range used by the training data.
    """
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(2, int(0.15 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray_img.shape[1], x + w + pad), min(gray_img.shape[0], y + h + pad)
        cropped = gray_img[y0:y1, x0:x1]
    else:
        cropped = gray_img

    resized = cv2.resize(cropped, (8, 8), interpolation=cv2.INTER_AREA)
    if resized.mean() > 127:
        resized = 255 - resized
    scaled = (resized.astype("float32") / 255.0) * 16.0
    return scaled.flatten().reshape(1, -1)


def predict_digit(gray_img: np.ndarray):
    model = load_digit_model()
    features = prepare_digit_input(gray_img)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(np.max(proba))
    return {
        "prediction": str(int(pred)),
        "confidence": round(confidence, 4),
        "engine": "sklearn-MLP (models/digit_model.pkl)",
    }


def predict_text(gray_img: np.ndarray, tmp_path: str = None):
    """
    Recognises a word/sentence. Prefers the trained CRNN if available,
    otherwise falls back to Tesseract OCR.
    """
    crnn = load_crnn_model()
    if crnn is not None:
        if tmp_path is None:
            tmp_path = os.path.join(BASE_DIR, "_tmp_ocr_input.png")
        from PIL import Image
        Image.fromarray(gray_img).save(tmp_path)
        try:
            img = crnn["preprocess"](tmp_path)
            img = np.expand_dims(img, axis=0)
            pred = crnn["model"].predict(img, verbose=0)
            text = crnn["decode"](pred, max_label_len=32)[0]
            return {"prediction": text, "engine": "CRNN + CTC (models/crnn_model.h5)"}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if tesseract_available():
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.fromarray(gray_img), config="--psm 7").strip()
        return {"prediction": text, "engine": "Tesseract OCR (fallback)"}

    raise RuntimeError(
        "No text-recognition engine available. Install pytesseract, or place a "
        "trained crnn_model.h5 in models/ and install tensorflow."
    )
