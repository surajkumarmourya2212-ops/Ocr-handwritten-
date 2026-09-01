import os
import logging
import numpy as np
import cv2

log = logging.getLogger("ocr_utils")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

DIGIT_MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")
CRNN_MODEL_PATH = os.path.join(MODEL_DIR, "crnn_model.h5")

_digit_model = None
_crnn_model = None
_crnn_ready = False
_tesseract_ready = False


# ============================================================
# DIGIT MODEL
# ============================================================

def load_digit_model():
    global _digit_model

    if _digit_model is None:
        if not os.path.isfile(DIGIT_MODEL_PATH):
            raise RuntimeError(
                f"Digit model not found at {DIGIT_MODEL_PATH}"
            )

        from tensorflow import keras

        _digit_model = keras.models.load_model(DIGIT_MODEL_PATH)

        log.info("Loaded MNIST CNN digit model")

    return _digit_model


def prepare_digit_input(gray_img):

    # Make sure grayscale
    gray_img = np.asarray(gray_img).astype(np.uint8)

    # Threshold
    _, thresh = cv2.threshold(
        gray_img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find handwriting
    coords = cv2.findNonZero(thresh)

    if coords is not None:

        x, y, w, h = cv2.boundingRect(coords)

        # Make square
        size = max(w, h)

        cx = x + w // 2
        cy = y + h // 2

        x0 = max(0, cx - size // 2)
        y0 = max(0, cy - size // 2)

        x1 = min(gray_img.shape[1], x0 + size)
        y1 = min(gray_img.shape[0], y0 + size)

        cropped = gray_img[y0:y1, x0:x1]

    else:
        cropped = gray_img

    # Resize to 28x28
    resized = cv2.resize(
        cropped,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Convert to black background / white digit
    _, binary = cv2.threshold(
        resized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure digit is white
    if np.mean(binary) > 127:
        binary = 255 - binary

    # Normalize
    image = binary.astype("float32") / 255.0

    # Shape: 1,28,28,1
    image = image.reshape(1, 28, 28, 1)

    return image


def predict_digit(gray_img):

    model = load_digit_model()

    image = prepare_digit_input(gray_img)

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    prediction = int(np.argmax(probabilities))

    confidence = float(
        np.max(probabilities)
    )

    return {
        "prediction": str(prediction),
        "confidence": round(confidence, 4),
        "engine": "MNIST CNN"
    }


# ============================================================
# CRNN MODEL
# ============================================================

def load_crnn_model():

    global _crnn_model
    global _crnn_ready

    if _crnn_ready:
        return _crnn_model

    _crnn_ready = True

    if not os.path.isfile(CRNN_MODEL_PATH):
        log.info("CRNN model not found")

        return None

    try:

        import tensorflow as tf

        from handwritten_ocr import (
            build_crnn_model,
            build_inference_model,
            preprocess_image,
            ctc_greedy_decode,
        )

        training_model = build_crnn_model(
            max_label_len=32
        )

        training_model.load_weights(
            CRNN_MODEL_PATH
        )

        inference_model = build_inference_model(
            training_model
        )

        _crnn_model = {
            "model": inference_model,
            "preprocess": preprocess_image,
            "decode": ctc_greedy_decode,
        }

        log.info("CRNN model loaded")

    except Exception as e:

        log.warning(
            f"Could not load CRNN: {e}"
        )

        _crnn_model = None

    return _crnn_model


# ============================================================
# TESSERACT
# ============================================================

def tesseract_available():

    global _tesseract_ready

    try:

        import pytesseract

        _tesseract_ready = True

    except ImportError:

        _tesseract_ready = False

    return _tesseract_ready


# ============================================================
# TEXT RECOGNITION
# ============================================================

def predict_text(gray_img, tmp_path=None):

    crnn = load_crnn_model()

    # Preferred: trained CRNN
    if crnn is not None:

        if tmp_path is None:

            tmp_path = os.path.join(
                BASE_DIR,
                "_tmp_ocr_input.png"
            )

        from PIL import Image

        Image.fromarray(gray_img).save(
            tmp_path
        )

        try:

            img = crnn["preprocess"](
                tmp_path
            )

            img = np.expand_dims(
                img,
                axis=0
            )

            prediction = crnn["model"].predict(
                img,
                verbose=0
            )

            text = crnn["decode"](
                prediction,
                max_label_len=32
            )[0]

            return {
                "prediction": text,
                "engine": "CRNN + CTC"
            }

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Fallback
    if tesseract_available():

        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(
            Image.fromarray(gray_img),
            config="--psm 7"
        ).strip()

        return {
            "prediction": text,
            "engine": "Tesseract OCR fallback"
        }

    raise RuntimeError(
        "No text recognition model is available."
    )
