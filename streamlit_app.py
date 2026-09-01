"""
streamlit_app.py
-----------------
Streamlit deployment of the Handwritten OCR project — this is the file to
point Streamlit Community Cloud at (share.streamlit.io):

    1. Push this whole folder to a GitHub repo.
    2. On share.streamlit.io: New app -> pick the repo -> main file path:
       streamlit_app.py
    3. Streamlit Cloud installs requirements.txt automatically, and
       packages.txt (in this folder) tells it to apt-get install
       tesseract-ocr, which pytesseract needs at runtime.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import numpy as np
from PIL import Image
import streamlit as st

import ocr_utils

st.set_page_config(page_title="Handwritten OCR — Live Demo", page_icon="✍️", layout="centered")

st.title("✍️ Handwritten OCR — Live Demo")
st.caption("Draw or upload a handwritten digit or word to see the deployed model in action.")

# --------------------------------------------------------------------
# Model status (loaded once, cached across reruns)
# --------------------------------------------------------------------
@st.cache_resource
def get_status():
    ocr_utils.load_digit_model()
    crnn = ocr_utils.load_crnn_model()
    return {
        "crnn_available": crnn is not None,
        "tesseract_available": ocr_utils.tesseract_available(),
    }

status = get_status()

tab_digit, tab_text = st.tabs(["🔢 Single Digit (0-9)", "📝 Word / Sentence"])

# --------------------------------------------------------------------
# TAB 1 — Digit recognition (drawable canvas)
# --------------------------------------------------------------------
with tab_digit:
    st.write("Draw a single digit below, then click **Predict**.")

    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        drawn_image = canvas_result.image_data
    except ImportError:
        st.warning(
            "`streamlit-drawable-canvas` is not installed, so drawing is disabled. "
            "Add it to requirements.txt (already included) — falling back to file upload."
        )
        drawn_image = None

    col1, col2 = st.columns([1, 1])
    predict_clicked = col1.button("Predict Digit", type="primary")
    uploaded_digit = col2.file_uploader("...or upload a digit image", type=["png", "jpg", "jpeg"], key="digit_upload")

    if predict_clicked:
        gray = None
        if uploaded_digit is not None:
            gray = np.array(Image.open(uploaded_digit).convert("L"))
        elif drawn_image is not None and drawn_image.sum() > 0:
            rgba = drawn_image.astype("uint8")
            gray = np.array(Image.fromarray(rgba).convert("L"))
        else:
            st.error("Please draw a digit or upload an image first.")

        if gray is not None:
            with st.spinner("Predicting..."):
                try:
                    result = ocr_utils.predict_digit(gray)
                    st.success(f"Prediction: **{result['prediction']}**")
                    st.caption(f"Confidence: {result['confidence']*100:.1f}% · Engine: {result['engine']}")
                except Exception as e:
                    st.error(f"Error: {e}")

# --------------------------------------------------------------------
# TAB 2 — Word / sentence recognition
# --------------------------------------------------------------------
with tab_text:
    engine_label = (
        "CRNN (trained model.h5)" if status["crnn_available"]
        else "Tesseract fallback" if status["tesseract_available"]
        else "No text engine available"
    )
    st.info(f"Engine in use: **{engine_label}**")

    uploaded_text = st.file_uploader("Upload a photo/scan of a handwritten word or short sentence",
                                      type=["png", "jpg", "jpeg"], key="text_upload")

    if st.button("Recognise Text", type="primary"):
        if uploaded_text is None:
            st.error("Please choose an image first.")
        else:
            gray = np.array(Image.open(uploaded_text).convert("L"))
            with st.spinner("Recognising..."):
                try:
                    result = ocr_utils.predict_text(gray)
                    st.success(f"Prediction: **{result['prediction'] or '(no text detected)'}**")
                    st.caption(f"Engine: {result['engine']}")
                except Exception as e:
                    st.error(f"Error: {e}")

st.divider()
st.caption(
    "Digit model: MNIST CNN for handwritten digit recognition. "
    "Text mode uses the trained CRNN when available; "
    "otherwise Tesseract is used as a fallback."
)
