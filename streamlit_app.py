import streamlit as st
import numpy as np
from PIL import Image
import ocr_utils

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="centered"
)

@st.cache_resource
def load_models():
    digit_model = ocr_utils.load_digit_model()
    crnn_model = ocr_utils.load_crnn_model()
    return digit_model, crnn_model

try:
    digit_model, crnn_model = load_models()
except Exception as e:
    st.error("❌ Could not load the MNIST digit model.")
    st.code(str(e))
    st.stop()

st.title("✍️ Handwritten OCR System")
st.write("Recognise handwritten digits, words and sentences.")

if crnn_model is not None:
    text_engine = "CRNN + CTC"
elif ocr_utils.tesseract_available():
    text_engine = "Tesseract OCR"
else:
    text_engine = "No text engine available"

st.caption(
    f"Digit Model: MNIST TensorFlow/Keras | Text Engine: {text_engine}"
)

tab_digit, tab_text = st.tabs([
    "🔢 Digit Recognition",
    "📝 Text Recognition"
])

with tab_digit:
    st.subheader("Handwritten Digit Recognition")
    st.write("Upload an image containing one handwritten digit (0-9).")

    uploaded_digit = st.file_uploader(
        "📷 Upload handwritten digit",
        type=["png", "jpg", "jpeg", "webp"],
        key="digit_upload"
    )

    if uploaded_digit is not None:
        image = Image.open(uploaded_digit).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button(
            "🔍 Predict Digit",
            type="primary",
            key="predict_digit"
        ):
            try:
                image_array = np.array(image)

                with st.spinner("Predicting digit..."):
                    result = ocr_utils.predict_digit(image_array)

                st.success(
                    f"✅ Predicted Digit: {result['prediction']}"
                )
                st.info(
                    f"Confidence: {result['confidence'] * 100:.2f}%"
                )
                st.caption(f"Engine: {result['engine']}")

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.info("Upload an image containing one handwritten digit.")

with tab_text:
    st.subheader("Handwritten Text Recognition")
    st.write(
        "Upload an image containing a handwritten word or short sentence."
    )
    st.info(f"Text engine currently available: **{text_engine}**")

    uploaded_text = st.file_uploader(
        "📷 Upload handwritten text",
        type=["png", "jpg", "jpeg", "webp"],
        key="text_upload"
    )

    if uploaded_text is not None:
        image = Image.open(uploaded_text).convert("RGB")
        st.image(
            image,
            caption="Uploaded Text Image",
            use_container_width=True
        )

        if st.button(
            "📝 Recognise Text",
            type="primary",
            key="predict_text"
        ):
            try:
                image_array = np.array(image)

                with st.spinner("Recognising text..."):
                    result = ocr_utils.predict_text(image_array)

                st.success("Recognition Complete")
                st.text_area(
                    "Recognised Text:",
                    value=result["prediction"],
                    height=150
                )
                st.caption(f"Engine: {result['engine']}")

            except Exception as e:
                st.error(f"❌ Error: {e}")

st.divider()
st.caption(
    "Digit Recognition: MNIST TensorFlow/Keras Model | "
    "Text Recognition: CRNN + CTC or Tesseract OCR"
)
