import streamlit as st
import numpy as np
from PIL import Image
import ocr_utils


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# LOAD MODELS
# ============================================================

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


# ============================================================
# TITLE
# ============================================================

st.title("✍️ Handwritten OCR System")

st.write(
    "Recognise handwritten digits, words and sentences."
)


# ============================================================
# TEXT ENGINE STATUS
# ============================================================

if crnn_model is not None:
    text_engine = "CRNN + CTC"

elif ocr_utils.tesseract_available():
    text_engine = "Tesseract OCR"

else:
    text_engine = "No text engine available"


st.caption(
    f"Digit Model: MNIST TensorFlow/Keras | "
    f"Text Engine: {text_engine}"
)


# ============================================================
# CREATE TABS
# ============================================================

tab_digit, tab_text = st.tabs([
    "🔢 Digit Recognition",
    "📝 Text Recognition"
])


# ============================================================
# DIGIT RECOGNITION TAB
# ============================================================

with tab_digit:

    st.subheader("Handwritten Digit Recognition")

    st.write(
        "Upload an image containing one handwritten digit (0-9)."
    )


    uploaded_digit = st.file_uploader(
        "📷 Upload handwritten digit",
        type=["png", "jpg", "jpeg", "webp"],
        key="digit_upload"
    )


    if uploaded_digit is not None:

        # Open image
        image = Image.open(uploaded_digit).convert("RGB")


        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True
        )


        # Prediction button
        if st.button(
            "🔍 Predict Digit",
            type="primary",
            key="predict_digit"
        ):

            try:

                image_array = np.array(image)


                with st.spinner("Predicting digit..."):

                    result = ocr_utils.predict_digit(
                        image_array
                    )


                # Display prediction
                st.success(
                    f"✅ Predicted Digit: "
                    f"{result['prediction']}"
                )


            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


    else:

        st.info(
            "Upload an image containing one handwritten digit."
        )


# ============================================================
# TEXT RECOGNITION TAB
# ============================================================

with tab_text:

    st.subheader("Handwritten Text Recognition")

    st.write(
        "Upload an image containing a handwritten word "
        "or short sentence."
    )


    st.info(
        f"Text engine currently available: "
        f"**{text_engine}**"
    )


    uploaded_text = st.file_uploader(
        "📷 Upload handwritten text",
        type=["png", "jpg", "jpeg", "webp"],
        key="text_upload"
    )


    if uploaded_text is not None:

        # Open image
        image = Image.open(uploaded_text).convert("RGB")


        # Display image
        st.image(
            image,
            caption="Uploaded Text Image",
            use_column_width=True
        )


        # Recognition button
        if st.button(
            "📝 Recognise Text",
            type="primary",
            key="predict_text"
        ):

            try:

                image_array = np.array(image)


                with st.spinner("Recognising text..."):

                    result = ocr_utils.predict_text(
                        image_array
                    )


                st.success(
                    "Recognition Complete"
                )


                st.text_area(
                    "Recognised Text:",
                    value=result["prediction"],
                    height=150
                )


                st.caption(
                    f"Engine: {result['engine']}"
                )


            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Digit Recognition: MNIST TensorFlow/Keras Model | "
    "Text Recognition: CRNN + CTC or Tesseract OCR"
)
