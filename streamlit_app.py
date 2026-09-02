import streamlit as st
import numpy as np
from PIL import Image
import ocr_utils


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Handwritten OCR System",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# CUSTOM CSS - PROFESSIONAL UI
# --------------------------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: #F8FAFC;
}

/* Main Container */
.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit Branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ---------------- HEADER ---------------- */

.hero-container {
    padding: 25px 28px;
    border-radius: 16px;
    background: linear-gradient(135deg, #151B28, #111827);
    border: 1px solid #263244;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 34px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #94A3B8;
    margin-bottom: 15px;
}

.tech-badge {
    display: inline-block;
    background-color: #1E293B;
    color: #CBD5E1;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin-right: 6px;
}


/* ---------------- TABS ---------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #263244;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    background-color: transparent;
    border-radius: 8px 8px 0 0;
    color: #94A3B8;
    font-size: 15px;
    font-weight: 600;
    padding: 0px 20px;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 3px solid #3B82F6 !important;
}


/* ---------------- CARDS ---------------- */

.section-card {
    background-color: #151B28;
    border: 1px solid #263244;
    border-radius: 14px;
    padding: 25px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 8px;
}

.section-description {
    color: #94A3B8;
    font-size: 15px;
}


/* ---------------- BUTTON ---------------- */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.2s;
}

.stButton > button[kind="primary"] {
    background-color: #2563EB;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1D4ED8;
    transform: translateY(-1px);
}


/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {
    background-color: #111827;
    border: 1px dashed #475569;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6;
}


/* ---------------- RESULT CARD ---------------- */

.result-card {
    background: linear-gradient(135deg, #0F2A1D, #10251C);
    border: 1px solid #1F7A45;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #86EFAC;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: #FFFFFF;
    font-size: 60px;
    font-weight: 800;
    margin-top: 5px;
}


/* ---------------- TEXT RESULT ---------------- */

.text-result {
    background-color: #111827;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
}

.footer-text {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding-top: 15px;
}


/* ---------------- ALERTS ---------------- */

.stAlert {
    border-radius: 10px;
}


/* ---------------- IMAGE ---------------- */

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
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


# --------------------------------------------------
# DETECT TEXT ENGINE
# --------------------------------------------------
if crnn_model is not None:
    text_engine = "CRNN + CTC"

elif ocr_utils.tesseract_available():
    text_engine = "Tesseract OCR"

else:
    text_engine = "No text engine available"


# --------------------------------------------------
# PROFESSIONAL HEADER
# --------------------------------------------------
st.markdown(f"""
<div class="hero-container">

<div class="hero-title">
✍️ Handwritten OCR System
</div>

<div class="hero-subtitle">
Recognise handwritten digits, words and short sentences using Artificial Intelligence.
</div>

<span class="tech-badge">MNIST Digit Model</span>
<span class="tech-badge">{text_engine}</span>
<span class="tech-badge">TensorFlow / Keras</span>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_digit, tab_text = st.tabs([
    "🔢 Digit Recognition",
    "📝 Text Recognition"
])


# ==================================================
# DIGIT RECOGNITION TAB
# ==================================================
with tab_digit:

    st.markdown("""
    <div class="section-card">

    <div class="section-title">
    Handwritten Digit Recognition
    </div>

    <div class="section-description">
    Upload an image containing a single handwritten digit from 0 to 9.
    </div>

    </div>
    """, unsafe_allow_html=True)


    uploaded_digit = st.file_uploader(
        "Upload Handwritten Digit",
        type=["png", "jpg", "jpeg", "webp"],
        key="digit_upload"
    )


    if uploaded_digit is not None:

        try:
            image = Image.open(uploaded_digit).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )


            if st.button(
                "🔍 Predict Digit",
                type="primary",
                key="predict_digit"
            ):

                try:
                    image_array = np.array(image)

                    with st.spinner("Analyzing handwritten digit..."):
                        result = ocr_utils.predict_digit(image_array)


                    # Professional Result Display
                    st.markdown(f"""
                    <div class="result-card">

                    <div class="result-label">
                    Predicted Digit
                    </div>

                    <div class="result-value">
                    {result['prediction']}
                    </div>

                    </div>
                    """, unsafe_allow_html=True)


                    st.caption(
                        f"Prediction Engine: {result['engine']}"
                    )


                except Exception as e:
                    st.error(f"❌ Prediction Error: {e}")


        except Exception as e:
            st.error("❌ Unable to process this image.")
            st.code(str(e))


    else:
        st.info("📤 Upload a clear image containing one handwritten digit.")



# ==================================================
# TEXT RECOGNITION TAB
# ==================================================
with tab_text:

    st.markdown("""
    <div class="section-card">

    <div class="section-title">
    Handwritten Text Recognition
    </div>

    <div class="section-description">
    Upload an image containing a handwritten word or short sentence.
    </div>

    </div>
    """, unsafe_allow_html=True)


    st.caption(f"Current Recognition Engine: {text_engine}")


    uploaded_text = st.file_uploader(
        "Upload Handwritten Text",
        type=["png", "jpg", "jpeg", "webp"],
        key="text_upload"
    )


    if uploaded_text is not None:

        try:
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

                    with st.spinner("Recognising handwritten text..."):
                        result = ocr_utils.predict_text(image_array)


                    st.success("Recognition Complete")


                    st.markdown("""
                    <div class="text-result">
                    <strong>Recognised Text</strong>
                    </div>
                    """, unsafe_allow_html=True)


                    st.text_area(
                        "Output",
                        value=result["prediction"],
                        height=150,
                        key="text_output"
                    )


                    st.caption(
                        f"Recognition Engine: {result['engine']}"
                    )


                except Exception as e:
                    st.error(f"❌ Recognition Error: {e}")


        except Exception as e:
            st.error("❌ Unable to process this image.")
            st.code(str(e))


    else:
        st.info(
            "📤 Upload an image containing handwritten words or a short sentence."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(f"""
<div class="footer-text">

Handwritten OCR System <br>
Digit Recognition: MNIST TensorFlow/Keras Model |
Text Recognition: {text_engine}

</div>
""", unsafe_allow_html=True)
