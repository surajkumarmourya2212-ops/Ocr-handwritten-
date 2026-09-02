# Handwritten OCR Project - MNIST Updated

## Digit recognition
Uses the included `models/digit_model.keras` MNIST TensorFlow/Keras model.

## Text recognition
- Uses `models/crnn_model.h5` if available.
- Otherwise uses Tesseract OCR.

## Run with Streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py

## Run with Flask
pip install -r requirements.txt
python app.py

## Optional CRNN model
Place your existing trained text model here:
models/crnn_model.h5

The original `handwritten_ocr.py` training file should remain in your project if
you use the CRNN model and its character mapping.
