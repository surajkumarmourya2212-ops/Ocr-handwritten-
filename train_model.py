"""
train_model.py
--------------
Trains and saves the handwritten-digit recognition model used by app.py.

This is the lightweight model that ships with the deployment (real weights,
trained on real handwritten digit samples from the UCI/sklearn "digits"
dataset — 1,797 8x8 grayscale scans of actual handwritten digits, 0-9).

It plays the same role as the CRNN in the full report: given a pre-processed
handwritten glyph image, output the recognised character. It is deliberately
small so it trains in seconds on CPU with no GPU/internet required, and can
be dropped straight into a Flask/REST deployment.

For full word/sentence-level recognition, train handwritten_ocr.py (the
CRNN + CTC model) separately on a GPU machine with the IAM dataset and drop
the resulting weights file at models/crnn_model.h5 — app.py will
auto-detect and use it (see app.py's MODEL LOADING section).

Usage:
    python train_model.py
"""

import os
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.pkl")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading handwritten digits dataset (sklearn.datasets.load_digits)...")
    digits = load_digits()
    X, y = digits.data, digits.target   # X: (1797, 64) flattened 8x8 images, values 0-16

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}   Test samples: {len(X_test)}")

    # Pipeline: feature scaling + a small MLP (multi-layer perceptron).
    # This mirrors, at a much smaller scale, the "CNN features -> dense
    # classifier" idea used in the full CRNN, but trains instantly on CPU.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=15,
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")
    return acc


if __name__ == "__main__":
    train()
