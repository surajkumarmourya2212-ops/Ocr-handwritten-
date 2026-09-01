"""
==============================================================================
 Handwritten Text Recognition using a CRNN (CNN + BiLSTM) with CTC Loss
==============================================================================
Project   : Optical Character Recognition (OCR) for Handwritten Text
Framework : TensorFlow / Keras
Dataset   : IAM Handwriting Word Database (or any word-level image dataset
            organised as  image_path <TAB> transcription_label)

This module is written as a small, self-contained, production-style
package rather than a single notebook cell. It supports three modes from
the command line:

    python handwritten_ocr.py --mode train    --data_dir ./data --epochs 60
    python handwritten_ocr.py --mode evaluate --data_dir ./data --weights best_model.h5
    python handwritten_ocr.py --mode predict  --image path/to/word.png --weights best_model.h5

------------------------------------------------------------------------------
Pipeline stages
------------------------------------------------------------------------------
 1. Data ingestion & cleaning      -> HTRDataset
 2. Pre-processing / augmentation  -> preprocess_image()
 3. Model architecture (CRNN)      -> build_crnn_model()
 4. CTC loss / decoding            -> CTCLayer, ctc_greedy_decode()
 5. Training loop with callbacks   -> train()
 6. Evaluation (CER / WER)         -> evaluate()
 7. Single image inference         -> predict_single_image()
==============================================================================
"""

import os
import re
import json
import argparse
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------------
IMG_WIDTH = 256
IMG_HEIGHT = 64
BATCH_SIZE = 32
EPOCHS_DEFAULT = 60
AUTOTUNE = tf.data.AUTOTUNE

# Character set the network is allowed to output. Extend as needed for the
# target language (this covers upper/lower case English letters, digits
# and common punctuation found in IAM word transcriptions).
CHARS = list(
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

char_to_num = layers.StringLookup(vocabulary=CHARS, mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


# ==========================================================================
# 1. DATA INGESTION
# ==========================================================================
class HTRDataset:
    """
    Loads a word-level handwriting dataset described by a simple
    tab-separated label file:

        images/a01-000u-00-00.png    A
        images/a01-000u-00-01.png    MOVE
        ...

    This mirrors how the IAM Handwriting Database's `words.txt` can be
    pre-processed. Corrupted / unreadable images are skipped and logged,
    which is essential for the IAM set (a small fraction of entries are
    flagged as segmentation errors).
    """

    def __init__(self, data_dir, labels_file="labels.txt"):
        self.data_dir = data_dir
        self.labels_path = os.path.join(data_dir, labels_file)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        skipped = 0
        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    skipped += 1
                    continue
                img_path, label = parts
                full_path = os.path.join(self.data_dir, img_path)
                if not os.path.isfile(full_path):
                    skipped += 1
                    continue
                # Keep only characters the model vocabulary supports
                if any(ch not in CHARS for ch in label):
                    skipped += 1
                    continue
                samples.append((full_path, label))
        print(f"[HTRDataset] loaded {len(samples)} samples, skipped {skipped}")
        return samples

    def split(self, val_frac=0.1, test_frac=0.05, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.samples))
        n = len(idx)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val_idx = idx[:n_val]
        test_idx = idx[n_val:n_val + n_test]
        train_idx = idx[n_val + n_test:]
        pick = lambda ids: [self.samples[i] for i in ids]
        return pick(train_idx), pick(val_idx), pick(test_idx)


# ==========================================================================
# 2. PRE-PROCESSING
# ==========================================================================
def preprocess_image(image_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """
    Reads a handwriting image and prepares it for the CRNN:
      - grayscale
      - adaptive thresholding to normalise ink/paper contrast
      - aspect-ratio preserving resize + right/bottom padding
      - scale to [0, 1] and transpose so time-steps run along width
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    h, w = img.shape
    scale = img_height / h
    new_w = min(int(w * scale), img_width)
    img = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)

    padded = np.ones((img_height, img_width), dtype=np.uint8) * 255
    padded[:, :new_w] = img

    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)      # (H, W, 1)
    padded = np.transpose(padded, (1, 0, 2))       # (W, H, 1) -> time major
    return padded


def augment(image):
    """Light augmentation used only during training: small rotation, shear,
    and random noise, which materially improves generalisation on
    handwriting because stroke slant/pressure varies a lot per writer."""
    if tf.random.uniform([]) < 0.5:
        image = tf.image.random_brightness(image, 0.15)
    if tf.random.uniform([]) < 0.3:
        noise = tf.random.normal(tf.shape(image), stddev=0.03)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image


def encode_label(label, max_len):
    label = char_to_num(tf.strings.unicode_split(label, "UTF-8"))
    pad_len = max_len - tf.shape(label)[0]
    label = tf.pad(label, [[0, pad_len]], constant_values=0)
    return label


def build_tf_dataset(samples, max_label_len, training=False):
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    def _load(path, label):
        img = tf.numpy_function(
            lambda p: preprocess_image(p.decode("utf-8")), [path], tf.float32
        )
        img.set_shape([IMG_WIDTH, IMG_HEIGHT, 1])
        if training:
            img = augment(img)
        enc_label = encode_label(label, max_label_len)
        return {"image": img, "label": enc_label}

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(samples))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# ==========================================================================
# 3. MODEL ARCHITECTURE — CRNN (CNN feature extractor + BiLSTM sequence model)
# ==========================================================================
class CTCLayer(layers.Layer):
    """Computes CTC loss internally so it can be attached straight onto the
    training graph (add_loss pattern) — the standard Keras-OCR approach."""

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def build_crnn_model(img_width=IMG_WIDTH, img_height=IMG_HEIGHT, max_label_len=32):
    """
    Architecture:
        Conv(32) -> Conv(64) -> MaxPool -> Conv(128) -> Conv(128) -> MaxPool
        -> reshape to sequence -> Dense(64) -> BiLSTM(128) x2 -> Dense(vocab+1, softmax)
        -> CTC loss

    This is the same family of architecture used in production HTR systems
    (Tesseract LSTM engine, Google Keras-OCR, ICDAR winning CRNNs): a CNN
    extracts local stroke features, and a bidirectional LSTM models the
    left-to-right / right-to-left sequential dependency between characters,
    with CTC removing the need for pre-segmented characters.
    """
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image")
    labels = layers.Input(shape=(max_label_len,), name="label")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 1))(x)   # preserve horizontal (time) resolution
    x = layers.Dropout(0.3)(x)

    new_shape = (img_width // 4, (img_height // 8) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    vocab_size = char_to_num.vocabulary_size()
    x = layers.Dense(vocab_size + 1, activation="softmax", name="dense_out")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.Model(inputs=[input_img, labels], outputs=output, name="handwritten_ocr_crnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    return model


def build_inference_model(training_model):
    """Strips the CTC/label branch so only image -> softmax remains, for
    fast deployment inference."""
    image_input = training_model.get_layer("image").input
    dense_out = training_model.get_layer("dense_out").output
    return keras.Model(image_input, dense_out, name="handwritten_ocr_inference")


# ==========================================================================
# 4. CTC DECODING
# ==========================================================================
def ctc_greedy_decode(pred, max_label_len):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    results = results[:, :max_label_len]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# ==========================================================================
# 5. TRAINING
# ==========================================================================
def train(data_dir, epochs=EPOCHS_DEFAULT, out_dir="checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    ds = HTRDataset(data_dir)
    train_s, val_s, test_s = ds.split()
    max_label_len = max(len(lbl) for _, lbl in ds.samples)

    train_ds = build_tf_dataset(train_s, max_label_len, training=True)
    val_ds = build_tf_dataset(val_s, max_label_len, training=False)

    model = build_crnn_model(max_label_len=max_label_len)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best_model.h5"),
            save_best_only=True, monitor="val_loss", mode="min"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.CSVLogger(os.path.join(out_dir, "training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    with open(os.path.join(out_dir, "max_label_len.json"), "w") as f:
        json.dump({"max_label_len": max_label_len}, f)

    evaluate(data_dir, weights=os.path.join(out_dir, "best_model.h5"),
              max_label_len=max_label_len, test_samples=test_s)
    return history


# ==========================================================================
# 6. EVALUATION — Character Error Rate (CER) and Word Error Rate (WER)
# ==========================================================================
def _levenshtein(a, b):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def evaluate(data_dir, weights, max_label_len=None, test_samples=None):
    if max_label_len is None:
        with open(os.path.join(os.path.dirname(weights), "max_label_len.json")) as f:
            max_label_len = json.load(f)["max_label_len"]

    if test_samples is None:
        ds = HTRDataset(data_dir)
        _, _, test_samples = ds.split()

    training_model = build_crnn_model(max_label_len=max_label_len)
    training_model.load_weights(weights)
    inf_model = build_inference_model(training_model)

    total_chars, total_char_errors = 0, 0
    total_words, total_word_errors = 0, 0

    for path, true_label in test_samples:
        img = preprocess_image(path)
        img = np.expand_dims(img, axis=0)
        pred = inf_model.predict(img, verbose=0)
        pred_text = ctc_greedy_decode(pred, max_label_len)[0]

        total_chars += len(true_label)
        total_char_errors += _levenshtein(true_label, pred_text)
        total_words += 1
        total_word_errors += int(pred_text != true_label)

    cer = total_char_errors / max(total_chars, 1)
    wer = total_word_errors / max(total_words, 1)
    accuracy = 1 - wer
    print(f"[Evaluate] N={len(test_samples)}  CER={cer:.4f}  WER={wer:.4f}  "
          f"Word-level accuracy={accuracy * 100:.2f}%")
    return {"cer": cer, "wer": wer, "accuracy": accuracy}


# ==========================================================================
# 7. SINGLE IMAGE / DEPLOYMENT INFERENCE
# ==========================================================================
def predict_single_image(image_path, weights, max_label_len=32):
    training_model = build_crnn_model(max_label_len=max_label_len)
    training_model.load_weights(weights)
    inf_model = build_inference_model(training_model)

    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    pred = inf_model.predict(img, verbose=0)
    text = ctc_greedy_decode(pred, max_label_len)[0]
    print(f"Predicted text: {text}")
    return text


# ==========================================================================
# CLI ENTRY POINT
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Handwritten OCR (CRNN + CTC)")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.h5")
    parser.add_argument("--image", type=str, help="Path to a single word image (predict mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, epochs=args.epochs)
    elif args.mode == "evaluate":
        evaluate(args.data_dir, weights=args.weights)
    elif args.mode == "predict":
        if not args.image:
            raise ValueError("--image is required for predict mode")
        predict_single_image(args.image, weights=args.weights)


if __name__ == "__main__":
    main()
