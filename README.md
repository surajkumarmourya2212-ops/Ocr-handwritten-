# Handwritten OCR — Deployment Package

Two front-ends are included, sharing the same model code (`ocr_utils.py`).
Pick the one that matches where you're deploying.

| Front-end | File | Deploy target |
|---|---|---|
| Flask | `app.py` | Render, Railway, PythonAnywhere, Fly.io, Docker, your own VPS |
| Streamlit | `streamlit_app.py` | **Streamlit Community Cloud** (share.streamlit.io) |

⚠️ Streamlit Cloud runs `streamlit run <file>.py` — it cannot run a Flask
app. If you deploy this repo there, set the **main file path to
`streamlit_app.py`**, not `app.py`.

## What's inside
```
app.py                    Flask web app (serves the UI + /predict API)
streamlit_app.py          Streamlit web app — use this one for Streamlit Cloud
ocr_utils.py              Shared model loading + preprocessing (used by both)
train_model.py            Trains and saves models/digit_model.pkl
handwritten_ocr.py         Full CRNN + CTC model definition (from the project report)
requirements.txt          Python dependencies (covers both front-ends)
packages.txt               System packages for Streamlit Cloud (installs tesseract-ocr)
models/digit_model.pkl     Trained, working model (real handwritten-digit data, 96.7% test accuracy)
templates/index.html       Flask UI (draw a digit, or upload a word/sentence image)
```

## Quick start — Flask (local or Render/Railway/Docker)
```bash
pip install -r requirements.txt
python train_model.py      # already run once — re-run any time to retrain
python app.py               # open http://127.0.0.1:5000
```

## Quick start — Streamlit (local or Streamlit Community Cloud)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploying to Streamlit Community Cloud
1. Push this whole folder to a **public GitHub repo** (private repos need a paid/linked plan).
2. Go to https://share.streamlit.io → **New app**.
3. Pick your repo/branch, and set **Main file path** to `streamlit_app.py`.
4. Streamlit Cloud automatically installs `requirements.txt` (Python packages)
   and `packages.txt` (the `tesseract-ocr` system binary, needed by
   `pytesseract` for the text-mode fallback).
5. Deploy — first build takes a few minutes.

Notes specific to Streamlit Cloud:
- `models/digit_model.pkl` (~420 KB) is small enough to commit directly to
  the repo — no Git LFS needed.
- If you later add `models/crnn_model.h5` (a trained TensorFlow model),
  check its file size — Streamlit Cloud/GitHub free tier struggles with
  files over ~100 MB. Use Git LFS or an external download step if so.
- The drawable canvas needs `streamlit-drawable-canvas`, already listed in
  `requirements.txt`.

## How it works
- **Digit mode** (works immediately, no setup): draw a digit 0-9 on the canvas.
  It's classified by `models/digit_model.pkl`, a scikit-learn MLP trained on
  1,797 real handwritten digit scans (`sklearn.datasets.load_digits`). This
  achieved **96.67% test accuracy** when trained (see `train_model.py` output).
- **Text mode** (words/sentences): upload an image.
  - If you've trained the full `handwritten_ocr.py` CRNN model (on a GPU
    machine, with the IAM dataset — see the project report) and placed the
    weights at `models/crnn_model.h5`, the app auto-detects and uses it.
  - Otherwise it falls back to **Tesseract OCR** (`pytesseract`), which is
    already installed and works out of the box, so the demo is never broken
    even before the CRNN is trained.

## Production deployment (Flask path)
```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```
Or containerize with a simple Dockerfile:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
```

## Upgrading to full word-level recognition
1. Train `handwritten_ocr.py` on the IAM Handwriting Database (GPU recommended):
   `python handwritten_ocr.py --mode train --data_dir ./data --epochs 60`
2. Copy the resulting `checkpoints/best_model.h5` to `models/crnn_model.h5`.
3. `pip install tensorflow` (uncomment it in requirements.txt).
4. Restart `app.py` — it will auto-detect the CRNN weights and switch
   "text" mode from the Tesseract fallback to your trained deep model.
