"""FastAPI backend for the instrument classifier.

Run from the project root:
    uvicorn app.main:app --reload --port 8000
"""
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_processor import AudioProcessor  # noqa: E402
from config_loader import config  # noqa: E402

MODEL_PATH = PROJECT_ROOT / config.get('output.model_dir', 'models') / 'best_model.joblib'
STATIC_DIR = Path(__file__).parent / 'static'
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}
MAX_FILE_SIZE = 25 * 1024 * 1024

state = {"model": None, "classes": [], "model_name": None, "processor": None}


def _load_model():
    artifact = joblib.load(MODEL_PATH)
    state["model"] = artifact["model"]
    state["classes"] = list(artifact["classes"])
    state["model_name"] = artifact["model_name"]
    state["processor"] = AudioProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        _load_model()
    yield


app = FastAPI(title="Instrument Detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_name": state["model_name"],
        "classes": state["classes"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Train one first: python src/main.py --train",
        )

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 25 MB)")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        features = state["processor"].extract_features_from_file(tmp_path).reshape(1, -1)
        pred_idx = int(state["model"].predict(features)[0])
        probas = state["model"].predict_proba(features)[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    ranked = np.argsort(probas)[::-1]
    return {
        "instrument": state["classes"][pred_idx],
        "confidence": round(float(probas[pred_idx]), 4),
        "model": state["model_name"],
        "top_predictions": [
            {"instrument": state["classes"][i], "probability": round(float(probas[i]), 4)}
            for i in ranked[:3]
        ],
        "all_probabilities": {
            state["classes"][i]: round(float(probas[i]), 4) for i in ranked
        },
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / 'index.html').read_text(encoding='utf-8')
