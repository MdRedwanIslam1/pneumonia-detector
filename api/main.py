"""FastAPI service for pneumonia-classification model inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field
from tensorflow import keras

from src.predict import (
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    InvalidImageError,
    create_gradcam_data_url,
    load_prediction_model,
    predict_image,
    prepare_image_bytes,
)


DISCLAIMER = (
    "Educational project only. This is not a certified diagnostic tool and "
    "must not be used for medical decisions."
)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class PredictionResponse(BaseModel):
    """The JSON fields returned by POST /predict."""

    predicted_class: Literal["NORMAL", "PNEUMONIA"]
    confidence: float = Field(ge=0.0, le=1.0)
    pneumonia_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(gt=0.0, lt=1.0)
    gradcam_overlay: str | None = Field(
        default=None,
        description="Optional PNG data URL when include_gradcam=true.",
    )
    disclaimer: str


class HealthResponse(BaseModel):
    """Small response used to check whether the API and model are ready."""

    status: Literal["ok"]
    model: str
    disclaimer: str


def configured_model_path() -> Path:
    """Allow deployment to override the local checkpoint with MODEL_PATH."""
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()


def configured_threshold() -> float:
    """Read the decision threshold once and reject invalid configuration."""
    try:
        threshold = float(os.getenv("PREDICTION_THRESHOLD", str(DEFAULT_THRESHOLD)))
    except ValueError as error:
        raise RuntimeError("PREDICTION_THRESHOLD must be a number.") from error
    if not 0.0 < threshold < 1.0:
        raise RuntimeError("PREDICTION_THRESHOLD must be between 0 and 1.")
    return threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and release its reference on shutdown."""
    model_path = configured_model_path()
    app.state.model = load_prediction_model(model_path)
    app.state.model_path = model_path
    app.state.threshold = configured_threshold()
    yield
    app.state.model = None


app = FastAPI(
    title="Pneumonia Detection API",
    version="1.0.0",
    description=DISCLAIMER,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Confirm that the process is running and the model was loaded."""
    return HealthResponse(
        status="ok",
        model=request.app.state.model_path.name,
        disclaimer=DISCLAIMER,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    file: UploadFile = File(description="A JPEG or PNG chest X-ray image."),
    include_gradcam: bool = Query(
        default=False,
        description="Include a base64-encoded Grad-CAM PNG overlay.",
    ),
) -> PredictionResponse:
    """Validate one upload, run inference, and return a JSON prediction."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG or PNG image.",
        )

    try:
        image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        file.file.close()

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The image is larger than the 10 MB upload limit.",
        )

    try:
        image_batch = prepare_image_bytes(image_bytes)
    except InvalidImageError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    model: keras.Model = request.app.state.model
    result = predict_image(model, image_batch, request.app.state.threshold)
    gradcam_overlay = (
        create_gradcam_data_url(model, image_batch) if include_gradcam else None
    )

    return PredictionResponse(
        **result,
        gradcam_overlay=gradcam_overlay,
        disclaimer=DISCLAIMER,
    )
