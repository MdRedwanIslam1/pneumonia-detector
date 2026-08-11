"""Streamlit interface for the Pneumonia Detection FastAPI service."""

from __future__ import annotations

import base64
import hashlib
import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


DISCLAIMER = (
    "Educational/portfolio project only. This is not a certified diagnostic "
    "tool and must not be used for real medical decisions."
)
DEFAULT_API_URL = "http://127.0.0.1:8000"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class APIRequestError(RuntimeError):
    """Raised when the inference API cannot return a usable prediction."""


def apply_styles() -> None:
    """Apply a restrained clinical-workspace visual style."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f6f8;
            color: #17212b;
        }
        [data-testid="stHeader"] {
            background: rgba(244, 246, 248, 0.96);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.75rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, p, label, button {
            letter-spacing: 0;
        }
        h1 {
            color: #17212b;
            font-size: 2rem !important;
            line-height: 1.2 !important;
            margin: 0 !important;
        }
        .product-subtitle {
            color: #52606d;
            font-size: 0.95rem;
            margin-top: 0.3rem;
        }
        .api-status {
            text-align: right;
            font-size: 0.88rem;
            color: #52606d;
            padding-top: 0.55rem;
        }
        .status-dot-online, .status-dot-offline {
            display: inline-block;
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 50%;
            margin-right: 0.4rem;
        }
        .status-dot-online { background: #18794e; }
        .status-dot-offline { background: #b42318; }
        .medical-warning {
            background: #fff4ed;
            border-left: 4px solid #b54708;
            color: #7a2e0e;
            margin: 1.2rem 0 1.5rem;
            padding: 0.8rem 1rem;
            font-size: 0.92rem;
        }
        .section-label {
            color: #52606d;
            font-size: 0.8rem;
            font-weight: 700;
            margin: 0 0 0.65rem;
            text-transform: uppercase;
        }
        .result-banner {
            border-left: 5px solid;
            padding: 0.8rem 1rem;
            margin-bottom: 1.25rem;
        }
        .result-normal {
            background: #edf8f2;
            border-color: #18794e;
            color: #0f5f3d;
        }
        .result-pneumonia {
            background: #fff1f0;
            border-color: #b42318;
            color: #8f1d18;
        }
        .result-name {
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.25;
        }
        .result-caption {
            font-size: 0.85rem;
            margin-top: 0.15rem;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border-color: #aeb8c2;
            border-radius: 6px;
        }
        div.stButton > button {
            border-radius: 6px;
            font-weight: 650;
            min-height: 2.75rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border-top: 2px solid #d7dde3;
            padding: 0.8rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] p {
            color: #52606d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=5, show_spinner=False)
def check_api(api_url: str) -> tuple[bool, str]:
    """Return whether FastAPI is reachable and which model it loaded."""
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        response.raise_for_status()
        payload = response.json()
        return payload.get("status") == "ok", str(payload.get("model", "model"))
    except (requests.RequestException, ValueError):
        return False, "Unavailable"


def validate_uploaded_image(image_bytes: bytes) -> Image.Image:
    """Check the upload and return a display-ready RGB image."""
    if not image_bytes:
        raise ValueError("The selected file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The selected image exceeds the 10 MB limit.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The selected file is not a readable image.") from error

    if image.format not in {"JPEG", "PNG"}:
        raise ValueError("Only JPEG and PNG images are supported.")
    return image.convert("RGB")


def request_prediction(
    api_url: str,
    filename: str,
    content_type: str,
    image_bytes: bytes,
    include_gradcam: bool,
) -> dict[str, Any]:
    """Send one multipart image upload to the FastAPI prediction endpoint."""
    try:
        response = requests.post(
            f"{api_url}/predict",
            params={"include_gradcam": str(include_gradcam).lower()},
            files={"file": (filename, image_bytes, content_type)},
            timeout=120,
        )
    except requests.ConnectionError as error:
        raise APIRequestError("The inference API is not reachable.") from error
    except requests.Timeout as error:
        raise APIRequestError("The prediction request timed out.") from error
    except requests.RequestException as error:
        raise APIRequestError("The prediction request could not be completed.") from error

    try:
        payload = response.json()
    except ValueError as error:
        raise APIRequestError("The API returned an unreadable response.") from error

    if not response.ok:
        detail = payload.get("detail", f"API request failed with status {response.status_code}.")
        raise APIRequestError(str(detail))

    required_fields = {
        "predicted_class",
        "confidence",
        "pneumonia_probability",
        "threshold",
        "disclaimer",
    }
    if not required_fields.issubset(payload):
        raise APIRequestError("The API response is missing required prediction fields.")
    return payload


def decode_gradcam(data_url: str) -> Image.Image:
    """Turn the API's PNG data URL back into a displayable image."""
    expected_prefix = "data:image/png;base64,"
    if not data_url.startswith(expected_prefix):
        raise ValueError("The Grad-CAM response is not a PNG data URL.")

    try:
        image_bytes = base64.b64decode(data_url[len(expected_prefix) :], validate=True)
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except (ValueError, UnidentifiedImageError, OSError) as error:
        raise ValueError("The Grad-CAM image could not be decoded.") from error


def render_prediction(result: dict[str, Any]) -> None:
    """Render the prediction metrics and optional explanatory overlay."""
    predicted_class = str(result["predicted_class"])
    confidence = float(result["confidence"])
    pneumonia_probability = float(result["pneumonia_probability"])
    threshold = float(result["threshold"])

    banner_class = (
        "result-pneumonia" if predicted_class == "PNEUMONIA" else "result-normal"
    )
    st.markdown(
        f"""
        <div class="result-banner {banner_class}">
            <div class="result-name">{predicted_class}</div>
            <div class="result-caption">Model classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Confidence", f"{confidence:.1%}")
    metric_columns[1].metric("Pneumonia score", f"{pneumonia_probability:.1%}")
    metric_columns[2].metric("Threshold", f"{threshold:.2f}")

    st.progress(
        max(0, min(100, int(round(pneumonia_probability * 100)))),
        text="Pneumonia probability",
    )

    gradcam_data_url = result.get("gradcam_overlay")
    if gradcam_data_url:
        st.markdown("<div class='section-label'>Grad-CAM overlay</div>", unsafe_allow_html=True)
        try:
            overlay = decode_gradcam(str(gradcam_data_url))
            st.image(overlay, width="stretch")
        except ValueError as error:
            st.error(str(error))
        st.caption("Grad-CAM shows model influence, not clinical evidence.")


def main() -> None:
    """Render the complete Streamlit application."""
    st.set_page_config(
        page_title="Pneumonia Detection",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()

    api_url = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")
    api_online, model_name = check_api(api_url)

    title_column, status_column = st.columns([3, 2])
    with title_column:
        st.title("Pneumonia Detection")
        st.markdown(
            "<div class='product-subtitle'>Chest X-ray screening interface</div>",
            unsafe_allow_html=True,
        )
    with status_column:
        status_class = "status-dot-online" if api_online else "status-dot-offline"
        status_text = f"API connected: {model_name}" if api_online else "API unavailable"
        st.markdown(
            f"<div class='api-status'><span class='{status_class}'></span>{status_text}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<div class='medical-warning'><strong>Notice:</strong> {DISCLAIMER}</div>", unsafe_allow_html=True)

    source_column, result_column = st.columns([1, 1], gap="large")

    with source_column:
        st.markdown("<div class='section-label'>Source X-ray</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Chest X-ray image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image_bytes: bytes | None = None
        image_is_valid = False
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            upload_id = hashlib.sha256(image_bytes).hexdigest()
            if st.session_state.get("upload_id") != upload_id:
                st.session_state.upload_id = upload_id
                st.session_state.pop("prediction", None)

            try:
                display_image = validate_uploaded_image(image_bytes)
                st.image(display_image, width="stretch")
                image_is_valid = True
            except ValueError as error:
                st.error(str(error))

        include_gradcam = st.toggle("Generate Grad-CAM overlay", value=True)
        analyze_clicked = st.button(
            "Analyze X-ray",
            type="primary",
            width="stretch",
            disabled=not (api_online and image_is_valid),
        )

        if analyze_clicked and uploaded_file is not None and image_bytes is not None:
            with st.spinner("Analyzing image..."):
                try:
                    st.session_state.prediction = request_prediction(
                        api_url=api_url,
                        filename=uploaded_file.name,
                        content_type=uploaded_file.type or "image/jpeg",
                        image_bytes=image_bytes,
                        include_gradcam=include_gradcam,
                    )
                except APIRequestError as error:
                    st.error(str(error))

        if not api_online:
            st.error(f"Inference API unavailable at {api_url}")

    with result_column:
        st.markdown("<div class='section-label'>Analysis</div>", unsafe_allow_html=True)
        prediction = st.session_state.get("prediction")
        if prediction is None:
            st.info("Awaiting analysis")
        else:
            render_prediction(prediction)

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
