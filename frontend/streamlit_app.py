"""Streamlit interface for the Pneumonia Detection FastAPI service."""

from __future__ import annotations

import base64
import hashlib
import io
import os
from html import escape
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
    """Apply the clinical-workspace visual system."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #17212b;
            --muted: #607080;
            --line: #d8e0e7;
            --surface: #ffffff;
            --canvas: #eef2f5;
            --nav: #101820;
            --normal: #18794e;
            --normal-soft: #eaf7f0;
            --alert: #b42318;
            --alert-soft: #fff0ef;
            --amber: #b54708;
            --amber-soft: #fff7ed;
        }
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(238, 242, 245, 0.96);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 4.25rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, p, label, button {
            letter-spacing: 0;
        }
        .app-header {
            align-items: center;
            background: var(--nav);
            border-bottom: 3px solid #cf4438;
            color: #ffffff;
            display: flex;
            justify-content: space-between;
            min-height: 76px;
            padding: 0.9rem 1.1rem;
        }
        .brand-lockup {
            align-items: center;
            display: flex;
            gap: 0.8rem;
        }
        .brand-mark {
            align-items: center;
            background: #ffffff;
            border-radius: 4px;
            color: var(--nav);
            display: flex;
            font-size: 0.78rem;
            font-weight: 800;
            height: 38px;
            justify-content: center;
            width: 38px;
        }
        .brand-title {
            font-size: 1.3rem;
            font-weight: 750;
            line-height: 1.15;
        }
        .brand-subtitle {
            color: #b9c5cf;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }
        .service-stack {
            align-items: flex-end;
            display: flex;
            flex-direction: column;
            gap: 0.28rem;
            min-width: 0;
        }
        .service-state {
            align-items: center;
            background: #1d2a35;
            border: 1px solid #354756;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 650;
            gap: 0.45rem;
            padding: 0.35rem 0.65rem;
        }
        .service-model {
            color: #aebdca;
            font-size: 0.7rem;
            max-width: 280px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .status-dot-online, .status-dot-offline {
            border-radius: 50%;
            display: inline-block;
            height: 0.5rem;
            width: 0.5rem;
        }
        .status-dot-online { background: #48c78e; }
        .status-dot-offline { background: #ef6b63; }
        .spec-strip {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 0;
            color: var(--muted);
            display: flex;
            font-size: 0.76rem;
            gap: 1.25rem;
            min-height: 38px;
            padding: 0.45rem 1.1rem;
        }
        .spec-strip strong {
            color: var(--ink);
            font-weight: 700;
            margin-left: 0.28rem;
        }
        .medical-warning {
            background: var(--amber-soft);
            border: 1px solid #f0d3b3;
            border-left: 4px solid var(--amber);
            color: #7a2e0e;
            font-size: 0.82rem;
            margin: 0.9rem 0 1.35rem;
            padding: 0.72rem 0.9rem;
        }
        .panel-heading {
            align-items: flex-end;
            border-bottom: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.9rem;
            padding-bottom: 0.65rem;
        }
        .panel-eyebrow {
            color: var(--alert);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .panel-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 750;
            margin-top: 0.14rem;
        }
        .panel-meta {
            color: var(--muted);
            font-size: 0.72rem;
        }
        .result-banner {
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--line);
            border-top: 4px solid;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.9rem;
            min-height: 104px;
            padding: 1rem 1.1rem;
        }
        .result-normal { border-top-color: var(--normal); }
        .result-pneumonia { border-top-color: var(--alert); }
        .result-kicker {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .result-name {
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.25;
            margin-top: 0.2rem;
        }
        .result-normal .result-name { color: var(--normal); }
        .result-pneumonia .result-name { color: var(--alert); }
        .confidence-block { text-align: right; }
        .confidence-value {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1;
        }
        .confidence-label {
            color: var(--muted);
            font-size: 0.7rem;
            margin-top: 0.35rem;
        }
        .metric-grid {
            display: grid;
            gap: 0.65rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-bottom: 0.9rem;
        }
        .metric-tile {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 6px;
            min-height: 76px;
            padding: 0.75rem 0.8rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 650;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.12rem;
            font-weight: 760;
            margin-top: 0.25rem;
        }
        .probability-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            margin-bottom: 1rem;
            padding: 0.85rem 0.9rem 0.75rem;
        }
        .probability-heading {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.65rem;
        }
        .probability-heading span:first-child {
            color: var(--ink);
            font-size: 0.76rem;
            font-weight: 700;
        }
        .probability-heading span:last-child {
            color: var(--muted);
            font-size: 0.7rem;
        }
        .probability-track {
            background: #e6ebef;
            height: 10px;
            overflow: visible;
            position: relative;
        }
        .probability-fill { height: 10px; }
        .probability-fill-normal { background: var(--normal); }
        .probability-fill-alert { background: var(--alert); }
        .threshold-marker {
            background: var(--ink);
            height: 18px;
            position: absolute;
            top: -4px;
            transform: translateX(-1px);
            width: 2px;
        }
        .probability-scale {
            color: var(--muted);
            display: flex;
            font-size: 0.64rem;
            justify-content: space-between;
            margin-top: 0.45rem;
        }
        .analysis-empty {
            align-items: center;
            background: #f8fafb;
            border: 1px dashed #b9c5cf;
            color: var(--muted);
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 360px;
            padding: 2rem;
            text-align: center;
        }
        .empty-mark {
            align-items: center;
            border: 2px solid #9aabb9;
            border-radius: 50%;
            color: #607080;
            display: flex;
            font-size: 0.76rem;
            font-weight: 800;
            height: 48px;
            justify-content: center;
            margin-bottom: 0.85rem;
            width: 48px;
        }
        .empty-title {
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 700;
        }
        .empty-caption {
            font-size: 0.75rem;
            margin-top: 0.28rem;
        }
        .gradcam-heading {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin: 1.1rem 0 0.65rem;
        }
        .gradcam-title {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 750;
        }
        .gradcam-badge {
            background: #e7eef4;
            border-radius: 999px;
            color: #40566a;
            font-size: 0.64rem;
            font-weight: 700;
            padding: 0.25rem 0.5rem;
        }
        .explainability-note {
            background: #eef5f8;
            border-left: 3px solid #39728f;
            color: #34556a;
            font-size: 0.72rem;
            margin-top: 0.55rem;
            padding: 0.6rem 0.7rem;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border: 1px dashed #9caebb;
            border-radius: 6px;
            min-height: 132px;
            padding: 1rem;
        }
        div[data-testid="stFileUploaderDropzone"]:hover {
            background: #fffafa;
            border-color: var(--alert);
        }
        [data-testid="stImage"] img {
            background: #101820;
            border: 1px solid #cbd5dd;
            border-radius: 6px;
            max-height: 520px;
            object-fit: contain;
            width: 100%;
        }
        [data-testid="stToggle"] {
            background: #f8fafb;
            border: 1px solid var(--line);
            border-radius: 6px;
            margin-top: 0.65rem;
            padding: 0.55rem 0.7rem;
        }
        div.stButton > button {
            border-radius: 6px;
            font-weight: 700;
            min-height: 2.9rem;
        }
        div.stButton > button[kind="primary"] {
            background: var(--alert);
            border-color: var(--alert);
        }
        div.stButton > button[kind="primary"]:hover {
            background: #941f16;
            border-color: #941f16;
        }
        .validation-strip {
            align-items: center;
            border-top: 1px solid var(--line);
            color: var(--muted);
            display: flex;
            font-size: 0.7rem;
            gap: 1.25rem;
            justify-content: center;
            margin-top: 1.5rem;
            padding: 0.9rem 0 0.5rem;
        }
        .validation-strip strong {
            color: var(--ink);
            margin-left: 0.2rem;
        }
        .footer-disclaimer {
            color: #6a7885;
            font-size: 0.7rem;
            margin-top: 0.25rem;
            text-align: center;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
                padding-top: 4.25rem;
            }
            .app-header {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.85rem;
                padding: 0.85rem;
            }
            .service-stack {
                align-items: flex-start;
                width: 100%;
            }
            .service-model { max-width: 100%; }
            .spec-strip {
                align-items: flex-start;
                flex-wrap: wrap;
                gap: 0.35rem 0.9rem;
                padding: 0.55rem 0.85rem;
            }
            .metric-grid {
                gap: 0.5rem;
                grid-template-columns: 1fr;
            }
            .metric-tile { min-height: 64px; }
            .result-banner { min-height: 94px; }
            .analysis-empty { min-height: 220px; }
            .validation-strip {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.3rem;
            }
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
    """Render the prediction, threshold context, and optional Grad-CAM."""
    predicted_class = str(result["predicted_class"])
    confidence = float(result["confidence"])
    pneumonia_probability = float(result["pneumonia_probability"])
    threshold = float(result["threshold"])
    score_percent = max(0.0, min(100.0, pneumonia_probability * 100.0))
    threshold_percent = max(0.0, min(100.0, threshold * 100.0))

    banner_class = (
        "result-pneumonia" if predicted_class == "PNEUMONIA" else "result-normal"
    )
    fill_class = (
        "probability-fill-alert"
        if pneumonia_probability >= threshold
        else "probability-fill-normal"
    )
    st.markdown(
        f"""
        <div class="result-banner {banner_class}">
            <div>
                <div class="result-kicker">Model classification</div>
                <div class="result-name">{escape(predicted_class)}</div>
            </div>
            <div class="confidence-block">
                <div class="confidence-value">{confidence:.1%}</div>
                <div class="confidence-label">Prediction confidence</div>
            </div>
        </div>
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-label">Pneumonia score</div>
                <div class="metric-value">{pneumonia_probability:.1%}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Decision threshold</div>
                <div class="metric-value">{threshold:.2f}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Output classes</div>
                <div class="metric-value">2</div>
            </div>
        </div>
        <div class="probability-panel">
            <div class="probability-heading">
                <span>Pneumonia probability</span>
                <span>Threshold {threshold:.0%}</span>
            </div>
            <div class="probability-track">
                <div class="probability-fill {fill_class}" style="width: {score_percent:.2f}%"></div>
                <div class="threshold-marker" style="left: {threshold_percent:.2f}%"></div>
            </div>
            <div class="probability-scale">
                <span>0% - Normal</span>
                <span>100% - Pneumonia</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gradcam_data_url = result.get("gradcam_overlay")
    if gradcam_data_url:
        st.markdown(
            """
            <div class="gradcam-heading">
                <div class="gradcam-title">Model attention map</div>
                <div class="gradcam-badge">GRAD-CAM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            overlay = decode_gradcam(str(gradcam_data_url))
            st.image(overlay, width="stretch")
        except ValueError as error:
            st.error(str(error))
        st.markdown(
            """
            <div class="explainability-note">
                Highlighted regions show model influence, not clinical evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header(api_online: bool, model_name: str) -> None:
    """Render product identity, model status, and input specification."""
    status_class = "status-dot-online" if api_online else "status-dot-offline"
    status_text = "Model online" if api_online else "Model offline"
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand-lockup">
                <div class="brand-mark">XR</div>
                <div>
                    <div class="brand-title">Pneumonia Detection</div>
                    <div class="brand-subtitle">Chest X-ray review workspace</div>
                </div>
            </div>
            <div class="service-stack">
                <div class="service-state">
                    <span class="{status_class}"></span>{status_text}
                </div>
                <div class="service-model">{escape(model_name)}</div>
            </div>
        </div>
        <div class="spec-strip">
            <span>Architecture <strong>DenseNet121</strong></span>
            <span>Input <strong>224 x 224 RGB</strong></span>
            <span>Task <strong>Binary classification</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_heading(eyebrow: str, title: str, meta: str) -> None:
    """Render one consistent workspace panel heading."""
    st.markdown(
        f"""
        <div class="panel-heading">
            <div>
                <div class="panel-eyebrow">{escape(eyebrow)}</div>
                <div class="panel-title">{escape(title)}</div>
            </div>
            <div class="panel-meta">{escape(meta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the complete Streamlit application."""
    st.set_page_config(
        page_title="Pneumonia Detection",
        page_icon=":material/monitor_heart:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()

    api_url = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")
    api_online, model_name = check_api(api_url)
    render_header(api_online, model_name)
    st.markdown(
        f"<div class='medical-warning'><strong>Notice:</strong> {escape(DISCLAIMER)}</div>",
        unsafe_allow_html=True,
    )

    source_column, result_column = st.columns([1, 1], gap="large")

    with source_column:
        render_panel_heading("01 / Source", "Chest X-ray", "JPEG or PNG")
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
            icon=":material/analytics:",
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
        render_panel_heading("02 / Analysis", "Model output", "Threshold 0.50")
        prediction = st.session_state.get("prediction")
        if prediction is None:
            st.markdown(
                """
                <div class="analysis-empty">
                    <div class="empty-mark">AI</div>
                    <div class="empty-title">Awaiting analysis</div>
                    <div class="empty-caption">No model output is available</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_prediction(prediction)

    st.markdown(
        f"""
        <div class="validation-strip">
            <span>Held-out test AUC <strong>0.957</strong></span>
            <span>Sensitivity <strong>99.2%</strong></span>
            <span>Test images <strong>624</strong></span>
        </div>
        <div class="footer-disclaimer">{escape(DISCLAIMER)}</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
