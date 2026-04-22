"""Prediction API routes for Duck Egg Fertility Detection."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from src.web.model_manager import get_default_model_manager
from src.web.prediction_service import PredictionService
from web.api.routes.history import add_to_history


prediction_bp = Blueprint("prediction", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
_prediction_service: PredictionService | None = None


def allowed_file(filename: str) -> bool:
    """Check if file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_prediction_service() -> PredictionService:
    """Return the app-local prediction service."""
    global _prediction_service
    if _prediction_service is None:
        model_manager = get_default_model_manager(current_app.config)
        _prediction_service = PredictionService(model_manager=model_manager)
    return _prediction_service


def save_upload(file_storage) -> tuple[str, str, Path]:
    """Persist an uploaded file and return original/saved names plus path."""
    original_filename = secure_filename(file_storage.filename or "upload")
    unique_filename = f"{uuid.uuid4()}_{original_filename}"
    upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "data/uploads"))
    upload_folder.mkdir(parents=True, exist_ok=True)
    filepath = upload_folder / unique_filename
    file_storage.save(str(filepath))
    return original_filename, unique_filename, filepath


def prediction_payload(result, original_filename: str, unique_filename: str) -> dict:
    """Build the JSON payload expected by the frontend."""
    payload = asdict(result)
    payload.update(
        {
            "id": str(uuid.uuid4()),
            "filename": unique_filename,
            "original_filename": original_filename,
            "prediction": result.label,
            "confidence": round(float(result.confidence), 4),
            "cluster_probability": round(float(result.cluster_probability), 4),
            "cluster_purity": round(float(result.cluster_purity), 4),
            "label_scores": {
                label: round(float(score), 4) for label, score in result.label_scores.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return payload


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """Predict fertility for one uploaded egg image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        original_filename, unique_filename, filepath = save_upload(file)
        result = get_prediction_service().predict_file(filepath)
        payload = prediction_payload(result, original_filename, unique_filename)
        add_to_history(payload)
        return jsonify(payload), 200
    except Exception as exc:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


@prediction_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predict fertility for multiple uploaded egg images."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    results = []
    for file in files:
        if not file or not allowed_file(file.filename):
            results.append(
                {
                    "filename": file.filename if file else "",
                    "error": "File type not allowed",
                }
            )
            continue

        try:
            original_filename, unique_filename, filepath = save_upload(file)
            result = get_prediction_service().predict_file(filepath)
            payload = prediction_payload(result, original_filename, unique_filename)
            add_to_history(payload)
            results.append(payload)
        except Exception as exc:
            current_app.logger.exception("Batch prediction failed")
            results.append({"filename": file.filename, "error": str(exc)})

    return jsonify({"results": results}), 200


@prediction_bp.route("/model/info", methods=["GET"])
def model_info():
    """Return currently loaded model metadata."""
    try:
        info = get_prediction_service().model_info()
        info["last_checked"] = datetime.now(timezone.utc).isoformat()
        return jsonify(info), 200
    except Exception as exc:
        current_app.logger.exception("Model info failed")
        return jsonify({"error": str(exc)}), 500
