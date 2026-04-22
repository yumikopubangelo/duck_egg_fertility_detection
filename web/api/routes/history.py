"""History API routes for Duck Egg Fertility Detection."""

from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, jsonify, request
from sqlalchemy import func

from web.api.extensions import db
from web.api.models import PredictionRecord

history_bp = Blueprint("history", __name__)


@history_bp.route("/history", methods=["GET"])
def get_history():
    page     = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    start_date = request.args.get("start_date")
    end_date   = request.args.get("end_date")

    query = PredictionRecord.query.order_by(PredictionRecord.timestamp.desc())

    if start_date:
        try:
            query = query.filter(PredictionRecord.timestamp >= datetime.fromisoformat(start_date))
        except ValueError:
            pass

    if end_date:
        try:
            query = query.filter(PredictionRecord.timestamp <= datetime.fromisoformat(end_date))
        except ValueError:
            pass

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "history":  [r.to_dict() for r in pagination.items],
        "total":    pagination.total,
        "page":     pagination.page,
        "per_page": per_page,
        "pages":    pagination.pages,
    }), 200


@history_bp.route("/history/stats", methods=["GET"])
def get_stats():
    total          = db.session.query(func.count(PredictionRecord.id)).scalar() or 0
    fertile_count  = db.session.query(func.count(PredictionRecord.id)).filter_by(prediction="fertile").scalar() or 0
    infertile_count = db.session.query(func.count(PredictionRecord.id)).filter_by(prediction="infertile").scalar() or 0
    avg_confidence = db.session.query(func.avg(PredictionRecord.confidence)).scalar() or 0.0

    return jsonify({
        "total_predictions":  total,
        "fertile_count":      fertile_count,
        "infertile_count":    infertile_count,
        "average_confidence": round(float(avg_confidence), 4),
        "accuracy_rate":      round(fertile_count / total, 4) if total > 0 else 0,
    }), 200


@history_bp.route("/history/<prediction_id>", methods=["GET"])
def get_prediction(prediction_id: str):
    record = PredictionRecord.query.get(prediction_id)
    if record is None:
        return jsonify({"error": "Prediction not found"}), 404
    return jsonify(record.to_dict()), 200


@history_bp.route("/history/<prediction_id>", methods=["DELETE"])
def delete_prediction(prediction_id: str):
    record = PredictionRecord.query.get(prediction_id)
    if record is None:
        return jsonify({"error": "Prediction not found"}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Prediction deleted"}), 200


def add_to_history(prediction_data: dict[str, Any]) -> dict[str, Any]:
    """Persist a prediction result to the database."""
    ts_raw = prediction_data.get("timestamp")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    record = PredictionRecord(
        id                  = prediction_data.get("id"),
        original_filename   = prediction_data.get("original_filename"),
        filename            = prediction_data.get("filename"),
        prediction          = prediction_data.get("prediction") or prediction_data.get("label", ""),
        confidence          = prediction_data.get("confidence"),
        cluster_id          = prediction_data.get("cluster_id"),
        cluster_probability = prediction_data.get("cluster_probability"),
        cluster_purity      = prediction_data.get("cluster_purity"),
        label_scores        = prediction_data.get("label_scores"),
        distances           = prediction_data.get("distances"),
        feature_count       = prediction_data.get("feature_count"),
        preprocessed_shape  = prediction_data.get("preprocessed_shape"),
        timestamp           = ts,
    )
    db.session.add(record)
    db.session.commit()
    return prediction_data
