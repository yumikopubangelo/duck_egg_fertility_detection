"""Admin API routes for Duck Egg Fertility Detection."""

from datetime import datetime
from flask import Blueprint, jsonify

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/health", methods=["GET"])
def admin_health():
    """Admin health endpoint."""
    return jsonify(
        {
            "status": "ok",
            "service": "admin",
            "timestamp": datetime.utcnow().isoformat(),
        }
    ), 200


@admin_bp.route("/admin/info", methods=["GET"])
def admin_info():
    """Basic admin metadata."""
    return jsonify(
        {
            "service": "duck-egg-fertility-api",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
        }
    ), 200
