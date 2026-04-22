"""SQLAlchemy models for Duck Egg Fertility Detection."""

from datetime import datetime, timezone
from web.api.extensions import db


class PredictionRecord(db.Model):
    __tablename__ = "prediction_history"

    id                  = db.Column(db.String(36), primary_key=True)
    original_filename   = db.Column(db.String(255))
    filename            = db.Column(db.String(255))
    prediction          = db.Column(db.String(50), nullable=False, index=True)
    confidence          = db.Column(db.Float)
    cluster_id          = db.Column(db.Integer)
    cluster_probability = db.Column(db.Float)
    cluster_purity      = db.Column(db.Float)
    label_scores        = db.Column(db.JSON)
    distances           = db.Column(db.JSON)
    feature_count       = db.Column(db.Integer)
    preprocessed_shape  = db.Column(db.JSON)
    timestamp           = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def to_dict(self) -> dict:
        return {
            "id":                   self.id,
            "original_filename":    self.original_filename,
            "filename":             self.filename,
            "prediction":           self.prediction,
            "confidence":           self.confidence,
            "cluster_id":           self.cluster_id,
            "cluster_probability":  self.cluster_probability,
            "cluster_purity":       self.cluster_purity,
            "label_scores":         self.label_scores or {},
            "distances":            self.distances or [],
            "feature_count":        self.feature_count,
            "preprocessed_shape":   self.preprocessed_shape,
            "timestamp":            self.timestamp.isoformat() if self.timestamp else None,
        }
