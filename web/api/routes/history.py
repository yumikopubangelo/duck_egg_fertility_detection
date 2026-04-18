"""History API routes for Duck Egg Fertility Detection."""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import uuid

history_bp = Blueprint('history', __name__)

# In-memory storage for development (replace with database in production)
_prediction_history = []


@history_bp.route('/history', methods=['GET'])
def get_history():
    """Get prediction history."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Filter by date if provided
    filtered_history = _prediction_history.copy()
    
    if start_date:
        try:
            start = datetime.fromisoformat(start_date)
            filtered_history = [h for h in filtered_history 
                              if datetime.fromisoformat(h['timestamp']) >= start]
        except ValueError:
            pass
    
    if end_date:
        try:
            end = datetime.fromisoformat(end_date)
            filtered_history = [h for h in filtered_history 
                              if datetime.fromisoformat(h['timestamp']) <= end]
        except ValueError:
            pass
    
    # Paginate
    total = len(filtered_history)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated = filtered_history[start_idx:end_idx]
    
    return jsonify({
        'history': paginated,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page
    }), 200


@history_bp.route('/history/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    """Get a specific prediction by ID."""
    for pred in _prediction_history:
        if pred['id'] == prediction_id:
            return jsonify(pred), 200
    
    return jsonify({'error': 'Prediction not found'}), 404


@history_bp.route('/history/<prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """Delete a prediction from history."""
    global _prediction_history
    for i, pred in enumerate(_prediction_history):
        if pred['id'] == prediction_id:
            _prediction_history.pop(i)
            return jsonify({'message': 'Prediction deleted'}), 200
    
    return jsonify({'error': 'Prediction not found'}), 404


@history_bp.route('/history/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics."""
    total = len(_prediction_history)
    fertile_count = sum(1 for p in _prediction_history if p.get('prediction') == 'fertile')
    infertile_count = sum(1 for p in _prediction_history if p.get('prediction') == 'infertile')
    
    # Calculate average confidence
    confidences = [p.get('confidence', 0) for p in _prediction_history]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return jsonify({
        'total_predictions': total,
        'fertile_count': fertile_count,
        'infertile_count': infertile_count,
        'average_confidence': round(avg_confidence, 4),
        'accuracy_rate': round(fertile_count / total, 4) if total > 0 else 0
    }), 200


def add_to_history(prediction_data):
    """Add a prediction to history (called from prediction route)."""
    prediction_data['id'] = str(uuid.uuid4())
    prediction_data['timestamp'] = datetime.utcnow().isoformat()
    _prediction_history.append(prediction_data)
    return prediction_data