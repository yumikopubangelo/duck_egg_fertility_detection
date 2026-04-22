"""Dataset API routes for Duck Egg Fertility Detection."""

from flask import Blueprint, request, jsonify, current_app
import os
from datetime import datetime

dataset_bp = Blueprint('dataset', __name__)


@dataset_bp.route('/dataset', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
    dataset_folder = os.path.join(upload_folder, 'dataset')
    
    datasets = []
    if os.path.exists(dataset_folder):
        for label in os.listdir(dataset_folder):
            label_path = os.path.join(dataset_folder, label)
            if os.path.isdir(label_path):
                count = len([f for f in os.listdir(label_path) 
                           if os.path.isfile(os.path.join(label_path, f))])
                datasets.append({
                    'label': label,
                    'count': count,
                    'path': label_path
                })
    
    return jsonify({'datasets': datasets}), 200


@dataset_bp.route('/dataset/stats', methods=['GET'])
def dataset_stats():
    """Get dataset statistics."""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
    dataset_folder = os.path.join(upload_folder, 'dataset')
    
    stats = {
        'total_images': 0,
        'labels': {},
        'last_updated': None
    }
    
    if os.path.exists(dataset_folder):
        latest_mtime = 0
        for label in os.listdir(dataset_folder):
            label_path = os.path.join(dataset_folder, label)
            if os.path.isdir(label_path):
                files = [f for f in os.listdir(label_path) 
                        if os.path.isfile(os.path.join(label_path, f))]
                count = len(files)
                stats['labels'][label] = count
                stats['total_images'] += count
                
                # Get latest modification time
                for f in files:
                    mtime = os.path.getmtime(os.path.join(label_path, f))
                    if mtime > latest_mtime:
                        latest_mtime = mtime
        
        if latest_mtime > 0:
            stats['last_updated'] = datetime.fromtimestamp(latest_mtime).isoformat()
    
    return jsonify(stats), 200


@dataset_bp.route('/dataset/<label>', methods=['GET'])
def get_dataset_by_label(label):
    """Get images for a specific label."""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
    label_path = os.path.join(upload_folder, 'dataset', label)
    
    if not os.path.exists(label_path):
        return jsonify({'error': 'Label not found'}), 404
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    files = [f for f in os.listdir(label_path) 
             if os.path.isfile(os.path.join(label_path, f))]
    
    total = len(files)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated = files[start_idx:end_idx]
    
    return jsonify({
        'label': label,
        'images': paginated,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page
    }), 200


@dataset_bp.route('/dataset/<label>/<filename>', methods=['DELETE'])
def delete_image(label, filename):
    """Delete an image from the dataset."""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
    filepath = os.path.join(upload_folder, 'dataset', label, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        os.remove(filepath)
        return jsonify({'message': 'Image deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dataset_bp.route('/dataset/split', methods=['POST'])
def split_dataset():
    """Split dataset into train/val/test sets."""
    data = request.get_json()
    train_ratio = data.get('train_ratio', 0.7)
    val_ratio = data.get('val_ratio', 0.15)
    test_ratio = data.get('test_ratio', 0.15)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        return jsonify({'error': 'Ratios must sum to 1.0'}), 400
    
    # TODO: Implement actual dataset splitting logic
    return jsonify({
        'message': 'Dataset split initiated',
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }), 200