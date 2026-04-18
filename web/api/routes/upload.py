"""Upload API routes for Duck Egg Fertility Detection."""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

upload_bp = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle single file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, unique_filename)
        file.save(filepath)
        
        return jsonify({
            'id': str(uuid.uuid4()),
            'filename': unique_filename,
            'original_filename': filename,
            'size': os.path.getsize(filepath),
            'uploaded_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@upload_bp.route('/upload/batch', methods=['POST'])
def upload_batch():
    """Handle batch file upload."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
    os.makedirs(upload_folder, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(upload_folder, unique_filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': unique_filename,
                    'original_filename': filename,
                    'size': os.path.getsize(filepath)
                })
            except Exception as e:
                uploaded_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    return jsonify({'uploaded': uploaded_files}), 200


@upload_bp.route('/upload/dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload with labels."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    label = request.form.get('label', 'unlabeled')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', '/app/data/uploads')
    dataset_folder = os.path.join(upload_folder, 'dataset', label)
    os.makedirs(dataset_folder, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(dataset_folder, unique_filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': unique_filename,
                    'label': label
                })
            except Exception as e:
                uploaded_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    return jsonify({'uploaded': uploaded_files, 'label': label}), 200