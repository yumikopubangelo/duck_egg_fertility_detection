"""Flask application entry point for Duck Egg Fertility Detection System."""

import os
from flask import Flask, render_template
from flask_cors import CORS
from web.api.extensions import db, migrate, jwt

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, 'templates')
STATIC_DIR = os.path.join(FRONTEND_DIR, 'static')

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

    # Load configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'dev-jwt-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'data/uploads')
    app.config['MODEL_FOLDER'] = os.environ.get('MODEL_FOLDER', 'models')

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    CORS(app)

    # Register models so SQLAlchemy knows about them, then create tables
    from web.api.models import PredictionRecord  # noqa: F401
    with app.app_context():
        db.create_all()

    # Create upload and model folders if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

    # Register blueprints
    from web.api.routes.prediction import prediction_bp
    from web.api.routes.upload import upload_bp
    from web.api.routes.history import history_bp
    from web.api.routes.dataset import dataset_bp
    from web.api.routes.retrain import retrain_bp
    from web.api.routes.admin import admin_bp
    from web.api.routes.analysis import analysis_bp
    from web.api.routes.segmentation import segmentation_bp

    app.register_blueprint(prediction_bp, url_prefix='/api')
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(history_bp, url_prefix='/api')
    app.register_blueprint(dataset_bp, url_prefix='/api')
    app.register_blueprint(retrain_bp, url_prefix='/api')
    app.register_blueprint(admin_bp, url_prefix='/api')
    app.register_blueprint(analysis_bp, url_prefix='/api')
    app.register_blueprint(segmentation_bp, url_prefix='/api')

    # --- HTML page routes ---
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload')
    def upload():
        return render_template('upload.html')

    @app.route('/history')
    def history():
        return render_template('history.html')

    @app.route('/dataset')
    def dataset():
        return render_template('dataset.html')

    @app.route('/retrain')
    def retrain():
        return render_template('retrain.html')

    @app.route('/admin')
    def admin():
        return render_template('admin.html')

    @app.route('/analysis')
    def analysis():
        return render_template('analysis.html')

    @app.route('/batch')
    def batch():
        return render_template('batch.html')

    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy'}, 200

    return app

# Create app instance for development
app = create_app()

from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/1'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/1')
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
