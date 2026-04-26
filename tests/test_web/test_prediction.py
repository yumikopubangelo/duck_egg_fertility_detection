from pathlib import Path

from src.web.prediction_service import PredictionService
from web.api.app import create_app


def test_prediction_service_runs_on_known_image():
    image_path = Path("data/test/fertile/IMG_8188.jpg")

    result = PredictionService().predict_file(image_path)

    assert result.label in {"fertile", "infertile"}
    assert result.feature_count == 70
    assert result.preprocessed_shape == (256, 256)
    assert 0.0 <= result.confidence <= 1.0


def test_predict_endpoint_returns_prediction(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path))
    image_path = Path("data/preprocessed/test/fertile/IMG_8188.jpg")

    with app.test_client() as client:
        with image_path.open("rb") as image_file:
            response = client.post(
                "/api/predict",
                data={"file": (image_file, image_path.name)},
                content_type="multipart/form-data",
            )

    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in {"fertile", "infertile"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert "cluster_id" in data
    assert data["feature_count"] == 70
