"""Run fertility inference on duck egg candling images."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).parent.parent))

from src.web.prediction_service import PredictionService


def iter_images(paths: list[str]) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            for pattern in patterns:
                images.extend(path.glob(pattern))
                images.extend(path.glob(pattern.upper()))
        else:
            images.append(path)
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict duck egg fertility from image files")
    parser.add_argument("images", nargs="+", help="Image file(s) or directory path(s)")
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    args = parser.parse_args()

    image_paths = iter_images(args.images)
    if not image_paths:
        raise SystemExit("No images found")

    service = PredictionService()
    results = []
    for image_path in image_paths:
        try:
            prediction = service.predict_file(image_path)
            payload = {
                "image": str(image_path),
                **asdict(prediction),
            }
            results.append(payload)
            if not args.json:
                print(
                    f"{image_path}: {prediction.label} "
                    f"confidence={prediction.confidence:.4f} "
                    f"cluster={prediction.cluster_id}"
                )
        except Exception as exc:
            payload = {"image": str(image_path), "error": str(exc)}
            results.append(payload)
            if not args.json:
                print(f"{image_path}: ERROR {exc}")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
