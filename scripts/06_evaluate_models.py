"""
Model Evaluation Script

This script evaluates trained models (UNet and AWC) on test datasets.
It calculates various metrics and generates visualizations for model performance.

Usage:
    python scripts/06_evaluate_models.py --config configs/evaluation_config.yaml

Evaluation Metrics:
- UNet: IoU, Dice coefficient, Accuracy, Precision, Recall
- AWC: Silhouette score, Davies-Bouldin index, Calinski-Harabasz index
- Overall: Confusion matrix, Classification report
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from torchvision import transforms

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.segmentation.unet import UNet, create_unet_for_eggs
from src.segmentation.data_loader import EggDataset
from src.segmentation.evaluation import (
    colorize_mask,
    denormalize_image_tensor,
    metrics_from_confusion_matrix,
    overlay_mask,
    per_sample_segmentation_metrics,
    predict_class_map,
    confusion_matrix_for_masks,
)
from src.clustering.awc import AdaptiveWeightedClustering, evaluate_clustering, visualize_clusters
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.file_utils import create_directories


def _is_classification_target(targets: torch.Tensor) -> bool:
    return targets.ndim <= 2 and (targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1))


def _reduce_logits_for_classification(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 4:
        return outputs.mean(dim=(2, 3)).squeeze(1)
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return outputs.squeeze(1)
    return outputs.view(-1)


SEGMENTATION_CLASS_NAMES = ["background", "vascularization", "embryo"]


class EvaluationMetrics:
    """Track evaluation metrics for models"""
    
    def __init__(self):
        self.unet_metrics = {
            'iou': [],
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        self.awc_metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        self.overall_metrics = {
            'confusion_matrix': None,
            'classification_report': None
        }
    
    def update_unet(self, iou: float, dice: float, accuracy: float, precision: float, recall: float):
        self.unet_metrics['iou'].append(iou)
        self.unet_metrics['dice'].append(dice)
        self.unet_metrics['accuracy'].append(accuracy)
        self.unet_metrics['precision'].append(precision)
        self.unet_metrics['recall'].append(recall)
    
    def update_awc(self, silhouette: float, davies_bouldin: float, calinski_harabasz: float):
        self.awc_metrics['silhouette'].append(silhouette)
        self.awc_metrics['davies_bouldin'].append(davies_bouldin)
        self.awc_metrics['calinski_harabasz'].append(calinski_harabasz)
    
    def update_confusion_matrix(self, matrix: np.ndarray):
        self.overall_metrics['confusion_matrix'] = matrix
    
    def update_classification_report(self, report: str):
        self.overall_metrics['classification_report'] = report
    
    def get_averages(self) -> Dict[str, float]:
        return {
            'unet': {
                'iou': np.mean(self.unet_metrics['iou']),
                'dice': np.mean(self.unet_metrics['dice']),
                'accuracy': np.mean(self.unet_metrics['accuracy']),
                'precision': np.mean(self.unet_metrics['precision']),
                'recall': np.mean(self.unet_metrics['recall'])
            },
            'awc': {
                'silhouette': np.mean(self.awc_metrics['silhouette']),
                'davies_bouldin': np.mean(self.awc_metrics['davies_bouldin']),
                'calinski_harabasz': np.mean(self.awc_metrics['calinski_harabasz'])
            }
        }


def evaluate_unet_model(
    model: UNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: EvaluationMetrics,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Evaluate a multiclass U-Net model and return a detailed report."""

    class_names = class_names or SEGMENTATION_CLASS_NAMES
    model.eval()
    num_classes = len(class_names)
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    sample_rows = []
    visualization_rows = []
    sample_index = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = predict_class_map(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()
            images_cpu = images.cpu()

            for i in range(len(images_cpu)):
                pred_mask = preds[i]
                true_mask = targets_np[i]
                total_cm += confusion_matrix_for_masks(pred_mask, true_mask, num_classes=num_classes)

                sample_metrics = per_sample_segmentation_metrics(
                    pred_mask,
                    true_mask,
                    class_names=class_names,
                    background_index=0,
                )
                metrics.update_unet(
                    sample_metrics["mean_iou"],
                    sample_metrics["mean_dice"],
                    sample_metrics["foreground_mean_dice"],
                    sample_metrics["foreground_mean_iou"],
                    sample_metrics["per_class"][2]["recall"] if len(sample_metrics["per_class"]) > 2 else 0.0,
                )

                sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "mean_iou": sample_metrics["mean_iou"],
                        "mean_dice": sample_metrics["mean_dice"],
                        "foreground_mean_iou": sample_metrics["foreground_mean_iou"],
                        "foreground_mean_dice": sample_metrics["foreground_mean_dice"],
                        "per_class": sample_metrics["per_class"],
                    }
                )
                visualization_rows.append(
                    {
                        "sample_index": sample_index,
                        "image_rgb": denormalize_image_tensor(images_cpu[i]),
                        "true_mask": true_mask.astype(np.uint8),
                        "pred_mask": pred_mask.astype(np.uint8),
                        "metrics": sample_metrics,
                    }
                )
                sample_index += 1

    report = metrics_from_confusion_matrix(total_cm, class_names=class_names, background_index=0)
    metrics.update_confusion_matrix(total_cm)

    sample_rows_sorted = sorted(sample_rows, key=lambda row: row["foreground_mean_dice"])
    visualization_rows_by_idx = {row["sample_index"]: row for row in visualization_rows}

    def _pick_sample(position: str) -> Optional[Dict]:
        if not sample_rows_sorted:
            return None
        if position == "best":
            chosen = sample_rows_sorted[-1]
        elif position == "worst":
            chosen = sample_rows_sorted[0]
        else:
            chosen = sample_rows_sorted[len(sample_rows_sorted) // 2]
        vis_row = visualization_rows_by_idx[chosen["sample_index"]]
        return {
            **chosen,
            "image_rgb": vis_row["image_rgb"],
            "true_mask": vis_row["true_mask"],
            "pred_mask": vis_row["pred_mask"],
        }

    report["sample_count"] = len(sample_rows)
    report["examples"] = {
        "best": _pick_sample("best"),
        "median": _pick_sample("median"),
        "worst": _pick_sample("worst"),
    }

    print(
        "UNet Multiclass Evaluation - "
        f"Mean IoU: {report['mean_iou']:.3f}, Mean Dice: {report['mean_dice']:.3f}, "
        f"Foreground IoU: {report['foreground_mean_iou']:.3f}, "
        f"Foreground Dice: {report['foreground_mean_dice']:.3f}, "
        f"Pixel Accuracy: {report['pixel_accuracy']:.3f}"
    )

    return report


def evaluate_awc_model(
    model: AdaptiveWeightedClustering,
    X: np.ndarray,
    true_labels: Optional[np.ndarray],
    metrics: EvaluationMetrics
) -> np.ndarray:
    """Evaluate AWC model on test dataset"""
    
    # Predict labels
    pred_labels = model.predict(X)
    
    # Calculate evaluation metrics
    evaluation = evaluate_clustering(X, pred_labels, true_labels)
    
    # Update metrics
    metrics.update_awc(
        evaluation.get('silhouette', 0.0),
        evaluation.get('davies_bouldin', 0.0),
        evaluation.get('calinski_harabasz', 0.0)
    )
    
    print(f"AWC Evaluation - Silhouette: {evaluation.get('silhouette', 0.0):.3f}, "
          f"Davies-Bouldin: {evaluation.get('davies_bouldin', 0.0):.3f}, "
          f"Calinski-Harabasz: {evaluation.get('calinski_harabasz', 0.0):.3f}")
    
    if 'adjusted_rand' in evaluation:
        print(f"Adjusted Rand Index: {evaluation['adjusted_rand']:.3f}")
    
    if 'normalized_mutual_info' in evaluation:
        print(f"Normalized Mutual Information: {evaluation['normalized_mutual_info']:.3f}")
    
    return pred_labels


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_dir: Path
) -> np.ndarray:
    """Generate and save confusion matrix"""
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    return cm


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_dir: Path
) -> str:
    """Generate and save classification report"""
    
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Save report as JSON
    report_path = output_dir / 'classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    return json.dumps(report, indent=4)


def visualize_predictions(report: Dict, output_dir: Path) -> None:
    """Visualize best, median, and worst multiclass predictions."""

    examples = [
        ("Terbaik", report.get("examples", {}).get("best")),
        ("Sedang", report.get("examples", {}).get("median")),
        ("Terburuk", report.get("examples", {}).get("worst")),
    ]
    examples = [(title, ex) for title, ex in examples if ex is not None]
    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 4.8 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (title, example) in enumerate(examples):
        image_rgb = example["image_rgb"]
        true_rgb = colorize_mask(example["true_mask"])
        pred_overlay = overlay_mask(image_rgb, example["pred_mask"])
        dice = example["foreground_mean_dice"]
        iou = example["foreground_mean_iou"]

        axes[row_idx, 0].imshow(image_rgb)
        axes[row_idx, 0].set_title(f"{title} - Citra Asli")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(true_rgb)
        axes[row_idx, 1].set_title("Mask Ground Truth")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred_overlay)
        axes[row_idx, 2].set_title(f"Prediksi | Dice FG {dice:.3f} | IoU FG {iou:.3f}")
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    vis_path = output_dir / "predictions_visualization.png"
    plt.savefig(vis_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Predictions visualization saved to: {vis_path}")


def save_segmentation_report(report: Dict, output_dir: Path) -> None:
    """Save the multiclass segmentation report as JSON and CSV."""

    examples = {}
    for label, example in dict(report.get("examples", {})).items():
        if example is None:
            examples[label] = None
            continue
        examples[label] = {
            key: value
            for key, value in example.items()
            if key not in {"image_rgb", "true_mask", "pred_mask"}
        }

    json_report = {
        key: value
        for key, value in report.items()
        if key != "examples"
    }
    json_report["examples"] = examples

    report_path = output_dir / "segmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(json_report, f, indent=4, cls=NumpyEncoder)

    per_class_df = pd.DataFrame(report["per_class"])
    per_class_df.to_csv(output_dir / "segmentation_per_class_metrics.csv", index=False)

    cm = np.asarray(report["confusion_matrix"], dtype=np.int64)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=report["class_names"],
        yticklabels=report["class_names"],
    )
    plt.title("Pixel-Level Confusion Matrix (U-Net)")
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_confusion_matrix.png", dpi=160)
    plt.close()


def save_evaluation_results(
    metrics: EvaluationMetrics,
    config: Dict,
    output_dir: Path,
    segmentation_report: Optional[Dict] = None,
) -> None:
    """Save evaluation results to file"""
    
    results = {
        'unet_metrics': metrics.unet_metrics,
        'awc_metrics': metrics.awc_metrics,
        'overall_metrics': metrics.overall_metrics,
        'averages': metrics.get_averages(),
        'config': config
    }
    if segmentation_report is not None:
        results['segmentation_report'] = {
            key: value
            for key, value in segmentation_report.items()
            if key != 'examples'
        }
    
    # Save as JSON
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    print(f"Evaluation results saved to: {results_path}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = Path(config['evaluation']['log_dir'])
    if args.output_dir:
        log_dir = Path(args.output_dir) / 'logs'
    
    create_directories(log_dir)
    logger = setup_logger(log_dir / 'evaluation.log')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['evaluation']['output_dir'])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    create_directories(output_dir)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Load test datasets
    logger.info("Loading test datasets...")

    seg_data_cfg = dict(config.get('segmentation_data', {}))
    if not seg_data_cfg:
        seg_data_cfg = {
            'test_image_dir': config['data'].get('test_image_dir') or 'data/segmentation/test/images',
            'test_mask_dir': config['data'].get('test_mask_dir') or 'data/segmentation/test/masks',
            'image_size': config['data'].get('segmentation_image_size') or [256, 256],
        }

    seg_image_size = tuple(seg_data_cfg.get('image_size', [256, 256]))
    test_transform = transforms.Compose([
        transforms.Resize(seg_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EggDataset(
        image_size=seg_image_size,
        transform=test_transform,
        image_dir=seg_data_cfg['test_image_dir'],
        mask_dir=seg_data_cfg['test_mask_dir'],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers'],
        pin_memory=True
    )
    
    # AWC test data
    awc_X = np.load(config['data']['awc_test_data'])
    awc_true_labels = np.load(config['data']['awc_test_labels']) if config['data']['awc_test_labels'] else None
    
    # Load models
    logger.info("Loading models...")
    
    # Load UNet model
    unet_model = create_unet_for_eggs(
        n_channels=config['model'].get('n_channels', 3),
        n_classes=config['model'].get('n_classes', 3),
        bilinear=config['model']['bilinear'],
        dropout_rate=config['model']['dropout_rate'],
        lightweight=config['model']['lightweight']
    )
    
    unet_checkpoint = torch.load(config['model']['unet_checkpoint'], map_location=device)
    unet_state = unet_checkpoint.get('model_state_dict', unet_checkpoint)
    unet_model.load_state_dict(unet_state)
    unet_model = unet_model.to(device)
    
    logger.info(f"UNet model loaded from: {config['model']['unet_checkpoint']}")
    
    # Load AWC model
    awc_model = AdaptiveWeightedClustering.load(config['model']['awc_checkpoint'])
    logger.info(f"AWC model loaded from: {config['model']['awc_checkpoint']}")
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    
    # Evaluate UNet model
    logger.info("\nEvaluating UNet model...")
    segmentation_report = evaluate_unet_model(
        unet_model,
        test_loader,
        device,
        metrics,
        class_names=SEGMENTATION_CLASS_NAMES[: config['model'].get('n_classes', 3)],
    )
    
    # Evaluate AWC model
    logger.info("\nEvaluating AWC model...")
    awc_pred_labels = evaluate_awc_model(awc_model, awc_X, awc_true_labels, metrics)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    visualize_predictions(segmentation_report, output_dir)
    save_segmentation_report(segmentation_report, output_dir)
    
    # Generate confusion matrix and classification report
    if awc_true_labels is not None:
        logger.info("\nGenerating confusion matrix and classification report...")
        
        # Generate confusion matrix
        cm = generate_confusion_matrix(
            awc_true_labels,
            awc_pred_labels,
            labels=[f"Cluster {i}" for i in range(config['model']['n_clusters'])],
            output_dir=output_dir
        )
        
        metrics.update_confusion_matrix(cm)
        
        # Generate classification report
        report = generate_classification_report(
            awc_true_labels,
            awc_pred_labels,
            labels=[f"Cluster {i}" for i in range(config['model']['n_clusters'])],
            output_dir=output_dir
        )
        
        metrics.update_classification_report(report)
    
    # Save evaluation results
    logger.info("\nSaving evaluation results...")
    save_evaluation_results(metrics, config, output_dir, segmentation_report=segmentation_report)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    averages = metrics.get_averages()
    logger.info(
        "UNet - Mean IoU: %.3f, Mean Dice: %.3f, Foreground IoU: %.3f, Foreground Dice: %.3f, Pixel Accuracy: %.3f",
        segmentation_report['mean_iou'],
        segmentation_report['mean_dice'],
        segmentation_report['foreground_mean_iou'],
        segmentation_report['foreground_mean_dice'],
        segmentation_report['pixel_accuracy'],
    )
    logger.info(f"AWC - Silhouette: {averages['awc']['silhouette']:.3f}")
    
    logger.info(f"\nEvaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
