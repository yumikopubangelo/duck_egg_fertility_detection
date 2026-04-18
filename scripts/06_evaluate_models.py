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
import yaml
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

from src.segmentation.unet import UNet, create_unet_for_eggs, calculate_iou, calculate_dice_coefficient
from src.segmentation.data_loader import EggDataset
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
    metrics: EvaluationMetrics
) -> None:
    """Evaluate UNet model on test dataset"""
    
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(images)

            if _is_classification_target(targets):
                logits = _reduce_logits_for_classification(outputs)
                probs = torch.sigmoid(logits)
                pred_labels = (probs > 0.5)
                true_labels = (targets.float().view(-1) > 0.5)

                tp = torch.logical_and(pred_labels, true_labels).sum().item()
                fp = torch.logical_and(pred_labels, ~true_labels).sum().item()
                fn = torch.logical_and(~pred_labels, true_labels).sum().item()
                tn = torch.logical_and(~pred_labels, ~true_labels).sum().item()

                accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
                dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

                metrics.update_unet(iou, dice, accuracy, precision, recall)

                total_iou += iou
                total_dice += dice
                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                n_samples += 1
            else:
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()

                # Calculate metrics for segmentation batches
                for i in range(len(images)):
                    iou = calculate_iou(outputs[i:i+1], targets[i:i+1])
                    dice = calculate_dice_coefficient(outputs[i:i+1], targets[i:i+1])

                    pred_mask = pred_masks[i].byte()
                    true_mask = targets[i].byte()

                    tp = (pred_mask & true_mask).sum().item()
                    fp = (pred_mask & ~true_mask).sum().item()
                    fn = (~pred_mask & true_mask).sum().item()
                    tn = (~pred_mask & ~true_mask).sum().item()

                    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                    metrics.update_unet(iou, dice, accuracy, precision, recall)

                    total_iou += iou
                    total_dice += dice
                    total_accuracy += accuracy
                    total_precision += precision
                    total_recall += recall
                    n_samples += 1
    
    if n_samples > 0:
        avg_iou = total_iou / n_samples
        avg_dice = total_dice / n_samples
        avg_accuracy = total_accuracy / n_samples
        avg_precision = total_precision / n_samples
        avg_recall = total_recall / n_samples
        
        print(f"UNet Evaluation - IoU: {avg_iou:.3f}, Dice: {avg_dice:.3f}, "
              f"Accuracy: {avg_accuracy:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")
    
    return metrics


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


def visualize_predictions(
    images: torch.Tensor,
    true_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    output_dir: Path,
    n_samples: int = 5
) -> None:
    """Visualize model predictions"""
    
    n_samples = min(n_samples, len(images))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(true_masks[i, 0].numpy(), cmap='gray')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(pred_masks[i, 0].numpy(), cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    # Save visualization
    vis_path = output_dir / 'predictions_visualization.png'
    plt.savefig(vis_path)
    plt.close()
    
    print(f"Predictions visualization saved to: {vis_path}")


def save_evaluation_results(
    metrics: EvaluationMetrics,
    config: Dict,
    output_dir: Path
) -> None:
    """Save evaluation results to file"""
    
    results = {
        'unet_metrics': metrics.unet_metrics,
        'awc_metrics': metrics.awc_metrics,
        'overall_metrics': metrics.overall_metrics,
        'averages': metrics.get_averages(),
        'config': config
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
    
    # UNet test dataset
    test_transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EggDataset(
        config['data']['test_fertile_dir'],
        config['data']['test_infertile_dir'],
        config['data']['image_size'],
        test_transform
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
        n_channels=3,
        n_classes=1,
        bilinear=config['model']['bilinear'],
        dropout_rate=config['model']['dropout_rate'],
        lightweight=config['model']['lightweight']
    )
    
    unet_checkpoint = torch.load(config['model']['unet_checkpoint'], map_location=device)
    unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
    unet_model = unet_model.to(device)
    
    logger.info(f"UNet model loaded from: {config['model']['unet_checkpoint']}")
    
    # Load AWC model
    awc_model = AdaptiveWeightedClustering.load(config['model']['awc_checkpoint'])
    logger.info(f"AWC model loaded from: {config['model']['awc_checkpoint']}")
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    
    # Evaluate UNet model
    logger.info("\nEvaluating UNet model...")
    evaluate_unet_model(unet_model, test_loader, device, metrics)
    
    # Evaluate AWC model
    logger.info("\nEvaluating AWC model...")
    awc_pred_labels = evaluate_awc_model(awc_model, awc_X, awc_true_labels, metrics)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Get sample predictions
    with torch.no_grad():
        images, masks = next(iter(test_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        unet_model.eval()
        pred_masks = (torch.sigmoid(unet_model(images)) > 0.5).float()

        if masks.ndim >= 3:
            visualize_predictions(images, masks, pred_masks, output_dir)
        else:
            logger.info("Skipping segmentation visualization for classification labels.")
    
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
    save_evaluation_results(metrics, config, output_dir)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    averages = metrics.get_averages()
    logger.info(f"UNet - IoU: {averages['unet']['iou']:.3f}, Dice: {averages['unet']['dice']:.3f}")
    logger.info(f"AWC - Silhouette: {averages['awc']['silhouette']:.3f}")
    
    logger.info(f"\nEvaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
