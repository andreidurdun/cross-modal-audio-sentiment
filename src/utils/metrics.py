"""
Evaluation metrics for emotion recognition.
Includes Macro-F1, Confusion Matrix, and other metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_accuracy(predictions, targets):
    """
    Compute accuracy.
    
    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels
    
    Returns:
        Accuracy value
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return accuracy_score(targets, predictions)


def compute_macro_f1(predictions, targets):
    """
    Compute Macro-F1 score.
    
    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels
    
    Returns:
        Macro-F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return f1_score(targets, predictions, average='macro')


def compute_weighted_f1(predictions, targets):
    """
    Compute Weighted-F1 score.
    
    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels
    
    Returns:
        Weighted-F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return f1_score(targets, predictions, average='weighted')


def compute_confusion_matrix(predictions, targets, num_classes=4):
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_classification_report(predictions, targets, class_names=None):
    """
    Generate detailed classification report.
    
    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels
        class_names: List of class names
    
    Returns:
        Classification report string
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return classification_report(
        targets,
        predictions,
        target_names=class_names,
        digits=4
    )


class MetricsTracker:
    """
    Track metrics during training and evaluation.
    """
    
    def __init__(self, num_classes=4, class_names=None):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.total_loss = 0.0
        self.num_samples = 0
    
    def update(self, predictions, targets, loss=None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted labels or logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            loss: Loss value for this batch
        """
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        
        if loss is not None:
            self.total_loss += loss * len(targets)
            self.num_samples += len(targets)
    
    def compute_metrics(self):
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'macro_f1': f1_score(targets, predictions, average='macro'),
            'weighted_f1': f1_score(targets, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(targets, predictions, labels=list(range(self.num_classes)))
        }
        
        if self.num_samples > 0:
            metrics['loss'] = self.total_loss / self.num_samples
        
        return metrics
    
    def print_metrics(self):
        """Print summary of metrics."""
        metrics = self.compute_metrics()
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        if 'loss' in metrics:
            print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nClassification Report:")
        print(generate_classification_report(
            np.array(self.all_predictions),
            np.array(self.all_targets),
            self.class_names
        ))
        
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test metrics
    num_classes = 4
    batch_size = 20
    
    # Create dummy predictions and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test individual metrics
    acc = compute_accuracy(logits, targets)
    macro_f1 = compute_macro_f1(logits, targets)
    cm = compute_confusion_matrix(logits, targets, num_classes)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Test metrics tracker
    class_names = ['Angry', 'Happy', 'Sad', 'Neutral']
    tracker = MetricsTracker(num_classes, class_names)
    tracker.update(logits, targets, loss=0.5)
    tracker.print_metrics()
