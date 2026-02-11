"""
Loss functions for training.
Implements Focal Loss to handle class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights (None or tensor of shape (num_classes,))
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # Get probabilities for target classes
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1))
        pt = (probs * targets_one_hot).sum(dim=-1)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = -torch.log(pt + 1e-8)
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                self.alpha = torch.tensor(self.alpha)
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for class imbalance.
    """
    
    def __init__(self, class_weights=None):
        """
        Args:
            class_weights: Tensor of class weights (num_classes,)
        """
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Compute weighted cross-entropy loss.
        
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Loss value
        """
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)
        else:
            weights = None
        
        return F.cross_entropy(inputs, targets, weight=weights)


def compute_class_weights(labels, num_classes):
    """
    Compute class weights based on inverse frequency.
    
    Args:
        labels: Tensor or list of all training labels
        num_classes: Number of classes
    
    Returns:
        Tensor of class weights
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # Count samples per class
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    
    # Compute weights as inverse frequency
    class_weights = 1.0 / (class_counts + 1e-8)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights


def test_losses():
    """Test loss functions."""
    batch_size = 8
    num_classes = 4
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss_value = focal_loss(logits, targets)
    print(f"Focal Loss: {loss_value.item():.4f}")
    
    # Test with class weights
    class_weights = torch.tensor([1.0, 2.0, 1.5, 0.8])
    focal_loss_weighted = FocalLoss(gamma=2.0, alpha=class_weights)
    loss_value_weighted = focal_loss_weighted(logits, targets)
    print(f"Focal Loss (weighted): {loss_value_weighted.item():.4f}")
    
    # Test weighted CE
    ce_loss = WeightedCrossEntropyLoss(class_weights)
    ce_value = ce_loss(logits, targets)
    print(f"Weighted CE Loss: {ce_value.item():.4f}")
    
    # Test class weight computation
    all_labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    weights = compute_class_weights(all_labels, num_classes)
    print(f"Computed class weights: {weights}")


if __name__ == "__main__":
    test_losses()
