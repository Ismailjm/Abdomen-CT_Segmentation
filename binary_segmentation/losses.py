import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Computes the Dice Loss (1 - Dice Coefficient)."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        total_pixels = probs_flat.sum() + targets_flat.sum()
        dice_coeff = (2. * intersection + self.smooth) / (total_pixels + self.smooth)
        
        return 1 - dice_coeff

class DiceBCELoss(nn.Module):
    """Computes a weighted sum of Dice Loss and BCEWithLogitsLoss."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        d_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        return (self.dice_weight * d_loss) + (self.bce_weight * bce_loss)