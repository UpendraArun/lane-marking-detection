import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLossCEAndDice(nn.Module):
    def __init__(self, weight_ce, weight_dice):
        super(CombinedLossCEAndDice, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_ce)
        self.weight_dice = weight_dice

    def forward(self, outputs, targets):
        epsilon = 1e-10

        # Cross-entropy loss
        ce_loss = self.ce_loss(outputs, targets)

        # Soft Dice loss
        probs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()

        # Permute dimensions to match the order (batch_size, height, width, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        # Calculate intersection and union
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 1, 2))
        union = torch.sum(probs + targets_one_hot, dim=(0, 1, 2))

        dice_loss = 1.0 - (2.0 * intersection + epsilon) / (union + intersection + epsilon)

        # Combine the losses
        combined_loss = ce_loss + self.weight_dice * dice_loss.mean()

        return combined_loss
