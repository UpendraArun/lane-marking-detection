import torch
import torchvision.models as models
from torchvision.models import resnet101

from torchvision.models import ResNet101_Weights

# Load the pre-trained ResNet-101 backbone
resnet101_backbone = models.resnet.resnet101(pretrained=True)

#resnet101_backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

# Save the ResNet-101 backbone's state dictionary to a .pth file
torch.save(resnet101_backbone.state_dict(), 'resnet101.pth')

print("ResNet-101 backbone, pre-trained model in .pth format saved.")
