import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
