import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from dataset import get_classes

classes, num_classes = get_classes()

class ResNet50CatDog(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
