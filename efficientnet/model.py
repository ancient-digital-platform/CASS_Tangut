import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
        self.model = efficientnet_b0(weights=weights)
        # EfficientNet的分类器在classifier属性中
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
