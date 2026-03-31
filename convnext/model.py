import torch.nn as nn
from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = None
        self.model = convnext_tiny(weights=weights)
        # ConvNeXt的分类器在classifier属性中
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
