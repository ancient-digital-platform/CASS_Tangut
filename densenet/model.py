import torch.nn as nn
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            weights = DenseNet121_Weights.DEFAULT
        else:
            weights = None
        self.model = densenet121(weights=weights)
        # DenseNet的分类器在classifier属性中
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
