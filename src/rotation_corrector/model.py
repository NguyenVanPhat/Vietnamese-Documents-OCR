import timm 
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, input):
        return self.model(input)