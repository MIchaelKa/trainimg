import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        model_name = 'efficientnet-b4'
        print(f'Init {model_name}, pretrained: {pretrained}')

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)

        self.fc = nn.Linear(self.backbone._fc.in_features, 5)     
        self.backbone._fc = nn.Identity()
        
    def forward(self, x):
        
        # x = self.backbone.extract_features(x)
        x = self.backbone(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x