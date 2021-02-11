import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        print('Init efficientnet-b0')

        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Linear(self.backbone._fc.in_features, 5)     
        self.backbone._fc = nn.Identity()
        
    def forward(self, x):
        
#         x = self.backbone.extract_features(x)       
        x = self.backbone(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x