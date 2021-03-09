import torch
import torch.nn as nn
import timm

from config import GlobalConfig

class CustomResNet(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        print(f'Init timm {model_name}, pretrained: {pretrained}')

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.conv1[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, GlobalConfig.target_size)

    def forward(self, x):
        x = self.model(x)
        return x