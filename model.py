import torch
import torch.nn as nn

from torchvision import models
import torch.nn.functional as F

from config import GlobalConfig

class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        print('Init densenet121')

        densenet = models.densenet121(pretrained=True)
        self.backbone = densenet.features
        # self.backbone = nn.Sequential(*list(densenet.children())[:-1])

        in_features = densenet.classifier.in_features  
        self.fc = nn.Linear(in_features, 5)
        
        
    def forward(self, x):
      
        x = self.backbone(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class ResNetModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        print(f'Init torchvision {model_name}, pretrained: {pretrained}')
        resnet = {
            'resnet18': models.resnet18(pretrained=pretrained),
            'resnet34': models.resnet34(pretrained=pretrained),
            'resnext50_32x4d': models.resnext50_32x4d(pretrained=pretrained),
        }[model_name]
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        in_features = resnet.fc.in_features  
        self.fc = nn.Linear(in_features, GlobalConfig.target_size)
        
    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        print('Init SimpleModel')

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Linear(256*8*8, 11)
        
    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_num_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)