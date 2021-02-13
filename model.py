import torch
import torch.nn as nn

from torchvision import models
import torch.nn.functional as F

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
    def __init__(self):
        super().__init__()
        print('Init resnext50_32x4d')
        # resnet = models.resnet18(pretrained=True)
        # resnet = models.resnet34(pretrained=True)
        resnet = models.resnext50_32x4d(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        in_features = resnet.fc.in_features  
        self.fc = nn.Linear(in_features, 5)
        
    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Linear(256*8*8, 5)
        
    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_num_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)