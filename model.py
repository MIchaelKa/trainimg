import torch
import torch.nn as nn

class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
#         resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    
        self.fc = nn.Linear(512, 5)
        
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