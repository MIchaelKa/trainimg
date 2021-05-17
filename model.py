import torch
import torch.nn as nn

from torchvision import models
import torch.nn.functional as F

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock

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

#
# ResNet
#
class ResNetModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        print(f'Init torchvision {model_name}, pretrained: {pretrained}')

        model_func = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnext50_32x4d': models.resnext50_32x4d,
        }[model_name]
        resnet = model_func(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # self.dropout = nn.Dropout(0.5)

        in_features = resnet.fc.in_features
        self.fc = nn.Linear(in_features, GlobalConfig.target_size)
        
    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        
        return x

class MyResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

        # self.inplanes = 16
            
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )

        # replace_stride_with_dilation = [False, False, False]

        # self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
    
        in_features = 256 * block.expansion
        self.fc = nn.Linear(in_features, GlobalConfig.target_size)

        
        # del self.conv1
        # del self.bn1
        # del self.relu
        # del self.maxpool

        
    def forward(self, x):
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # print(x.shape)

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)     
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    return MyResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

#
# SimpleModel
#
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

#
# SimpleNet
#
class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.MaxPool2d(2)

        # self.downsample = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        # )

    def forward(self, x):
        x = self.block(x)
        x = self.downsample(x)
        return x 


class SimpleNet(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        # print('init SimpleNet')

        channels = [3, 64, 128, 256, 512]
        self.backbone = nn.Sequential(
            BaseBlock(channels[0], channels[1]), # 16x16
            BaseBlock(channels[1], channels[2]), # 8x8
            BaseBlock(channels[2], channels[3]), # 4x4
            BaseBlock(channels[3], channels[4]), # 2x2
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.dropout = nn.Dropout(dropout_p)

        image_size = 2
        self.fc = nn.Linear(channels[-1]*(image_size**2), 10)

        # self.fc = nn.Linear(channels[-1], 10)

    def forward(self, x):  
        x = self.backbone(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


def get_num_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)