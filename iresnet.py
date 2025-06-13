import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    BasicBlock used in ResNet18.
    Consists of two convolutional layers with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    """
    ResNet-18 Architecture, input size: 100x100x3, output feature map: 32x32x128
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # (3, 100, 100) -> (64, 50, 50)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (64, 50, 50) -> (64, 25, 25)
        
        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # (64, 25, 25) -> (64, 25, 25)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # (64, 25, 25) -> (128, 12, 12)
        self.layer3 = self._make_layer(128, 128, 2, stride=2) # (128, 12, 12) -> (128, 6, 6)
        
        # Final Layer (output size: 32x32)
        self.avgpool = nn.AdaptiveAvgPool2d((32, 32))  # Ensures final output is 32x32
        
        # A fully connected layer to generate the final output vector
        self.fc = nn.Conv2d(128, 128, kernel_size=1)  # To get the exact feature map size: 32x32x128

        self.atten = Channle_SpatialSENet(in_channels=128)  # 假设 ChannelSpatialSENet 是你自己定义的类
       

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (3, 100, 100) -> (64, 50, 50)
        x = self.maxpool(x)  # (64, 50, 50) -> (64, 25, 25)
        
        # Pass through residual blocks
        x = self.layer1(x)  # (64, 25, 25) -> (64, 25, 25)
        x = self.layer2(x)  # (64, 25, 25) -> (128, 12, 12)
        x = self.layer3(x)  # (128, 12, 12) -> (128, 6, 6)
        
        # Adaptive pooling to ensure the size is 32x32
        x = self.avgpool(x)  # (128, 6, 6) -> (128, 32, 32)
        
        # Final feature map transformation
        x = self.fc(x)  # (128, 32, 32)

        attention = self.atten(x)
        x = x * attention

        return x


class Vector(nn.Module):
    def __init__(self, num_classes=500):
        super(Vector, self).__init__()
        
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.fc1 = nn.Linear(32 * 8 * 8, 500)  
        self.relu_fc1 = nn.ReLU()
        
       
        self.fc2 = nn.Linear(500, num_classes)
        self.features = nn.BatchNorm1d(num_classes, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        x = self.features(x)
        
        return x

class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, s, m, easy_margin=True):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.weight)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = torch.mm(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        cosine = cosine * self.s

        return output, cosine

class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  

class Channle_SpatialSENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(Channle_SpatialSENet, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        combined_attention = channel_out * spatial_out  #[B, C, H, W]
        return combined_attention  

class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.sigmoid(self.deconv(x))
        
        return x

class SoftmaxBuilder(nn.Module):
    def __init__(self, args):
        super(SoftmaxBuilder, self).__init__()
        self.device = args.device
        self.encoder = ResNet18()
        self.feature = Vector(num_classes=args.embedding_size)
        self.fc = ArcFace(args.input_fc_size, args.last_fc_size, 64, 0.1)
        
    def forward(self, x, target):
        x = self.encoder(x)
        
        x = self.feature(x)
        logits, cosine = self.fc(x, target)

        return logits, cosine


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    

class DiffusionCattle(nn.Module):
    def __init__(self, args, diffusion):
        super(DiffusionCattle, self).__init__()
        self.device = args.device
        self.encoder = ResNet18()
        self.feature = Vector(num_classes=args.embedding_size)
        self.diffusion = diffusion
        self.fc = ArcFace(args.input_fc_size, args.last_fc_size, 64, 0.5)

    def forward(self, ground, cond, target):
        if ground is not None:
            frontal = self.encoder(ground)
        else:
            frontal = ''
        side = self.encoder(cond)
        xx = self.diffusion(frontal, side)
        loss_mse = F.mse_loss(xx, frontal)

        x = self.feature(xx)

        logits, cosine = self.fc(x, target)

        return logits, cosine, loss_mse