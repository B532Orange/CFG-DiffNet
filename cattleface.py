import sys

sys.path.append("..")
from model import iresnet
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn


def builder(args):
    model = SoftmaxBuilder(args)
    return model


class SoftmaxBuilder(nn.Module):
    def __init__(self, args):
        super(SoftmaxBuilder, self).__init__()
        self.device = args.device
        self.features =iresnet.iresnet(num_classes=args.embedding_size)
        self.fc = ArcFace(args.input_fc_size, args.last_fc_size, 64, 0.5)

    def forward(self, x, target):
        x,_ = self.features(x)
    
        logits, cosine = self.fc(x, target)

        return logits, cosine


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
