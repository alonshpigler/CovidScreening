import math

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class ArcMarginModel(nn.Module):
    def __init__(self, m=0.5, s=64, easy_margin=False, emb_size=512):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        # num_classes Total number of face classifications in the training set
        # emb_size eigenvector length
        nn.init.xavier_uniform_(self.weight)
        # Use uniform distribution to initialize weight

        self.easy_margin = easy_margin
        self.m = m
        #  0.5 in the formula
        self.s = s
        # radius 64 s in the formula
        # Both sizes are recommended values ​​in the paper

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # cos and sin
        self.th = math.cos(math.pi - self.m)
        # threshold, avoid theta + m >= pi
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        # Regularization
        cosine = F.linear(x, W)
        # cos
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # sin
        phi = cosine * self.cos_m - sine * self.sin_m
        # cos(theta + m) Cosine formula
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
            # if using easy_margin
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # Map the label of the sample to one hot form. For example, N labels are mapped to (N, num_classes).
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # For the correct category (1*phi) is the cos(theta + m) in the formula, for the wrong category (1*cosine) ie the cos(theta) in the formula
        # Thus for each sample, such as [0,0,0,1,0,0] belongs to the fourth category, the final result is [cosine, cosine, cosine, phi, cosine, cosine]
        # Multiply by radius, after cross entropy, just the formula of ArcFace
        output *= self.s
        # multiply by radius
