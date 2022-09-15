---
title: PyTorch调试模块实践
date: 2022-09-15 18:49:18
categories:
- 深度学习
tags:
- Pytorch
- 笔记
---

本文记录一下如何在PyTorch中调试单个模块在处理张量过程中张量的尺寸变化。
<!--more-->

以一个边缘生成模块为例，以下为该边缘生成模块的代码：
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride,
            padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)
        

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x4, x1):
        # x1 256 * 104 * 104
        # x4 2048 * 13 * 13
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', 
        align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out
```
可以通过以下测试模块来测试该模块：
```python
class test_EAM(nn.Module):
    def __init__(self):
        super(test_EAM, self).__init__()
        self.eam = EAM()

    def forward(self, x4, x1):
        edge = self.eam(x4, x1)
        edge_att = torch.sigmoid(edge)

        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return edge, edge_att, oe

if __name__=='__main__':
    model = test_EAM()
    model.eval()
    x1 = torch.rand((4, 256, 80, 80))
    # x1 = torch.rand((4, 256, 104, 104))
    x4 = torch.rand((4, 2048, 10, 10))
    # x4 = torch.rand((4, 2048, 13, 13))
    edge, edge_att, oe = model(x4, x1)
```

