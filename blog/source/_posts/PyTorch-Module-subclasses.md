---
title: PyTorch Module subclasses
date: 2022-07-20 22:17:55
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下PyTorch `nn.Module`基类的常用subclasses。

<!--more-->

### torch.nn.Conv2d

对由多个input planes组成的input signal进行二维卷积。其完整声明形式为：

```python
CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

`torch.nn.Conv2d`有如下的继承关系：

```python
class Conv2d(_ConvNd):
class _ConvND(Module):
```

在最简单的样例中，input size为$(N, Cin, H, W)$的层的输出值和输出$(N, Cout, Hout, Wout)$可以精确地描述为：

![image20220718203916534](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220718203916534.png)

其中*表示有效的2D cross-correlation 操作，N表示batch size，C表示通道数，H和W分别表示像素高宽。

> 参考资料：
>
> 1. [CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

### torch.nn.BatchNorm2d

### torch.nn.GroupNorm

### torch.nn.Sequential

