---
title: PyTorch数据采样
date: 2022-07-20 22:23:44
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下PyTorch中的数据采样操作。

<!--more-->

### torch.nn.functional.interpolate

其完整声明形式为：

```python
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
```

下/上采样输入到给定的尺寸或者缩放因子。

用于插值的算法由`mode`参数定义。

目前temporal、spatial和volumetric的采样是支持的，例如，期望的输入在shape上是3-D，4-D或者5-D。

输入维度以该方式呈现：mni-batch x channels x [optional depth] x [optional height] x width

可以用来resize的mode有：nearest, linear(3D-only), bilinear, bicubic(4D-only), trilinear(5D-only), area, nearest-exact。

参数：

- `input(Tensor)`：输入tensor
- `size(int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int])`：输入空间尺寸
- `scale_factor (float or Tuple[float])`：对空间尺寸的缩放操作，如果`scale_factor`为一个元组，其长度必须与`input.dim()`匹配
- `mode(str)`：用来上采样的算法：`'nearest'` | `'linear'` | `'bilinear'` | `'bicubic'` | `'trilinear'` | `'area'` | `'nearest-exact'`，默认为`'nearest'`。
- `align_corners(bool, optional)`：略
- `recompute_scale_factor(bool, optional)`：重新计算`scale_factor`以在插值计算中使用。
- `antialias(bool, optional)`：略

> 参考资料：
>
> 1. [TORCH.NN.FUNCTIONAL.INTERPOLATE](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html#torch.nn.functional.interpolate)

### torch.nn.Upsample

对一个给定的多通道 1D (temporal), 2D (spatial) 或 3D (volumetric) 数据进行上采样。

> 参考资料：
>
> 1. [UPSAMPLE](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)

