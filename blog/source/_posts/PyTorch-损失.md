---
title: PyTorch 损失
date: 2022-07-20 22:00:12
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下PyTorch中的常用损失。

<!--more-->

### torch.nn.CrossEntropyLoss

用来计算input和target之间交叉熵损失的criterion。其完整调用形式为：

    torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)

当计算C类元素的分类问题时很有用。如果提供`weights`参数，该参数应该是一个1D Tensor，用来为每类元素分配权重，当训练集unbalance时尤其有用。

input应当包含原始的，未归一化的每一类的分数，input应当是一个Tensor，尺寸为unbatched input的尺寸，例如`(minibatch, C)`或者`(minibatch, C, d1, d2, ...., dk)`，k表示k维数据。后者对高维input很有用，例如计算2D图像的每个pixel的交叉熵损失。

该criterion期望的target应该包含以下两种中的一种：

* 范围为$[0, C)$的类索引，C表示类数目；如果指定`ignore_index`参数，该损失也接受不在前述范围内的类索引。`reduction='none'`时，unreduced loss可以被描述为：
  
  ![image20220704094749082](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704094749082.png)
  
  x，y表示input和target。如果reduction is not 'none'（默认为'mean'），则有
  
  ![image20220704095008493](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704095008493.png)
  
* 每一类的概率；当labels超出每个minibatch项要求的单类之外时有用，例如blended labels，label smoothing。unreduced loss（`reduction='none'`）可以描述为：
  
  ![image20220704095317040](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704095317040.png)
  
  如果reduction is not 'none'（默认为'mean'），则有：
  
  ![image20220704095420471](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704095420471.png)
  

PS：该criterion的性能在target包含类索引时通常更好，因为此时可以进行优化计算。只在对minibatch想来说单类label太受限的情况下，才考虑将target作为类的概率。

参数：

* `weights`：Tensor类型，可选择。作用于每个类的manual rescaling weight。如果给出，应该是一个具有尺寸C的Tensor
* `size_average`：bool类型，可选择。不建议使用（deprecated），见reduction。
* `ignore_index`：int类型，可选。指定一个被忽略的target value，该值不对input gradident起作用。当`size_average=True`，损失在non-ignored targets上取平均。注意只有当target包含类索引时`ignore_index`才有用。
* `reduce`：bool类型，可选。deprecated，见reduction。
* `reduction`：string类型，可选。指定应用到output上的reduction：`none`|`mean`|`sum`。
* `label_smoothing`：floate类型，可选。范围为[0.0, 1.0]的float值，指定计算损失时的平滑程度，0.0表示不进行平滑，默认为0.0。通过平滑targets变成了一个原始真值和高斯分布的混合物，见 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

Shape：C表示类数目，N表示batchsize

* Input：(C)，(N,C)，或者(N, C, d1, d2, ..., dk)
* Target：如果包含类索引，形状为()，(N)或者(N, d1, d2, ..., dk)，每个值的范围为[0, C)。如果包含类概率，形状和Input相同，每个值的范围为[0, 1]
* Output：如果reduction='none'，形状和target相同，否则为标量。

调用实例：

    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()
    # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    output = loss(input, target)
    output.backward()

`torch.nn.CrossEntropyLoss()`有如下的继承关系：

    class CrossEntropyLoss(_WeightedLoss):
    class _WeightedLoss(_Loss):
    class _Loss(Module)

> 参考资料：
>
> 1. [CROSSENTROPYLOSS](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
> 2. [Loss Functions](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#loss-functions)