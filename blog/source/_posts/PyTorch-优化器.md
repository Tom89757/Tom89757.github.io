---
title: PyTorch 优化器
date: 2022-07-20 21:58:23
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下PyTorch中的常用优化器。

<!--more-->

### troch.optim.SGD

其完整调用形式为：

    torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None)

`torch.optim.SGD`有如下的继承关系：

    class SGD(Optimizer):
    class Optimizer(object):
    class object:

该类是对随即梯队下降法的实现（momentum可选）。以下是对随机梯度下降法的简单说明：

![image20220704161015625](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704161015625.png)

Nesterov momentum是基于来自[On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)的公式。

参数：

* `params(iterable)`：用来优化参数的迭代器，或者定义参数组的dicts
* `lr(float)`：学习率
* `momentum(float, optional)`：momentum因子，默认为0
* `weight_decay(float, optional)`：权重衰减（L2惩罚），默认为0
* `dampening(float, optional)`：用来抑制momentum，默认为0
* `nesterov(bool, optional)`：启用Nesterov momentum，默认为False
* `maximize(bool, optional)`：最大化基于the objective的参数，而不是最小化，默认为False
* `foreach(bool, optional)`：whether foreach implementation of optimizer is used，默认为None

调用实例：

    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

可以通过以下方式将从模型net中获得的参数传入优化器：

    base, head = [], []
        for name, param in net.named_parameters():
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

并可通过下述代码访问优化器中的对象和值：

    >>> for key in optimizer.param_groups[1]:
    ...     print(key)
    ... 
    params
    lr
    momentum
    dampening
    weight_decay
    nesterov
    >>> type(optimizer.param_groups[1]['params'])
    <class 'list'>
    >>> len(optimizer.param_groups[0]['params'])
    159
    >>> optimizer.param_groups[1]['lr']
    0.001
    >>> type(optimizer.param_groups[0]['params'][0])
    <class 'torch.nn.parameter.Parameter'>

PS：上述结果中的154正好与前面`base`数组中元素个数相等。

此外，可以通过以下代码直接给优化器添加新的属性（所有的类都可以通过该方式添加属性，也可以通过`setattr`设置属性）：

    optimizer.momentum = momentum # 此前optimizer没有momentum属性
    optimzer.a = 1 # 此前optimizer没有a属性

> 参考资料：
>
> 1. [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
> 2. [Python - Dynamic Class Attributes](https://medium.com/@nschairer/python-dynamic-class-attributes-24a89df8da7d)