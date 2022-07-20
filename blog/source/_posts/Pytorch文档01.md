---
title: Pytorch文档01
date: 2022-07-04 15:29:47
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录Pytorch中一些常用类、函数和概念的文档（主要由[官方文档](https://pytorch.org/docs/stable/index.html)翻译而来）和对它们的理解。

<!--more-->

### torch.nn.parameter.Parameter

其完整声明形式为：

```python
CLASS torch.nn.parameter.Parameter(data=None, requires_grad=True)
```

`torch.nn.parameter.Parameter`有如下的继承关系：

```python
class Parameter(torch.Tensor):
class Tensor(torch._C._TensorBase):
class _TensorBase(metaclass=_TensorMeta)
class _TensorMeta(type):
class type:
```

一种Tensor，可以被认为是一个模型参数。

`Parameter`是`Tensor`的子类，当和`Module`一起使用时有一种专门的特性——当被赋值作为模型属性时它们会自动地添加到模型的参数列表中，并且将会出现在`parameter()`迭代器中。给模型属性赋值一个张量时则没有这种作用。这是因为用户可能想存储一些临时状态，像在模型中RNN的最后的隐藏状态。如果没有类似`Parametr`这样的类，这些临时的值也会被注册。

参数：

- `data(Tensor)`：参数张量。
- `requires_grad(bool, optional)`：如果参数要求梯度，看 [Locally disabling gradient computation](https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) 了解更多的细节，默认为True。

> 参考资料：
> 
> 1. [PARAMETER](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

### torch.nn.functional

静态函数库。其包含的函数可分为以下几种类型：

- Convolution function
- Pooling function
- Non-linear activation functions
- Linear functions
- Sparse functions
- Distance functions
- Loss functions
- Vision functions
- DataParallel functions (multi-GPU, distributed)：`torch.nn.parallel.data_parallel`

### torch.nn.init

静态函数库，用于权重等的初始化

#### torch.nn.init.kaiming_normal_

其完整声明形式为：

```python
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

根据 *Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)* 中描述的方法使用正态分布给输入Tensor填充值。结果Tensor的值取样于$N(0, std^2)$。其中：

![image-20220707105811403](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220707105811403.png)

也被称之为何氏初始化（He initialization）

参数：

- `tensor`：一个n维`torch.Tensor`
- `a`：the negative slope of the rectifier used after this layer (only used with `'leaky_relu'`)
- `mode`：`'fan_in'`(默认)或者`'fan_out'`。选择`'fan_in'`在前向传播过程中保存权重方差大小；选择`'fan_out'`在反向传播过程中保存方差大小。
- `nonlinearity`：非线性函数（如`nn.functional`名），建议只使用`'relu'`或`'leaky_relu'`（默认）。

调用实例：

```python
w = torch.empty(3, 5)
nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
```

#### torch.nn.init.zeros_

其完整声明形式为：

```python
torch.nn.init.zeros_(tensor)
```

使用标量0来填充输入Tensor.

调用实例：

```python
>>> w = torch.empty(3, 5)
>>> nn.init.zeros_(w)
```

`torch.nn.init.ones_`与之类似，使用标量1填充输入Tensor。

> 参考资料：
> 
> 1. [TORCH.NN.INIT](https://pytorch.org/docs/stable/nn.init.html)

### torch.nn.functional vs torch.nn

前者使用静态的函数，后者则定义了一个`nn.Module`类。对`nn.Module`类来说，如`nn.Conv2d`，其拥有一些内置的属性如`self.weight`，并不需要传递`weights`和`bias`等参数（模块通常会在其`forward`方法中调用对应的函数）；而对`functional.Conv2d`来说，其只是定义了操作，需要给其传递所有的参数。

所以下述两种关于Conv2D的实现是等价的：

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 6, 3)

    def forward(self, x):
        x = self.conv(x)
        return x
```

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        conv = nn.Conv2d(1, 6, 3)
        self.weight = conv.weight
        self.bias = conv.bias

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias)
        return x
```

但下面的操作将`weights`、`bias`值和`conv2d`操作进行了解耦。

经验上看，由于`relu`等操作不需要像`Conv2d`操作一样需要对参数进行更新，通常在`forward`中直接通过`F.relu`进行调用，而不需要在`__init__`中初始化`nn.ReLU`模块。

这里补充下面第5和第6个参考资料中提到的有关`backward()`的知识：

Pytorch使用计算图来计算backward gradients，计算图会追踪在forward pass中做了哪些操作。任何在一个`Variable`上做的惭怍隐式地被registered。然后就只需要从variable被调用的地方反向穿过计算图根据导数的链式法则来计算梯度。下面是Pytorch中计算图的一个可视化图片：

![image-20220707160538483](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220707160538483.png)

> 参考资料：
> 
> 1. [What is the difference between torch.nn and torch.nn.functional?](https://discuss.pytorch.org/t/what-is-the-difference-between-torch-nn-and-torch-nn-functional/33597)
> 2. [Conv2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
> 3. [functional.conv2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d)
> 4. **[Beginner: Should ReLU/sigmoid be called in the __init__ method?](https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689)**
> 5. [How does PyTorch module do the back prop](https://stackoverflow.com/questions/49594858/how-does-pytorch-module-do-the-back-prop)
> 6. [How Computational Graphs are Constructed in PyTorch](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)

### Pytorch Internals

PS：Linux系统中site-packages路径为`../anaconda3/envs/mobilesal/lib/python3.6/site-packages/`

> 参考资料：
> 
> 1. [A Tour of PyTorch Internals (Part I)](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
> 2. [A Tour of PyTorch Internals (Part II) - The Build System](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)
> 3. [PyTorch – Internal Architecture Tour](https://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/)
> 4. [Where is site-packages located in a Conda environment?](https://stackoverflow.com/questions/31003994/where-is-site-packages-located-in-a-conda-environment)
