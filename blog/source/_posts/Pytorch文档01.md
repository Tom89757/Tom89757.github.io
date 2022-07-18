---
title: Pytorch文档01
date: 2022-07-04 15:29:47
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录Pytorch中一些关键类和函数的文档（主要由[官方文档](https://pytorch.org/docs/stable/index.html)翻译而来）和对它们的理解。

<!--more-->

### torch.nn.CrossEntropyLoss

用来计算input和target之间交叉熵损失的criterion。其完整调用形式为：

```python
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
```

当计算C类元素的分类问题时很有用。如果提供`weights`参数，该参数应该是一个1D Tensor，用来为每类元素分配权重，当训练集unbalance时尤其有用。

input应当包含原始的，未归一化的每一类的分数，input应当是一个Tensor，尺寸为unbatched input的尺寸，例如`(minibatch, C)`或者`(minibatch, C, d1, d2, ...., dk)`，k表示k维数据。后者对高维input很有用，例如计算2D图像的每个pixel的交叉熵损失。

该criterion期望的target应该包含以下两种中的一种：

- 范围为$[0, C)$的类索引，C表示类数目；如果指定`ignore_index`参数，该损失也接受不在前述范围内的类索引。`reduction='none'`时，unreduced loss可以被描述为：

  ![image-20220704094749082](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704094749082.png)

  x，y表示input和target。如果reduction is not 'none'（默认为'mean'），则有

  ![image-20220704095008493](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704095008493.png)

- 每一类的概率；当labels超出每个minibatch项要求的单类之外时有用，例如blended labels，label smoothing。unreduced loss（`reduction='none'`）可以描述为：

  ![image-20220704095317040](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704095317040.png)

  如果reduction is not 'none'（默认为'mean'），则有：

  ![image-20220704095420471](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704095420471.png)

PS：该criterion的性能在target包含类索引时通常更好，因为此时可以进行优化计算。只在对minibatch想来说单类label太受限的情况下，才考虑将target作为类的概率。

参数：

- `weights`：Tensor类型，可选择。作用于每个类的manual rescaling weight。如果给出，应该是一个具有尺寸C的Tensor
- `size_average`：bool类型，可选择。不建议使用（deprecated），见reduction。
- `ignore_index`：int类型，可选。指定一个被忽略的target value，该值不对input gradident起作用。当`size_average=True`，损失在non-ignored targets上取平均。注意只有当target包含类索引时`ignore_index`才有用。
- `reduce`：bool类型，可选。deprecated，见reduction。
- `reduction`：string类型，可选。指定应用到output上的reduction：`none`|`mean`|`sum`。
- `label_smoothing`：floate类型，可选。范围为[0.0, 1.0]的float值，指定计算损失时的平滑程度，0.0表示不进行平滑，默认为0.0。通过平滑targets变成了一个原始真值和高斯分布的混合物，见 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

Shape：C表示类数目，N表示batchsize

- Input：(C)，(N,C)，或者(N, C, d1, d2, ..., dk)
- Target：如果包含类索引，形状为()，(N)或者(N, d1, d2, ..., dk)，每个值的范围为[0, C)。如果包含类概率，形状和Input相同，每个值的范围为[0, 1]
- Output：如果reduction='none'，形状和target相同，否则为标量。

调用实例：

```python
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
```

`torch.nn.CrossEntropyLoss()`有如下的继承关系：

```python
class CrossEntropyLoss(_WeightedLoss):
class _WeightedLoss(_Loss):
class _Loss(Module)
```

> 参考资料：
>
> 1. [CROSSENTROPYLOSS](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
> 2. [Loss Functions](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#loss-functions)

### torch.nn.Module

所有神经网络modules的基本类，所有的模型（包括自定义模型）都应该是该类的子类。Modules也可以包含其它的Module，即允许以树的结构嵌套。能够将子模块赋值给模块属性：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

通过上述方式赋值的子模块将被注册，调用`to(device)`时，子模块的参数也会转换为`cuda Tensor`。

PS：在赋值给子模块之前必须调用父类的`__init__()`。

`training`：bool类型，表示模块处于training还是evaluation模式

`add_module(name, module)`：给当前模块添加一个子模块，该子模块可以通过给出的`name`作为属性被获取。

`apply(fn)`：递归地将`fn`应用到每个子模块（通过`.children()`返回）以及自身。典型的使用方式包括初始化模型的参数（see also [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html#nn-init-doc)）

- 参数：`fn(Module->None)`，应用到每个子模块的函数

- 返回：self

- 返回类型：Module

- 调用实例：

  ```python
  @torch.no_grad()
  def init_weights(m):
      print(m)
      if type(m) == nn.Linear:
          m.weight.fill_(1.0)
          print(m.weight)
  net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
  net.apply(init_weights)
  Linear(in_features=2, out_features=2, bias=True)
  Parameter containing:
  tensor([[ 1.,  1.],
          [ 1.,  1.]])
  Linear(in_features=2, out_features=2, bias=True)
  Parameter containing:
  tensor([[ 1.,  1.],
          [ 1.,  1.]])
  Sequential(
    (0): Linear(in_features=2, out_features=2, bias=True)
    (1): Linear(in_features=2, out_features=2, bias=True)
  )
  Sequential(
    (0): Linear(in_features=2, out_features=2, bias=True)
    (1): Linear(in_features=2, out_features=2, bias=True)
  )
  ```

`buffers(recurse=True)`：返回module buffers的迭代器。

- 参数：`recurse(bool)`，如果为True，返回该模块和所有子模块的buffers迭代器；如果为False，只返回该模块直接成员的buffers迭代器

- Yields：torch.Tensor，一个模型的buffer

- 调用实例：

  ```python
  >>> for buf in model.buffers():
  >>>     print(type(buf), buf.size())
  <class 'torch.Tensor'> (20L,)
  <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
  ```

`children()`：返回直接的子模块的迭代器。

- Yields：Module类型，一个子模块

`cpu()`：将所有的模型参数和buffers转移到CPU。该方法modify the module in-place。

- Returns：self
- Return type：Module

`cuda(device=None)`：将所有的模型参数和buffers转移到GPU。这也使得相关联的参数和buffers为不同的对象。所以如果模块在优化期间保留在GPU上时，该函数应在在构造优化器之前被调用。该方法modify the module in-place。

- 参数：`device(int, optional)`，如果指定的话，所有的参数将被复制到该device。
- Returns：self
- Return type：Module

`eval()`：设置module到evaluation模式。它和`self.train(False)`等价，可以在 [Localling disabling gradient computation](https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) 看到其和一些相似机制的比较

- Returns：self
- Return type：Module

`forward(*input)`：定义在每次调用它时进行的计算，应该被所有的子类覆盖。

- PS：不太懂，尽管forward操作定义在该函数内部，但是应该调用Module instance而不是该函数，因为前者会考虑registered hooks而后者不会

`get_parameter(target)`：返回由target给出的参数，如果存在的话，否则抛出error。

- 参数：`target`，fully-qualified的对应参数的字符串名
- Returns：target引用的参数
- Return type：torch.nn.Parameter
- Raises：AttributeError

`load_state_dict(state_dict, strict=True)`：将来自`state_dict`的参数和buffers复制到该模块和它的子模块。如果`strict=True`，那么`state_dict`的keys必须和该模块`state_dict()`函数返回的keys完全匹配。

- 参数：
  - `state_dict(dict)`：一个包含参数和永久buffers的dict
  - `strict(bool, optional)`：使用严格限制keys的匹配，默认为True。

- Returns：missing_keys，一个包含missing keys的字符串列表；unexpected_keys，一个包含unexpected_keys的字符串列表
- Return type：具有`missing_keys`和`unexpected_keys`名的元组
- PS：如果一个参数或buffer registered as None并且它对应的key存在在`state_dict`中，调用`load_state_dict()`会raise `RuntimeError`。

`modules()`：返回在网络中的所有模块的迭代器

- Yields：Module，网络中的模块

- PS：重复的模块只返回一次。在下面的例子中，`l`只返回一次

- 调用实例：

  ```python
  >>> l = nn.Linear(2, 2)
  >>> net = nn.Sequential(l, l)
  >>> for idx, m in enumerate(net.modules()):
          print(idx, '->', m)
  
  0 -> Sequential(
    (0): Linear(in_features=2, out_features=2, bias=True)
    (1): Linear(in_features=2, out_features=2, bias=True)
  )
  1 -> Linear(in_features=2, out_features=2, bias=True)
  ```

`named_children()`：返回之间子模块的迭代器，生成模块名和模块自身。

- Yields：(string, Module)，包含一个模块名和子模块的元组

- 调用实例：

  ```python
  for name, module in model.named_children():
      if name in ['conv4', 'conv5']:
          print(module)
  ```

`named_parameters(prefix='', recurse=True)`：返回模块参数的迭代器，生成参数名和参数。

- 参数：
  - `prefix(str)`：添加到所有参数名之前的前缀
  - `recurse(bool)`：如果为True，生成该模块和所有此模块的参数。

可以通过以下方式获取`named_parameters`的参数名和参数值：

```python
base_name, base_value = [], []
for name, param in net.named_parameters():
        if 'bkbone' in name:
        	base_name.append(name)
            base_value.append(param)
```

得到的部分参数名示例如下：

```python
bkbone.conv1.weight
bkbone.bn1.weight
bkbone.bn1.bias
bkbone.layer1.0.conv1.weight
bkbone.layer1.0.bn1.weight
bkbone.layer1.0.bn1.bias
bkbone.layer1.0.conv2.weight
bkbone.layer1.0.bn2.weight
bkbone.layer1.0.bn2.bias
bkbone.layer1.0.conv3.weight
bkbone.layer1.0.bn3.weight
bkbone.layer1.0.bn3.bias
bkbone.layer1.0.downsample.0.weight
bkbone.layer1.0.downsample.1.weight
bkbone.layer1.0.downsample.1.bias
```

`parameters(recurse=True)`：返回模块参数的迭代器，通常传给一个优化器。

- 参数：`recurse(bool)`，如果为True，迭代所有的模块和子模块。否则，只迭代该模块的直接成员参数。

- Yields：Parameter，模块参数

- 调用实例：

  ```python
  >>> for param in model.parameters():
  >>>     print(type(param), param.size())
  <class 'torch.Tensor'> (20L,)
  <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
  ```

`state_dict(**args*, *destination=None*, *prefix=''*, *keep_vars=False*)`：返回包含模块整个状态的字典。参数和永久性buffers(例如running averages)被包含，Keys是相应的参数和buffer名。如果不包含参数和buffers设置为None。

- Warning：目前`state_dict()`也按顺序接受位置参数`destination`，`prefix`和`keep_vars`，它们是deprecated；此外，避免参数`destination`的使用，它不是为用户设计的。

- 参数：

  - `destination(dict, optional)`：略。
  - `prefix(str, optional)`：略
  - `key_vars(bool, optional)`：略

- Returns：一个包含模块整个状态的字典

- Return type：dict

- 调用实例：

  ```python
  >>> module.state_dict().keys()
  ['bias', 'weight']
  ```

`to(*args, **kwargs)`：移动参数和buffers或者cast它们到指定类型。能够以以下的形式调用：

```python
to(device=None, dtype=None, non_blocking=False)
to(dtype, non_blocking=False)
to(tensor, non_blocking=False)
to(memory_format=torch.channels_last)
```

其特性和`torch.Tensor.to()`类似，但是只接受浮点或符合类型的`dtype`。

- PS：该方法 modify the modules in-place

- 参数：

  - `device（torch.device)`：在模块中参数和buffers所需的device
  - `dtype(torch.dtype)`：在模块中参数和buffers所需的浮点或复合 dtype
  - `tensor(torch.Tensor)`：对在模块中的所有参数和buffers来说该tensor的dtype和device是它们所需的。
  - `memory_format(torch.memory_format)`：略

- Returns：self

- Return type：Module

- 调用实例：

  ```python
  >>> linear = nn.Linear(2, 2)
  >>> linear.weight
  Parameter containing:
  tensor([[ 0.1913, -0.3420],
          [-0.5113, -0.2325]])
  >>> linear.to(torch.double)
  Linear(in_features=2, out_features=2, bias=True)
  >>> linear.weight
  Parameter containing:
  tensor([[ 0.1913, -0.3420],
          [-0.5113, -0.2325]], dtype=torch.float64)
  >>> gpu1 = torch.device("cuda:1")
  >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
  Linear(in_features=2, out_features=2, bias=True)
  >>> linear.weight
  Parameter containing:
  tensor([[ 0.1914, -0.3420],
          [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
  >>> cpu = torch.device("cpu")
  >>> linear.to(cpu)
  Linear(in_features=2, out_features=2, bias=True)
  >>> linear.weight
  Parameter containing:
  tensor([[ 0.1914, -0.3420],
          [-0.5112, -0.2324]], dtype=torch.float16)
  
  >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
  >>> linear.weight
  Parameter containing:
  tensor([[ 0.3741+0.j,  0.2382+0.j],
          [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
  >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
  tensor([[0.6122+0.j, 0.1150+0.j],
          [0.6122+0.j, 0.1150+0.j],
          [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
  ```

`train(mode=True)`：设置模块为training mode。

- 参数：`mode(bool)`，是否设置training mode还是evaluation mode，默认为True
- Returns：self
- Return type：Module

`type(dst_type)`：将所有参数和buffers转换为`dst_type`。该方法modify the module in-place。

- 参数：`dst_type(type or string)`，目标类型
- Returns：self
- Return type：Module

`zero_grad(set_to_none=False)`：设置所有模型参数的梯度为0。可以看 [torch.optim.Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) 下相似的函数获得更多信息。

- 参数：`set_to_none(bool)`：设置为None而不是0，可以看 [torch.optim.Optimizer.zero_grad](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad) 了解更多细节

> 参考资料：
>
> 1. [MODULE](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
> 2. [class torch.nn.Module](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#containers)

### troch.optim.SGD

其完整调用形式为：

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None)
```

`torch.optim.SGD`有如下的继承关系：

```python
class SGD(Optimizer):
class Optimizer(object):
class object:
```

该类是对随即梯队下降法的实现（momentum可选）。以下是对随机梯度下降法的简单说明：

![image-20220704161015625](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220704161015625.png)

Nesterov momentum是基于来自[On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)的公式。

参数：

- `params(iterable)`：用来优化参数的迭代器，或者定义参数组的dicts
- `lr(float)`：学习率
- `momentum(float, optional)`：momentum因子，默认为0
- `weight_decay(float, optional)`：权重衰减（L2惩罚），默认为0
- `dampening(float, optional)`：用来抑制momentum，默认为0
- `nesterov(bool, optional)`：启用Nesterov momentum，默认为False
- `maximize(bool, optional)`：最大化基于the objective的参数，而不是最小化，默认为False
- `foreach(bool, optional)`：whether foreach implementation of optimizer is used，默认为None

调用实例：

```python
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
```

可以通过以下方式将从模型net中获得的参数传入优化器：

```python
base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
```

并可通过下述代码访问优化器中的对象和值：

```python
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
```

PS：上述结果中的154正好与前面`base`数组中元素个数相等。

此外，可以通过以下代码直接给优化器添加新的属性（所有的类都可以通过该方式添加属性，也可以通过`setattr`设置属性）：

```python
optimizer.momentum = momentum # 此前optimizer没有momentum属性
optimzer.a = 1 # 此前optimizer没有a属性
```

> 参考资料：
>
> 1. [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
> 1. [Python - Dynamic Class Attributes](https://medium.com/@nschairer/python-dynamic-class-attributes-24a89df8da7d)

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

#### torch.nn.functional.interpolate

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

### saving and loading models

本文档提供了对Pytorch models进行存储和加载的不同使用场景的解决方案。当谈到存储和加载模型，有三个核心函数很相似：

- `torch.save`：存储一个serialized object到磁盘，该函数使用Python的`pickle` utility来序列化（serialization）。Models/tensors和各种类型对象的字典都可以使用该函数存储
- `torch.load`：使用`pickle`的unpickling能力来反序列化pickled对象文件到内存中。该函数也可以设置用来加载数据的设备（如gpu），见 [Saving & Loading Model Across Devices](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)。
- `torch.nn.Module.load_state_dict`：使用反序列话的state_dict加载模型的参数字典，详细信息见 [What is a state_dict?](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)。

**什么是`state_dict`?**

在Pytorch中， 一个`torch.nn.Module`模型的可学习的参数（如权重和偏差）被包含在模型的参数中（可以通过`model.parameters()`获取。一个state_dict就是一个简单的Python字典对象，其将每个layer映射到它的参数tensor。注意只有具有可学习参数的layers（如卷积层，线性层等）和具有registered buffers（batchnorm's running_mean）的layers在模型的state_dict中有入口。Optimizer对象（`torch.optim`）也有一个state_dict，它包含关于优化器的状态信息和使用的超参数。

因为state_dict是Python字典，所以它们可以很容易地存储、更新、更变和恢复，这使得Pytorch的模型和优化器得以模块化。

**Example**

下面看一下 [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) 教程中一个简单的分类器的state_dict：

```python
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

其输出为：

```python
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```

**存储和加载模型用于推断**

存储/加载`state_dict`（建议）

Save：`torch.save(model.state_dict(), PATH)`

Load：

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

PS：PyTorch1.6版本将`torch.save`的存储格式转换为了一个新的基于zipfile的文件格式。`torch.load`仍然保持加载老的pth/pt格式文件的能力。如果想要使用`torch.save`存储老的文件格式pth/pt，可以使用参数 `_use_new_zipfile_serialization=False`。

当加载一个模型用于推断时，只有必要存储训练模型的可学习的参数。使用`torch.save()`存储模型的state_dict将对以后恢复模型给出最大的灵活性，这也是推荐它存储模型的原因。

一个PyTorch的惯例是使用pt/pth扩展名来存储模型。

记住在进行推断之前你必须调用`model.eval()`来设置dropout和batch normalization层来评估模型，不做这一步将导致生成不一致的推断结果。

PS：注意`load_state_dict()`函数将一个字典对象而不是一个存储对象的路径作为参数，这意味着在将state_dict传给该函数之前必须对其反序列化，例如，不能加载模型通过`model.load_state_dict(PATH)`。

PS：如果逆想要保存性能最好的模型（根据获得的验证损失），不要忘记`best_model_state=model.state_dict()`返回的是对state的引用而不是它的copy。你必须序列化`best_model_state`或者使用 `best_model_state = deepcopy(model.state_dict())` 否则你的`best_model_state`将会随着后续的训练迭代继续更新。结果，最终的模型state可能是一个过拟合模型的state。

**存储和加载模型**

Save：`torch.save(model, PATH)`

Load：

```python
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```

上述的存储/加载过程使用最直观的语法，涉及最少的代码。以这种方式存储模型将使用Python的pickle模块存储整个模型。该方法的缺点在于序列化的数据和特定的类以及当模型存储时的目录结构绑定。其原因在于pickle不存储模型类本身，而是存储一个包含该类的文件的路径，这个类会在加载时用到。因为这个原因，你的代码在其他的项目或者在重构后中使用可能会以多种形式中断。

一个PyTorch的惯例是使用pt/pth扩展名来存储模型。

记住在进行推断之前你必须调用`model.eval()`来设置dropout和batch normalization层来评估模型，不做这一步将导致生成不一致的推断结果。



> 参考资料：
>
> 1. [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

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

### torch.nn.Conv2d

对由多个input planes组成的input signal进行二维卷积。其完整声明形式为：

```python
CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

在最简单的样例中，input size为$(N, Cin, H, W)$的层的输出值和输出$(N, Cout, Hout, Wout)$可以精确地描述为：

![image-20220718203916534](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220718203916534.png)

其中*表示有效的2D cross-correlation 操作，N表示batch size，C表示通道数，H和W分别表示像素高宽。

> 参考资料：
>
> 1. [CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

### torch.nn.BatchNorm2d



### torch.nn.GroupNorm



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

### torch.nn.Sequential



### Pytorch Internals

PS：Linux系统中site-packages路径为`../anaconda3/envs/mobilesal/lib/python3.6/site-packages/`

> 参考资料：
>
> 1. [A Tour of PyTorch Internals (Part I)](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
> 2. [A Tour of PyTorch Internals (Part II) - The Build System](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)
> 3. [PyTorch – Internal Architecture Tour](https://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/)
> 4. [Where is site-packages located in a Conda environment?](https://stackoverflow.com/questions/31003994/where-is-site-packages-located-in-a-conda-environment)

### nn.Module.forward

为什么在下述代码中通过`model(input)`获取`output`，而不是`model.forward(input)`。

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

if __name__ == '__main__':
	model = MyModel()
	print(model)
	input=torch.randn(1, 1, 32, 32)
	output = model(input)
```

当直接调用model时，会调用内置的`__call__`函数。在代码 [the code](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L471) 中可以看到，该函数会管理所有的registered hooks并且在之后调用`forward`。这也是应该直接调用model的原因，因为可能hooks可能不会work。

> 参考资料：
>
> 1. [About the ‘nn.Module.forward’](https://discuss.pytorch.org/t/about-the-nn-module-forward/20858)
> 2. [NEURAL NETWORKS](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network)

### nn.Upsample

对一个给定的多通道 1D (temporal), 2D (spatial) 或 3D (volumetric) 数据进行shang'cai'y



























