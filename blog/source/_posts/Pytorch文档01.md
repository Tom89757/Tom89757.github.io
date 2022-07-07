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

































