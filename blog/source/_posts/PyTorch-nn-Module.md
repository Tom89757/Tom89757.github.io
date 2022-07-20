---
title: PyTorch nn.Module
date: 2022-07-20 21:49:56
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下PyTorch中最核心的类之一——`torch.nn.Module`。

<!--more-->

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

#### nn.Module.forward

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