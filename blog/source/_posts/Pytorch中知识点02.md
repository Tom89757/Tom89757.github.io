---
title: Pytorch中知识点02
date: 2022-06-03 15:15:43
categories:
- 深度学习
tags:
- Pytorch
- python
---

本文记录一下在实现 [DDRQM](https://link.springer.com/article/10.1007/s11042-016-3392-4) 过程中的一些 Pytorch 框架和 python 相关知识点。

<!--more-->

1.`torch.utils.data.Dataset`：一个表示数据集的抽象类。

其完整形式为：`CLASS torch.utils.data.Dataset(*args, **kwds)`。

所有表示从`keys`到`data samples`的映射的数据集都应该是该抽象类的子集。它的所有子类都应该重写`__getitem__()`方法，从而支持通过`key`获取`data sample`；其子类可以选择重写`__len__()`方法，该方法返回许多通过`Sampler`实现或`Dataloader`默认实现的数据集尺寸。

PS：`Dataloader`默认构造一个生成整数索引的`index sampler`，要想其对一个具有非整数的`indices/keys`的 map-style 的数据集生效，需要提供定制化的`sampler`。

参考资料：

1. [torch.utils.data](https://pytorch.org/docs/stable/data.html)
2. [torch.utils.data 中文文档](https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/)

</br>

2.Creating a Custom Dataset for your files：给自己的文件创建一个定制化的数据集。

一个定制化的数据集必须实现三种函数：`__init__`、`__len__`和`__getitem__`。看一下经典的 FashionMNIST 数据集的实现，我们可以发现图像存储在`img_dir`目录中，labels 存储在一个 CSV 文件`annotation_file`中。下面我们看一下在每个函数中发生了什么：

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

- `__init__`：该函数在实例化数据集对象的时候运行一次，该函数初始化包含图像数据的目录，注释文件和 transforms。`labels.csv`文件格式如下图所示：

```css
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

- `__len__`：该函数返回数据集中的样本数目- `__getitem__`：该函数加载和返回在给定索引`idx`处的一个样本。基于索引，该函数定位在磁盘中图像的位置，通过`read_image`将其转换为一个`tensor`，从`self.img_labels`的 csv 数据中取到对应的 label，如果需要的话在它们身上应用 transform 函数，最后以元组的形式返回 tensor 图像和对应的 label。

> 参考资料：
>
> 1. [Datasets & Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
> 2. [Understanding `__getitem__` method](https://stackoverflow.com/questions/43627405/understanding-getitem-method)

</br>

3.`argparse`：Parser for command-line options, arguments and sub-command.

其源码位于[`Lib/argparse.py`](https://github.com/python/cpython/tree/3.10/Lib/argparse.py)。下面是该 API 的参考信息，`argparse`模块使得写用户友好的命令行界面变得很容易，该程序定义了它要求的 arguments，`argparse`将推算出如何从`sys.argv`中解析出这些 arguments。当用户给出对程序来说无效的 arguments 时`argparse`模块也就自动生成帮助信息和错误信息。下面通过例子来说明：

> 在编程中，arguments 是指在程序、子线程或函数之间传递的值，是包含数据或者代码的独立的 items (表示一个数据单元) 或者 variables。当一个 argument 被用来为一个用户定制化一个程序时，它通常也被称为参数。在 C 语言中，当程序运行时，argc (ARGumentC) 为默认变量，表示被加入到命令行的参数的数量（argument count）。

下面的代码是一个将一系列整数作为输入的程序，并得到它们的和或者最大值：

```python
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```

假设上述代码存入`prog.py`文件。它能够在命令行运行并提供有用的帮助信息：

```bash
$ python prog.py -h
usage: prog.py [-h] [--sum] N [N ...]

Process some integers.

positional arguments:
 N           an integer for the accumulator

options:
 -h, --help  show this help message and exit
 --sum       sum the integers (default: find the max)
```

当从命令行给出有效的 arguments 时，会打印出这些整数的和或者最大值：

```bash
$ python prog.py 1 2 3 4
4

$ python prog.py 1 2 3 4 --sum
10
```

当传入无效的 arguments 时，会生成一个 error：

```bash
$ python prog.py a b c
usage: prog.py [-h] [--sum] N [N ...]
prog.py: error: argument N: invalid int value: 'a'
```

下面对这个例子做详细说明：

1. Creating a parser：第一步使用`argparse`模块创建一个`ArgumentParser`对象

   ```python
   parser = argparse.ArgumentParser(description='Process some integers.')
   ```

   该`ArgumentParser`对象包含将命令行解析为 Python data types 的所有必要的信息。

2. Adding arguments：通过调用`add_argument()`方法向`ArgumentParser`对象填入和程序 arguments 有关的信息。通常来说，这些调用告诉`ArgumentParser`如何取得命令行中的字符串并将其转化为对象。这些信息被存储起来并可以通过调用`parse_args()`来使用，例如：

   ```python
   parser = argparse.ArgumentParser(description='Process some integers.')
   parser.add_argument('integers', metavar='N', type=int, nargs='+',
                       help='an integer for the accumulator')
   parser.add_argument('--sum', dest='accumulate', action='store_const',
                       const=sum, default=max,
                       help='sum the integers (default: find the max)')
   
   args = parser.parse_args()
   ```

   调用`parse_args()`将会返回一个具有两个 attributes ——`integers`和`accumulate`的对象，`integers`属性是一个或多个整数值的列表；`accumulate`是`sum()`或`max()`函数。

3. Parsing arguments：`ArgumentParser`通过`parse_args()`解析 arguments。其过程中会监测命令行，并将每个 argument 转换为合适的 type，然后采取合适的 action。在大多数情况下，这意味着将从命令行解析的 attributes 中创建一个简单的 `Namespace` 对象。

   ```python
   >>> parser.parse_args(['--sum', '7', '-1', '42'])
   Namespace(accumulate=<built-in function sum>, integers=[7, -1, 42])
   ```

更详细的内容可见：[Argparse Tutorial](https://docs.python.org/3/howto/argparse.html)

> 参考资料：
>
> 1. [argparse](https://docs.python.org/3/library/argparse.html)
> 2. [Argparse Tutorial](https://docs.python.org/3/howto/argparse.html)

</br>

4.Reading and Writing Files：读取和写入文件

`open()`返回一个文件对象（file object），该函数通常通过两个 positional arguments 和 一个 keyword argument 进行调用：`open(filename, mode, encoding=None)`。如下图所示：

```python
f = open('workfile', 'w', encoding='utf-8')
```

- 第一个参数表示文件名；
- 第二个参数表示打开文件的模式，`r`表示文件只读，`w`表示文件只写（已存在的同名文件中数据将被擦除），`a`表示在文件内容之后`appending`，写入文件中的数据将被添加到文件最后，`r+`表示文件可同时读和写，模式参数是可选的，默认为`r`
- 第三个参数表示文件的编码格式，正常情况下文件以`text`模式打开，从该文件中读取和写入字符串。当编码格式没有被指定时，默认编码格式是 `platform dependent`，由于 UTF-8 是现行的标准，建议使用该格式。在`text`模式，在读取文件时会将 `platform-specific line endings` 转换为`\n`，在写入文件时则反之。

当处理文件对象时建议使用`with`关键字，其优点在于在操作完成后文件能被合适地关闭，即使异常发生。其也比等价的`try-finally`块更短：

```python
>>> with open('workfile', encoding="utf-8") as f:
...     read_data = f.read()

>>> # We can check that the file has been automatically closed.
>>> f.closed
True
```

> 参考资料：
>
> 1. [Reading and Writing Files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)

</br>

5.`threading.Thread`：多线程。

> 参考资料：
>
> 1. [Python 多线程](https://www.runoob.com/python/python-multithreading.html)

</br>

6.`multiprocessing.Process`：多进程。

> 参考资料：
>
> 1. [想要利用CPU多核资源一Python中多进程（一）](https://developer.51cto.com/article/632081.html)

7.在python文件中包含`from PIL import PILLOW_VERSION`代码时，可能会出现如下报错：

```python
ImportError: cannot import name 'PILLOW_VERSION' from 'PIL' (/storage/FT/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/PIL/__init__.py)
```

其原因在于在较新的pillow版本中`PILLOW_VERSION`已被去除，可以代替使用`__version__`或者安装较老的pillow版本`pip install Pillow==6.1`。

> 参考资料：
>
> 1. [ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'](https://github.com/python-pillow/Pillow/issues/4130)
> 2. [PILLOW_VERSION constant](https://pillow.readthedocs.io/en/stable/releasenotes/7.0.0.html#pillow-version-constant)

</br>

8.Python中的Logging包，在SCWSSOD中的用法示例为：

```python
import logging as logger
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")
...
logger.info(msg)
```

该模块定义了一系列的函数和类，为applications和libraries实现了一个灵活的event logging system。由一个标准的库模块提供logging API的关键好处在于，所有的Python模块都能加入logging，所以application log可以包含自己的信息以及整合来自第三方模块的信息。简单示例为：

```python
>>> import logging
>>> logging.warning('Watch Out!')
WARNING:root:Watch Out!
```

> 参考资料：
>
> 1. [Logging facility for Python](https://docs.python.org/3/library/logging.html)
> 2. [Logging HOWTo](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)

</br>

9.`torch.nn.CrossEntropyLoss`：用来计算input和target之间交叉熵损失的criterion。其完整调用形式为：

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

> 参考资料：
>
> 1. [CROSSENTROPYLOSS](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
> 2. [Loss Functions](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#loss-functions)

</br>

10.`torch.nn.Module`：所有神经网络modules的基本类，所有的模型（包括自定义模型）都应该是该类的子类。Modules也可以包含其它的Module，即允许以树的结构嵌套。能够将子模块赋值给模块属性：

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

































