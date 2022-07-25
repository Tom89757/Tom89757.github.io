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
> 2. [多进程multiprocess](https://www.liujiangblog.com/course/python/82)

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

9.在Pytorch中register意味着什么？

在pytorch文档和方法名中register意味着“在一个官方的列表中记录一个名字或者信息的行为”。

例如，`register_backward_hook(hook)`将函数`hook`添加到一个其他函数的列表中，`nn.Module`会在`forward`过程中执行这些函数。

与之相似，`register_parameter(name, param)`添加一个`nn.Parameter`类型的名为`name`的参数`param`到`nn.Module`的可训练参数的列表之中。register可训练参数很关键，这样pytorch才会知道那些tensors传送给优化器，那些tensors作为`nn.Module`的state_dict存储。

> 参考资料：
>
> 1. [What do we mean by 'register' in PyTorch?](https://stackoverflow.com/questions/68463009/what-do-we-mean-by-register-in-pytorch)

10.Pytorch、CUDA版本与显卡驱动版本对应关系：

- CUDA驱动和CUDAToolkit对应版本

  ![image-20220717110226488](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220717110226488.png)

- Pytorch和cudatoolkit版本

  | cuda和pytorch版本        | 安装命令                                                     |
  | ------------------------ | ------------------------------------------------------------ |
  | cuda==10.1 pytorch=1.7.1 | `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch` |
  | cuda==10.1 pytorch=1.7.0 | `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch` |
  | cuda==10.1 pytorch=1.6.0 | `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch` |
  | cuda==10.1 pytorch=1.5.1 | `conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch` |
  | cuda==10.1 pytorch=1.5.0 | `conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch` |
  | cuda==10.1 pytorch=1.4.0 | `conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch` |

> 参考资料：
>
> 1. [pytorch版本，cuda版本，系统cuda版本查询和对应关系](https://www.cnblogs.com/Wanggcong/p/12625540.html)
> 2. [INSTALLING PREVIOUS VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/)
> 3. [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

</br>

11.以如下目录组织文件：

```python
/model
|_ vgg.py
|_ vgg_models.py
test.py
```

如果`test.py`文件中包含对`vgg_models.py`的依赖：`from model.vgg_models import Back_VGG`

同时，`vgg_models.py`又包含对`vgg.py`的依赖：`from vgg import B2_VGG`。

运行`python test.py`可能会出现如下报错：

![image-20220719095058661](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220719095058661.png)

这是由于运行`test.py`时将当前目录`./`作为导入包时的本地查找路径，`vgg_models.py`在导入包时只会在`./`中查找，而不会在`./model/`中查找，导致找不到包。此时可以通过在`test.py`开头添加如下代码把`./model/`添加为查找路径来解决该问题：

```python
import sys
sys.path.insert(0, './model')
```

> 参考资料：
>
> 1. [import error: 'No module named' *does* exist](https://stackoverflow.com/questions/23417941/import-error-no-module-named-does-exist)

</br>

12.使用`cv2.imwrite`写入文件时，可能会出现如下问题：

![image-20220719095819493](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220719095819493.png)

这是由于存入路径`save_path+name`无文件扩展名，可以通过在`name`后添加`.png`扩展名解决。

> 参考资料：
>
> 1. [cv::imwrite could not find a writer for the specified extension](https://stackoverflow.com/questions/9868963/cvimwrite-could-not-find-a-writer-for-the-specified-extension)

</br>

13.当使用如下代码进行权重初始化时：

```python
def _initialize_weights(self, pre_train):
        keys = pre_train.keys()
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
```

可能会出现以下报错：

![image-20220719100227462](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220719100227462.png)

这是由于在Python2中`Class collections.OrderedDict`的`keys()`属性返回的是一个`list`，而在Python3中其返回一个`odict_keys`，此时可以通过将`odict_keys`转换为`list`解决该问题：

```python
def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
```

> 参考资料：
>
> 1. []'odict_keys' object does not support indexing #1](https://github.com/taehoonlee/tensornets/issues/1)

</br>

14.为什么在Pytorch中通常使用`PIL` (即PILLOW) 包，而不是`cv2` (即opencv)。有以下几个原因：

- OpenCV2以BGR的形式加载图片，可能需要包装类在内部将其转换为RGB
- 会导致在`torchvision`中的用于transforms的`functional`的代码重复，因为许多`functional`使用PIL的操作实现
- OpenCV加载图片为`np.array`，在arrays上做transformations并没有那么容易
- PIL和OpenCV对图像不同的表示可能会导致用户很难捕捉到bugs
- Pytorch的modelzoo也依赖于RGB格式，它们想要很容易地支持RGB格式



> 参考资料：
>
> 1. [Why is PIL used so often with Pytorch?](https://stackoverflow.com/questions/61346009/why-is-pil-used-so-often-with-pytorch)
> 2. [OpenCV transforms with tests #34](https://github.com/pytorch/vision/pull/34)
> 3. [I wonder why Pytorch uses PIL not the cv2](https://discuss.pytorch.org/t/i-wonder-why-pytorch-uses-pil-not-the-cv2/19482)

</br>

15.在加载模型权重进行测试时，可能会出现如下报错：

```python
Missing keys & unexpected keys in state_dict when loading self trained model
```

其原因可能在于在训练模型时使用了`nn.DataParallel`，因此存储的模型权重和不使用前者时的权重的keys有所不同。其解决方法为，在创建模型时同样用`nn.DataParallel`进行包装：

```python
# Network
self.model = TRACER(args).to(self.device)
if args.multi_gpu:
	self.model = nn.DataParallel(self.model).to(self.device)
```

也可以直接去除`.module`key：

```python
check_point = torch.load('myfile.pth.tar')
check_point.key()


from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v

model.load_state_dict(new_state_dict)
```

> 参考资料：
>
> 1. [Missing keys & unexpected keys in state_dict when loading self trained model](https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379)
> 2. [[solved] KeyError: ‘unexpected key “module.encoder.embedding.weight” in state_dict’](https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686)

</br>











