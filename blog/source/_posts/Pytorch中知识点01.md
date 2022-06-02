---
title: Pytorch中知识点01
date: 2022-05-16 15:48:09
categories:
- 深度学习
tags:
- Pytorch
- python
---

本文记录一下在调试 [MobileSal](https://github.com/yuhuan-wu/MobileSal) 过程中的一些 Pytorch 框架和 python 相关知识点。

<!--more-->

1.当出现`import cv2`，需要安装 `opencv`包：`pip install opencv-python`。

2.`import torch.utils.data` 。该 package 的核心类为 `torch.utils.data.DataLoader`，表示在一个数据集上的迭代，其支持：

- map-style 和 iterable-style 的数据集
- 定制化数据加载顺序
- 自动 batching
- 单线程和多线程的数据加载
- 自动内存 pinning (固定)

其通过以下的 `DataLoader` 对象的构造器配置：

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

3.

```python
def __getitem__(self, key):
    return self.list[key]
```

当在对象中实现以上方法时，使得可以通过索引访问对象中的元素，如 `list[key]`

4.`img = cv.imread(filename[, flags])`根据一个文件名中加载图像然后返回它。如果图像文件不能读取，返回一个空矩阵。可以通过 `img.shape[0]` 和 `img.shape[1]`访问图像的高和宽。

5.`import torch.nn.functional as F`。该代码导入 `torch.nn.functional`，其包含许多对神经网络层进行操作的函数

6.`utils`在编程语言中通常为 Utility Class 的缩写，也被称为 Helper class，是一个只包含静态方法的类，无状态且不能被实例化。

7.`img = cv2.resize(img, (self.weight, self.height))`调用 OpenCV 中的`resize()`方法将源文件 `img` 尺寸转换为所需尺寸 `(self.weight, self.height)`。其详细声明如下：

```python
cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
```

`image interpolation`  发生在对图像进行 resize 时，上述的`interpolation`用于指定不同的插值策略，从而在进行 resize 时按指定策略用已知像素点的值评估未知点的值。

8.

```python
#/usr/bin/env python
class Test:
    def __init__(self,a):
        self.a = a
    def __call__(self,b):
        c = self.a+b
        print c
    def display(self):
        print self.a

test = Test("This is test!")
test.display()
Test("##Append something")
```

`__call__` 用于将一个类重载，使得类也能像函数一样调用。

9.`isinstance()`函数用于判断一个对象是否是一个已知类型，类似 `type()`。

```python
>>>a = 2
>>> isinstance (a,int)
True
>>> isinstance (a,str)
False
>>> isinstance (a,(str,int,list))    # 是元组中的一个返回 True
True
```

`type()` 和 `isinstance()`的区别：

```python
class A:
    pass
 
class B(A):
    pass
 
isinstance(A(), A)    # returns True
type(A()) == A        # returns True
isinstance(B(), A)    # returns True
type(B()) == A        # returns False
```

10.`img = cv2.flip(img, 0)` 表示将图像做水平翻转；`img = cv2.flip(img, 1)`表示将图像做垂直翻转。`img = cv2.flip(img, -1)` 表示将图像同时做水平和垂直翻转。

11.`random.random()`方法返回随机生成的一个实数，范围为 [0,1)；`random.randint(a,b)`方法返回随机生成的一个整数，其值>=a，<=b。

12.

```python
label_tensor =  torch.LongTensor(np.array(label, dtype=np.int)).unsqueeze(dim=0)
```

Torch 定义了10种 tensor 类型，其中 `torch.LongTensor` 为64-bit integer (signed) 对应CPU的类型，在GPU中对应的类型为 `torch.cuda.LongTensor`。`torch.Tensor`是默认的张量类型 `torch.FloatTensor`。

13.`torch.unsqueeze(input, dim)`返回一个新张量，一个维度被插入指定的位置。例如：

```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

14.`torch.tensor` vs `torch.Tensor`：

```python
>>> torch.Tensor([1,2,3]).dtype
torch.float32
>>> torch.tensor([1, 2, 3]).dtype
Out[32]: torch.int64
>>> torch.Tensor([True, False]).dtype
torch.float32
>>> torch.tensor([True, False]).dtype
torch.uint8
>>> torch.Tensor(10)
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
>>> torch.tensor(10)
tensor(10)
```

15.补充关于 `torch.tensor` 的说明，以下为对其调用的语法，其返回一个张量：

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

以下为其使用的例子：

```python
torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])

torch.tensor([0, 1])  # Type inference on data

torch.tensor([[0.11111, 0.222222, 0.3333333]],
             dtype=torch.float64,
             device=torch.device('cuda:0'))  # creates a double tensor on a CUDA device

torch.tensor(3.14159)  # Create a zero-dimensional (scalar) tensor

torch.tensor([])  # Create an empty tensor (of size (0,))
```



PS：当操作tensors推荐使用 `torch.Tensor.clone()`和`torch.Tensor.detach()`以及`torch.Tensor.requires_grad_()`。当t为一个tensor时，下述操作等价：

- `t.clone().detach()` 和 `torch.tensor(t)`
- `t.clone().detach().requires_grad_(True)`和`torch.tensor(t, requires_grad=True)`

16.`torch.rand`返回一个张量，用符合N(0,1)正态分布的随机数字填充，张量形状由`size`决定：

```python
torch.rand(size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

17.`torch.from_numpy(ndarray)`从`numpy.ndarray`创建一个张量。如下：

```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])
```

18.

```python
import sys
sys.path.insert(0, '.')
```

上述命令将当前目录添加到python运行时的查找路径的最开始，使得当python运行时首先在当前目录文件中查找函数或类。

19.Pytorch Hub 是一个预训练模型仓库，用来设计便于研究重现。Pytorch Hub支持发布预训练模型（模型定义和预训练权重）到一个github仓库，只需通过添加一个 `hubconf.py`文件。预训练权重可以被存储在github仓库，也可以通过 `torch.hub.load_state_dict_from_url()`加载。

20.

```python
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
```

其中 `super(InvertedResidual, self).__init__()` 等价于 `super().__init__()`。

21.`self.modules()`返回定义在模型类中对多层模块的迭代器：

```python
# weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
```

上述代码对模型权重进行初始化。

22.`enumerate()`用于将一个可遍历的数据对象（如对象、元组或字符串）组合为一个索引序列，同时列出数据和数据下标，一般用在for循环中：`enumerate(sequence, [start=0])`，其返回一个enumerate(枚举)对象：

```python
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

23.`*args`用于传入多个无名参数，这些参数以tuple的形式导入，一般放在参数列表的最后。若不放在最后，需要指明其他传入参数名称：

```python
def print_func(x, y, *args):
    print(type(x))
    print(x)
    print(y)
    print(type(args))
    print(args)

print_func(1, 2, '呵呵哒', [], x='x', y='y')
```

24.`**kwargs`将参数以字典的形式导入：

```python
def foo(a, b=10, *args, **kwargs):
    print (a)
    print (b)
    print (args)
    print (kwargs)
foo(1, 2, 3, 4, e=5, f=6, g=7)
#输出结果
1
2
(3, 4)
{'e': 5, 'f': 6, 'g': 7}
```

25.Pytorch中的`state_dict`简单来说是一个Python 字典对象，将模型每一层映射到模型参数张量。只有具有可学习参数（卷积层，线性层等）和 registered buffers (batchnorm’s running_mean) 在 `state_dict`中有入口。同理，可以使用`load_state_dict`加载这些参数。

PS：优化器对象`torch.optim`也有`state_dict`，包含关于优化器状态的信息以及所有的超参数。

26.`self.register_buffer` 和`self.register_parameter`。前者用于register模型参数为buffers，buffers不能调用`model.parameters()`返回参数，优化器也无法更新它们。

27.`__repr__`用于返回传入对象相关信息，可以重写。

```python
def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s
```

28.`torch.no_grad`用来关闭梯度计算，当你确定你不会调用 `Tensor.backward()`。这一操作可以减少用于计算的内存消耗，此时`requires_grad=True`。

29.装饰器（Decorators）是用于修改其他函数的功能的函数。python装饰器封装一个函数，并且用这样或那样的方式修改它的行为：

```python
def a_new_decorator(a_func):
 
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
 
        a_func()
 
        print("I am doing some boring work after executing a_func()")
 
    return wrapTheFunction
 
def a_function_requiring_decoration():
    print("I am the function which needs some decoration to remove my foul smell")
 
a_function_requiring_decoration()
#outputs: "I am the function which needs some decoration to remove my foul smell"
 
a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
#now a_function_requiring_decoration is wrapped by wrapTheFunction()
 
a_function_requiring_decoration()
#outputs:I am doing some boring work before executing a_func()
#        I am the function which needs some decoration to remove my foul smell
#        I am doing some boring work after executing a_func()
```

上述代码中 `a_new_decorator` 即为一个装饰器，可以通过`@a_new_decorator`来精简上述代码：

```python
@a_new_decorator
def a_function_requiring_decoration():
    """Hey you! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")
 
a_function_requiring_decoration()
#outputs: I am doing some boring work before executing a_func()
#         I am the function which needs some decoration to remove my foul smell
#         I am doing some boring work after executing a_func()
 
#the @a_new_decorator is just a short way of saying:
a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
```

此时会有一个问题，对用装饰器修饰的函数调用`__name__`会有以下输出：

```python
print(a_function_requiring_decoration.__name__)
# Output: wrapTheFunction
```

此时，可以通过Python提供的`functools.wraps`来解决上述问题：

```python
from functools import wraps
 
def a_new_decorator(a_func):
    @wraps(a_func)
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        a_func()
        print("I am doing some boring work after executing a_func()")
    return wrapTheFunction
 
@a_new_decorator
def a_function_requiring_decoration():
    """Hey yo! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")
 
print(a_function_requiring_decoration.__name__)
# Output: a_function_requiring_decoration
```

30.当需要线上加载模型时，不同版本的Pytorch有不同的加载方法，具体如下：

```python
from torch import nn
import torch
try:
    from torchvision.models.utils import load_state_dict_from_url # torchvision 0.4+
except ModuleNotFoundError:
    try:
        from torch.hub import load_state_dict_from_url # torch 1.x
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url # torch 0.4.1
```

PS：当版本不同时有不同的异常，如上述的`ModuleNotFoundError`和`ImportError`。

























