---
title: PyTorch TensorBoard
date: 2022-08-06 15:12:52
categories:
- 深度学习
tags:
- Pytorch
- 文档

---

本文记录一下PyTorch中工具类`torch.utils.tensorboard`的使用教程。

<!--more-->

在进一步学习之前，更多关于TensorBoard的细节见 https://www.tensorflow.org/tensorboard/。

一旦安装了TensorBoard，这些utilities让我们能够将models和metrics记录到目录中，用于在TensorBoard UI中的可视化。对PyTorch models和tensors（以及Caffe2 nets和blobs）支持对Scalars、images、histograms、graphs和embedding的可视化。

SummaryWriter类是通过TensorBoard进行log data的利用和可视化的主要入口。例如：

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default, change to ./mnist/
writer = SummaryWriter("mnist")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
```

- 运行`tensorboard --logdir=mnist`：

  ![image-20220806154817790](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806154817790.png)

- 打开`http://localhost:6006/`，可以看到如下结果：

  ![image-20220806154841644](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806154841644.png)

对一次实验可以记录很多信息。为了避免UI混乱和得到更好的结果聚类，我们eyi通过分层地命名来对plots进行分组。例如，"Loss/train"和"Loss/test"可以放在同一组，"Accuracy/train"和"Accuracy/test"可以放在TensorBoard界面的另一组：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Writer outputs to ./loss/ directory
writer = SummaryWriter("loss")

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

- 运行`tensorboard --logdir=mnist`

- 打开`http://localhost:6006/`，可以看到如下结果：

  ![image-20220806155433060](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806155433060.png)

下面是Summary类的完整声明：

```python
CLASS torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
```

将entry直接写入log_dir中的event文件，该文件可以直接被TensorBoard读取。

SummaryWriter提供高级API来在给定目录中创建一个event文件，并且向其中添加summaries和events。该类异步地更新文件内容。这允许一个训练程序调用方法直接从训练loop中向这个文件添加数据，而不会拖慢训练速度。

**其初始化代码为**：

```python
__init__(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
```

参数：

- `log_dir(string)`：存储文件路径，默认为`runs/CURRENT_DATETIME_HOSTNAME`，会在每一次run之后更改。使用分层的文件夹结构可以很容易地在runs之间比较。例如，传入`runs/exp1`，`runs/exp2`等，对每个新的实验在它们之间进行比较。
- `comment(string)`：加载默认`log_dir`后面的后缀，如果`log_dir`被赋值，该参数不起作用。
- `purge_step(int)`：当logging在第T+X步崩溃，然后在第T步重启时，任何global_step大于或等于T的events将被清除。注意crashed和resumed的实验应该有相同的`log_dir`
- `max_queue(int)`：在通过调用`add`刷新磁盘之前pending的events和summaries的队列尺寸。默认为10个。
- `flush_secs(int)`：刷新pending的events和summaries到磁盘的频率，默认为每两分钟。
- `filename_suffix(string)`：在log_dir目录中添加到所有event文件名后面的后缀。更多的关于文件名构建的细节见`tensorboard.summary.writer.event_file_writer.EventFileWriter`。

例如：

```python
from torch.utils.tensorboard import SummaryWriter

# create a summary writer with automatically generated folder name.
writer = SummaryWriter()
# folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

# create a summary writer using the specified folder name.
writer = SummaryWriter("my_experiment")
# folder location: my_experiment

# create a summary writer with comment appended.
writer = SummaryWriter(comment="LR_0.1_BATCH_16")
# folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
```

**添加scalar数据到summary**：

```python
add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
```

参数：

- `tag(string)`：数据标识器（identifier）
- `scalar_value(float or string/blobname)`：存储的值
- `global_step(int)`：记录的Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒
- `new_style(boolean)`：是否使用新风格（tensor field）或者旧风格（simple_value field）。新风格有更快的data loading。

例如：

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()
```

`tensorboard --logdir=add_scalar`运行结果如下：

![image-20220806170609904](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806170609904.png)

**添加许多标量数据到summary**：

```python
add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
```

参数：

- `main_tag(string)`：tags的parent name
- `tag_scalar_dict(dict)`：存储tag和对应value的键值对
- `global_step(int)`：记录的Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

例如：

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.
```

`tensorboard --logdir=add_scalars`运行结果如下：

![image-20220806171201083](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806171201083.png)

**添加histogram到summary**：

```python
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
```

参数：

- `tag(string)`：数据标识器（identifier）
- `values(torch.Tensor, numpy.array, or string/blobname)`：用于构建histogram的值
- `global_step(int)`：记录的Global step value
- `bins(string)`：{'tensorflow', 'auto', 'fd', ...}中的一个，决定如何形成bins。可以找到更多选项：https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

例如：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter()
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)
writer.close()
```

`tensorboard --logdir=add_histogram`运行结果如下（和官方结果不一致）：

![image-20220806172307712](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806172307712.png)

**添加image data到summary**：注意要求`pillow`包

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

参数：

- `tag(string)`：数据标识器（identifier）
- `img_tensor(torch.Tensor, numpy.array, or string/blobname)`：image data
- `global_step(int)`：记录的Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒
- `datafromats(string)`：指定特定的image数据格式。

> 参考资料：
>
> 1. [TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)



































