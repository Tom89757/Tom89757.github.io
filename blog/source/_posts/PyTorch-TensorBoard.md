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

### 其初始化代码为

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

### 添加scalar数据到summary

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

### 添加许多标量数据到summary

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

### 添加histogram到summary

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

### 添加image data到summary

注意要求`pillow`包

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

参数：

- `tag(string)`：数据标识器（identifier）
- `img_tensor(torch.Tensor, numpy.array, or string/blobname)`：image data
- `global_step(int)`：记录的Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒
- `datafromats(string)`：指定特定的image数据格式（CHW, HWC, HW, WH等）。

Shape：

`img_tensor`默认为(3, H, W)，可以使用`torchvision.utils.make_grid()`来转换a batch of tensor为3xHxW，或者调用`add_images`来做这件事。只要传递相应的`dataformats`(1, H, W)，(H, W)或者(H, W, 3)形状的Tensor也支持。

例如：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
img = np.zeros((3, 100, 100))
img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

img_HWC = np.zeros((100, 100, 3))
img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

writer = SummaryWriter()
writer.add_image('my_image', img, 0)

# If you have non-default dimension setting, set the dataformats argument.
writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
writer.close()
```

`tensorboard --logdir=add_image`运行结果如下：

![image-20220806203930908](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806203930908.png)

### 添加batched image data到summary

注意其要求`pillow`包

参数：

- `tag(string)`：数据标识器（identifier）
- `img_tensor(torch.Tensor, numpy.array, or string/blobname)`：image data
- `global_step(int)`：记录的Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒
- `dataformats(string)`：指定特定的image数据格式（CHW, HWC, HW, WH等）。

Shape：

`img_tensor`默认为(N, 3, H, W)，如果指定`dataformats`，其他shape也可以被接受，如NCHW或NHWC。

例如：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

img_batch = np.zeros((16, 3, 100, 100))
for i in range(16):
    img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
    img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

writer = SummaryWriter()
writer.add_images('my_image_batch', img_batch, 0)
writer.close()
```

`tensorboard --logdir=add_images`运行结果如下：

![image-20220806205216711](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220806205216711.png)

### 渲染matplotlib figure为一张image并加入summary

注意要求`matplotlib`包

```python
add_figure(tag, figure, global_step=None, close=True, walltime=None)
```

参数：

- `tag(string)`：数据标识器（identifier）
- `figure(matplotlib.pyplot.figure)`：Figure或者figures列表
- `close(bool)`：自动关闭figure的flag
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

### 添加video数据到summary

主要要求`moviepy`包

```python
add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)
```

参数：

- `tag(string)`：数据标识器（identifier）
- `vid_tensor(torch.Tensor)`：Video数据
- `global_step(int)`：记录的Global step值
- `fps(float or int)`：每秒帧数
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

Shape：

`vid_tensor`为(N, T, C, H, W)。值应该位于[0, 255]为uint8类型，或者[0, 1]的float类型

### 添加audio数据到summary

```python
add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)
```

参数：

- `tag(string)`：数据标识器（identifier）
- `snd_tensor(torch.Tensor)`：Sound数据
- `global_step(int)`：记录的Global step value
- `sample_rate(int)`：sample rate，单位为Hz
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

Shape：

`snd_tensor`为(1, L)，值位于[-1, 1]

### 添加text数据到summary

```python
add_text(tag, text_string, global_step=None, walltime=None)
```

参数：

- `tag(string)`：数据标识器（identifier）
- `text_string(string)`：存储的String。
- `global_step(int)`：记录的Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

例如：

```python
writer.add_text('lstm', 'This is an lstm', 0)
writer.add_text('rnn', 'This is an rnn', 10)
```

### 添加graph数据到summary

```python
add_graph(model, input_to_model=None, verbose=False, use_strict_trace=True)
```

参数：

- `model(torch.nn.Module)`：要画的模型
- `input_to_model(torch.Tensor or list of torch.Tensor)`：一个变量或者要输入的一个变量元组。
- `verbose(bool)`：是否在console中打印出graph结构。
- `use_strict_trace(bool)`：是否传递keyword 参数 strict到`torch.jit.trace`。当你想tracer记录mutable container类型（如list, dict）时传递False

### 添加embedding projector数据到summary

```python
add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
```

参数：

- `mat(torch.Tensor or numpy.array)`：一个矩阵，每一行为data point的feature vector
- `metadata(list)`：labels列表，每个元素将转换为string
- `label_img(torch.Tensor)`：对应每个data point的image
- `global_step(int)`：记录的Global step值
- `tag(string)`：embedding名

Shape：

`mat`为(N, D)，N为data数量，D为特征维数；`label_img`为(N, C, H, W)

例如：输出有误

```python
import keyword
import torch
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0

writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.add_embedding(torch.randn(100, 5), label_img=label_img)
writer.add_embedding(torch.randn(100, 5), metadata=meta)
```

### 添加precision recall曲线

画precision-recall曲线让你理解在不同阈值设置下你的模型性能。使用这个函数，你提供真值labeling(T/F)和对每个target的预测置信度（通常为模型输出）。TensorBoard UI将会让你交互式地选择阈值。

```python
add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)
```

参数：

- `tag(string)`：数据标识器（identifier）

- `labels(torch.Tensor, numpy.array, or string/blobname)`：真值数据，每个元素的二值label。
- `predictions(torch.Tensor, numpy.array, or string/blobname)`：每个元素正确分类的概率，值应该在[0, 1]
- `global_step(int)`：记录的Global step值
- `num_thresholds(int)`：用来画曲线的阈值数量
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

例如：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
labels = np.random.randint(2, size=100)  # binary label
predictions = np.random.rand(100)
writer = SummaryWriter("add_pr_curve")
writer.add_pr_curve('pr_curve', labels, predictions, 0)
writer.close()
```

`tensorboard --logdir=add_pr_curve`运行结果如下：

![image-20220807000205981](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220807000205981.png)

### 通过在scalars中收集charts tags创建专门的chart

```python
add_custom_scalars(layout)
```

注意该函数对每个SummaryWriter()对象只能调用一次。因为它只对tensorboard提供元数据，该函数可以在training loop之前或之后调用。

参数：

- `layout(dict)`：{categoryName: charts}，charts也是一个字典{chartName: ListOfProperties}。ListOfProperties的第一个元素是chart的类型（多行或者Margin之一），第二个元素应该是一个列表，包含在add_scalar函数中使用的tags，它们会被收集到new chart。

例如：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("add_custom_scalars")

layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
             'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                  'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

writer.add_custom_scalars(layout)
```

`tensorboard --logdir=add_custom_scalars`运行结果如下：

![image-20220807102302198](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220807102302198.png)

### 添加meshes或3D point clouds到TensorBoard

```python
add_mesh(tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None)
```

此可视化基于Three.js，所以它允许用户和被渲染的对象交互。除了基础的定义如vertices/faces外，用户可以提供camera parameters/lighting condition等。在 https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene 进一步学习

参数：

- `tag(string)`：数据标识器（identifier）
- `vertices(torch.Tensor)`：3D 顶点坐标列表
- `colors(torch.Tensor)`：每个顶点颜色
- `faces(torch.Tensor)`：每个矩形中顶点索引（可选）
- `config_dict`：ThreeJS类名和配置的字典
- `global_step(int)`：记录Global step value
- `walltime(float)`：可选择，用于覆盖默认的walltime(`time.time()`)，表示在epoch of event后的几秒

Shape：

- `vertices`：(B, N, 3)，顶点数目，通道数
- `colors`：(B, N, 3)，值应该为[0, 255]的uint8，或者[0, 1]的float
- `faces`：(B, N, 3)，值应该为[0, 顶点数]的uint8

例如：

```python
from torch.utils.tensorboard import SummaryWriter
vertices_tensor = torch.as_tensor([
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, -1],
    [-1, 1, -1],
], dtype=torch.float).unsqueeze(0)
colors_tensor = torch.as_tensor([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
], dtype=torch.int).unsqueeze(0)
faces_tensor = torch.as_tensor([
    [0, 2, 3],
    [0, 3, 1],
    [0, 1, 2],
    [1, 3, 2],
], dtype=torch.int).unsqueeze(0)

writer = SummaryWriter()
writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

writer.close()
```

`tensorboard --logdir=add_mesh`运行结果如下：

![image-20220807103349415](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220807103349415.png)

### 添加一系列hyperparameters到TensorBoard中比较

```python
add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
```

参数：

- `hparam_dict(dict)`：每个字典中的键值对为超参数名和其对应值。值类型为bool/string/float/int/None。
- `metric_dict(dict)`：每个键值对为metric名和对应值。注意用在这的key应该是unique，否则通过`add_scalar`添加的值将显示在hparam部分。大多数场景下这不是我们想要的。
- `hparam_domain_discrete`：可选，包含超参数名和其离散值的字典
- `run_time(str)`：run的名，包含在logdir部分。如果不指定，将使用当前时间戳。

例如：

```python
from torch.utils.tensorboard import SummaryWriter
with SummaryWriter() as w:
    for i in range(5):
        w.add_hparams({'lr': 0.1*i, 'bsize': i},
                      {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
```

`tensorboard --logdir=add_hparams`运行结果如下：

![image-20220807104044985](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220807104044985.png)

`flush()`：刷新写入disk的event文件，调用该方法确保所有pending events写入disk

`close()`：关闭SummaryWriter对象，停止写入

> 参考资料：
>
> 1. [TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)



































