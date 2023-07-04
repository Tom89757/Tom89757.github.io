---
title: Pytorch中知识点03
date: 2022-10-06 17:06:19
categories:
- 深度学习
tags:
- Pytorch
- python
---

本文记录一下在调试模型过程中的一些 Pytorch 框架和 python 相关知识点。
<!--more-->

1. 报错：
```
TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect … This means that the trace might not generalize to other inputs!
```
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221011160635.png)

解决方案：将对应位置条件语句删除
```python
# if c.size() != att.size():
#     att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
# 删除if语句
att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
# if lf.size()[2:] != hf.size()[2:]:
hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
```

> 参考资料：
> 1. [TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect…This means that the trace might not generalize to other inputs!](https://discuss.pytorch.org/t/tracerwarning-converting-a-tensor-to-a-python-index-might-cause-the-trace-to-be-incorrect-this-means-that-the-trace-might-not-generalize-to-other-inputs/42282)
> 2. [TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs! #2836](https://github.com/onnx/onnx/issues/2836)
> 3. [Torch JIT Trace = TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect](https://stackoverflow.com/questions/66746307/torch-jit-trace-tracerwarning-converting-a-tensor-to-a-python-boolean-might-c)

</br>

2.报如下错误：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221006184108.png)
其原因在于数据集的大小除以batch_size余1，最后只包含单个数据的batch无法完成`batch_norm`操作。
解决方案：更改batch_size。

</br>
3.出现报错` Can't parse 'dsize'. Sequence item with index 0 has a wrong type`：

![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221010164417.png)

其原因为`(W,H)`中数据类型不对，应该将`torch.Tensor`转为`int`。由于`H`和`W`均为一维Tensor，可以通过以下代码实现：
```python
H, W = H.item(), W.item()
```
> 参考资料：
> 1. [How to cast a 1-d IntTensor to int in Pytorch](https://stackoverflow.com/questions/47588682/how-to-cast-a-1-d-inttensor-to-int-in-pytorch)

</br>
4.当通过`state_dict = model_zoo.load_url(url_map_[model_name]`在线下载预训练的模型权重文件时，出现`urllib.error.HTTPError: HTTP Error 503: Egress is over the account limit.`错误。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221018162424.png)

- 原因：状态码为`503 Service Unavailable`，表示临时的服务器维护或者过载，服务器当前无法处理请求，这个情况是暂时的，会在一段时间后恢复（实际上过了好多天都没修复）
- 解决方案：因为调用`model_zoo.load_url()`时会打印出该权重文件的下载地址和保存路径，可以通过手动下载并放入该路径来解决该问题，如下图所示
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221018162553.png)

> 参考资料：
> 1. [HTTP状态码](https://zh.m.wikipedia.org/zh/HTTP%E7%8A%B6%E6%80%81%E7%A0%81)

</br>
5.报错：
```python
Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>' as a data type
```
解决方法：更新pandas
> 参考资料：
> 1. [TypeError: Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>' as a data type · Issue #18355 · numpy/numpy · GitHub](https://github.com/numpy/numpy/issues/18355)


</br>
6.`loss.backward()`报错：`grad can be implicitly created only for scalar outputs`
解决方法：`loss.mean().backward()`。
> 参考资料：
> 1. [Loss.backward() raises error 'grad can be implicitly created only for scalar outputs' - autograd - PyTorch Forums](https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152/2)


</br>
7. Pytorch创建随机张量和创建张量：
- 随机张量均匀分布
```python
x1 = torch.rand((4, 256, 80, 80))
x2 = torch.rand((4, 512, 40, 40))
x3 = torch.rand((4, 1024, 20, 20))
x4 = torch.rand((4, 2048, 10, 10))
```
- 随机张量标准正态分布
```python
x = torch.randn((4, 3, 320, 320))
```
- 全部为1：
```python
x = torch.ones(2, 3)
```
- 全部为0：
```python
x = torch.zeros(2, 3)
```
> 参考资料：
> 1. [PyTorch 常用方法总结1：生成随机数Tensor的方法汇总（标准分布、正态分布……） - 知乎](https://zhuanlan.zhihu.com/p/31231210)
> 2. [torch.ones — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.ones.html)
> 3. [torch.ones_like — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.ones_like.html)
> 4. [torch.zeros — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.zeros.html)
> 5. [torch.zeros_like — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.zeros_like.html)


</br>
8.BCE Loss vs Cross Entropy：

> 参考资料：
> 1. [BCE Loss vs Cross Entropy - vision - PyTorch Forums](https://discuss.pytorch.org/t/bce-loss-vs-cross-entropy/97437)
> 2. [Learning Day 57/Practical 5: Loss function — CrossEntropyLoss vs BCELoss in Pytorch; Softmax vs sigmoid; Loss calculation | by De Jun Huang | dejunhuang | Medium](https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23#:~:text=Difference%20in%20purpose%20(in%20practice,probability%2C%20you%20should%20use%20BCE.&text=We%20cannot%20use%20sigmoid%20for,CrossEntropyLoss%20as%20the%20loss%20function.)
> 3. [BCELoss — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
> 4. [CrossEntropyLoss — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

</br>
9.张量最大值和最小值：
```python
# 最大
x.max()
# 最小
x.min()
```

</br>
10.维度扩充、复制和压缩：
```python
# 原始tensor x
x = torch.randn((366, 400)) # shape: 366, 400
# 扩充维度
x = x.unsqueeze(0) # shape: 1, 366, 400
# 复制通道
x = x.repeat(3, 1, 1) # shape: 3, 366, 400
# 压缩维度
x = x.squeeze(0) # shape: 3, 366, 400 不能压缩非1的维度
x = torch.randn((1, 366, 400))
x = x.squeeze(0) # shape: 366, 400
```
> 参考资料：
> 1. [记录一个Tensor操作——扩充维度+复制 - 知乎](https://zhuanlan.zhihu.com/p/442263715)

</br>
11.在Module中的如下操作报如下错误：
```python
gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
```
```error
Leaf variable was used in an inplace operation
```
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230218224821.png)
解决方案：注释掉151行
> 参考资料：
> 1. [Leaf variable was used in an inplace operation - PyTorch Forums](https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308)

</br>
12.在Module的batch上面迭代：
```python
def forward(self, x):
	B, C, H, W = x.shape
	for i in range(B):
		xi = x[i, ...][None, ...]
		xi = self.net(xi)
```

</br>
13.对多个Tensor进行join。有两种方式：
- `torch.cat((tens_1, tens_2, -, tens_n), dim=0, *, out=None)`：在相同的dimension上concatenate多个tensor：
```python
x1 = torch.randn((1,3,320,320)) # x1.shape: (1,3,320,320)
x2 = torch.randn((1,3,320,320)) # x2.shape: (1,3,320,320)
x = torch.cat((x1,x2), dim=0) # x.shape: (2,3,320,320)
```
- `torch.stack((tens_1, tens_2, -, tens_n), dim=0, *, out=None)`：在一个新的dimension上concatenate一个tensor序列，tensor需要为相同的尺寸：
```python
x1 = torch.randn((1,3,320,320)) # x1.shape: (1,3,320,320)
x2 = torch.randn((1,3,320,320)) # x2.shape: (1,3,320,320)
x = torch.stack((x1, x2), dim=0) # x.shape: (2,1,3,320,320)
```
> 参考资料：
> 1. [How to join tensors in Pytorch](https://www.geeksforgeeks.org/how-to-join-tensors-in-pytorch/)
> 2. [How to join tensors in Pytorch?](https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch) 

</br>
14.Pytorch对tensor取指定维度的切片。
```python
a = torch.randn((16,3,320,320)) # (16,3,320,320)
a1 = a[0] # (3,320,320)
a2 = a[0:1] # (1,3,320,320)
a3 = a[None, 0] # (1,3,320,320)

b = torch.randn((1, 1, 320, 320)) # (1, 1, 320, 320)
b1 = b[0] # (1, 320, 320)
b2 = b[0].squeeze(0) # (320, 320)
b3 = b[0].unsqueeze(0) # (1, 1, 320, 320)

c = torch.randn((16, 2, 320, 320)) # (16, 2, 320, 320)
c1 = c[:, None, 1, :] # (16, 1, 320, 320)
```
> 参考资料：
> 1. [python - Tensorflow: How to slice tensor with number of dimension not changed? - Stack Overflow](https://stackoverflow.com/questions/51670073/tensorflow-how-to-slice-tensor-with-number-of-dimension-not-changed)

</br>
15.报错`RuntimeError: Function 'SqrtBackward0' returned nan values in its 0th output.`
解决方案：
```python
grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
# 添加1e-8项
grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2 + 1e-8)
```
> 参考资料：
> 1. [RuntimeError: Function 'SqrtBackward' returned nan values in its 0th output - autograd - PyTorch Forums](https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702/5)

</br>
16.报错`ImportError: cannot import name 'container_abcs' from 'torch._six'`。
原因：torch版本问题，本人版本为`1.10`。
解决方案：
```python
from torch._six import container_abcs
# 改为
import collections.abc as container_abcs
```
> 参考资料：
> 1. [Colab Notebook: Cannot import name 'container_abcs' from 'torch._six'](https://stackoverflow.com/questions/70193443/colab-notebook-cannot-import-name-container-abcs-from-torch-six)

</br>
17.高效的tensor张量矩阵阈值过滤操作：
```python
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
        [-1.2035,  1.2252,  0.5002,  0.6248],
        [ 0.1307, -2.0608,  0.1244,  2.0139]])
>>> mask = x.ge(0.5)
>>> mask
tensor([[False, False, False, False],
        [False, True, True, True],
        [False, False, False, True]])
>>> torch.masked_select(x, mask)
tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
```
`torch.masked_select(x, mask)`和`x[mask]`作用相似，可能的差别见参考资料3。但二者返回的tensor均为一维张量，而不是和输入的x和mask相同的shape。
若想保持原始tensor的尺寸，可以进行如下操作：
```python
x = torch.randn(3,4)
mask = torch.zeros(x.shape)
mask[x>0.5] = 1
result = torch.mul(x, mask)
```
> 参考资料：
> 1. [torch.masked_select — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.masked_select.html)
> 2. [PyTorch中的masked_select选择函数 - 知乎](https://zhuanlan.zhihu.com/p/348035584)
> 3. [Please add "dim" feature for function "torch.masked_select" · Issue #48830 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/48830)

</br>
18.PyTorch中的矩阵乘法操作：
点乘：矩阵逐个元素（element-wise）乘法
```python
import torch
import cv2
mask = cv2.imread('./mask.png', 0)
mask = torch.from_numpy(mask)

edge = cv2.imread('./edge.png', 0)
edge = torch.from_numpy(edge)

mask1 = mask / 255.0

mask_region = mask1.ge(0.5)
mask_final = torch.zeros(edge.shape)
mask_final[mask_region==True] = 1
mask_final[mask_region==False] = 0

edge1 = torch.mul(edge, mask_final).numpy()
```
> 参考资料：
> 1. [随笔1: PyTorch中矩阵乘法总结 - 知乎](https://zhuanlan.zhihu.com/p/100069938)
> 2. [torch.Tensor的4种乘法_torch tensor 相乘_da_kao_la的博客-CSDN博客](https://blog.csdn.net/da_kao_la/article/details/87484403)

</br>
19.报错`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! when resuming training`
> 参考资料：
> 1. [python - RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! when resuming training - Stack Overflow](https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least)

</br>
20.报错`Could not load dynamic library 'libnvinfer_plugin.so.7`。
解决方案：建立从`libvinfer`版本7到版本8的symbolic link
```python
# the follwoing path will be different for you - depending on your install method
$ cd env/lib/python3.10/site-packages/tensorrt

# create symbolic links
$ ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7
$ ln -s libnvinfer.so.8 libnvinfer.so.7

# add tensorrt to library path
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/env/lib/python3.10/site-packages/tensorrt/
```
> 参考资料：
> 1. [tensorflow - Could not load dynamic library 'libnvinfer.so.7' - Stack Overflow](https://stackoverflow.com/questions/74956134/could-not-load-dynamic-library-libnvinfer-so-7)

</br>
21.报错`tensorflow.python.framework.errors_impl.PermissionDeniedError: : /storage/FT/pth/SCWSSOD/SCWSSOD28/events.out.tfevents.1678606756.node4.11263.0; Permission denied`
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230312170110.png)
解决方案：更改`/storage/FT/pth/SCWSSOD/SCWSSOD28`文件夹的权限
```bash
chmod -R 777 /storage/FT/pth/SCWSSOD/SCWSSOD28
```
> 参考资料：
> 1. [tensorflow.python.framework.errors_impl.PermissionDeniedError: data · Issue #6393 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/6393)
> 2. [tensorflow.python.framework.errors_impl.PermissionDeniedError: data · Issue #6393 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/6393)

</br>
22.Pytorch保存中间权重作为checkpoint，并从中间权重加载权重开始继续训练：
1. 定义和初始化网络
```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

net = Net()
print(net)
```
2. 初始化优化器
```python
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```
3. 保存checkpoint
```python
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
```
4. 加载checkpoint
```python
model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```
> 参考资料：
> 1. [Saving and loading a general checkpoint in PyTorch — PyTorch Tutorials 1.13.1+cu117 documentation](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

</br>
23.将损失和张量加载到gpu中：
```python
tree_loss = TreeEnergyLoss().cuda()
preds = torch.randn((2, 21, 128, 128)).to(device='cuda')
low_feats = torch.randn((2, 3, 512, 512)).to(device='cuda')
high_feats = torch.randn((2, 256, 128, 128)).to(device='cuda')
unlabeled_ROIs = torch.randn((2, 512, 512)).to(device='cuda')
```
> 参考资料：
> 1. [Moving tensor to cuda - PyTorch Forums](https://discuss.pytorch.org/t/moving-tensor-to-cuda/39318)

</br>
24.报错`AttributeError: module 'distutils' has no attribute 'version'`。
问题：setuptools新版本中移除了某些属性
解决方案：对setuptools进行降级
```python
pip install setuptools==59.5.0
```
> 参考资料：
> 1. [AttributeError: module 'distutils' has no attribute 'version' : with setuptools 59.6.0 · Issue #69894 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/69894)

</br>
25.报错`Default process group has not been initialized, please make sure to call init_process_group`。
问题：网络中包含了`SyncBatchNorm`操作，该操作必须在两张卡上进行。
解决方案：将`SyncBatchNorm`改为`BatchNorm2d`。

> 参考资料：
> 1. [RuntimeError: Default process group has not been initialized, please make sure to call init_process_group. · Issue #3972 · facebookresearch/detectron2 · GitHub](https://github.com/facebookresearch/detectron2/issues/3972)

</br>
26.报错`Assertion 't>=0' && 't<n_classes' failed error`。
问题：通过以下方式初始化张量会导致出现`<0`的概率值，无法计算损失
```python
images = torch.randn((8, 2, 3, 512, 512)).to(device='cuda')
masks = torch.randn((8, 2, 512, 512)).to(device='cuda')
image, mask = images[0], masks[0]
outputs = model(image)
loss = seg_loss([outputs[0], outputs[0]], mask)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
解决方案：将`torch.randn()`改为`torch.ones()`。
```python
images = torch.ones((8, 2, 3, 512, 512)).to(device='cuda')
masks = torch.ones((8, 2, 512, 512)).to(device='cuda')
```
> 参考资料：
> 1. [Assertion `t >= 0 && t < n_classes` failed error - vision - PyTorch Forums](https://discuss.pytorch.org/t/assertion-t-0-t-n-classes-failed-error/133794)

</br>
27.报错`TracerWarning: Encountering a list at the output of the tracer might cause the trace to be incorrect, this is only valid if the container structure does not change based on the module's inputs. Consider using a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead). If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior.`
问题：

> 参考资料：
> 1. [PyTorch TensorBoard add_graph() dictionary input error](https://stackoverflow.com/questions/70706389/pytorch-tensorboard-add-graph-dictionary-input-error)

</br>
28.张量(tensor)所在设备和加载到cpu/gpu：
```python
>> a = torch.randn((1,1))
>> a.device
device(type='cpu')
>> a = a.cuda()
>> a.device
device(type='cuda', index=0)
>> a = a.cpu()
>> a.device
device(type='cpu')
```

</br>
29.处理图片时报错`ValueError: the input array must have size along channel_axis, got (267, 400)`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230421182748.png)

问题：在使用Skimage处理单通道图片时进行了多余的转换：
```python
MASK = color.rgb2gray(MASK)  # shape of [h, w]
```
解决方案：将上述代码改为，
```python
if len(MASK.shape==3): 
	MASK = color.rgb2gray(MASK)  # shape of [h, w]
```
> 参考资料：
> 1. [/data/__init__.py 54 MASK = color.rgb2gray(MASK) · Issue #2 · BarCodeReader/SelfReformer · GitHub](https://github.com/BarCodeReader/SelfReformer/issues/2)
> 2. [python - Skimage rgb2gray giving errors, the input array must have size 3 along - Stack Overflow](https://stackoverflow.com/questions/70895576/skimage-rgb2gray-giving-errors-the-input-array-must-have-size-3-along)

</br>
20.在`test.py`中设置GPU编号无效，仍使用GPU0
```python
GPU_ID=1
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
```
原因：在导入`kornia`时出现了问题
解决方案：将上述代码放在`test.py`最前面

> 参考资料：
> 1. [[1.12] os.environ["CUDA_VISIBLE_DEVICES"] has no effect · Issue #80876 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/80876)
> 2. [`import kornia` break CUDA lazy init · Issue #1951 · kornia/kornia · GitHub](https://github.com/kornia/kornia/issues/1951)

</br>
21.报错`tensorflow.python.framework.errors_impl.PermissionDeniedError:`

> 参考资料：
> 1. [python - TensorFlow Permission Denied Error /Location - Stack Overflow](https://stackoverflow.com/questions/41606854/tensorflow-permission-denied-error-location)

</br>
22.在`bgnet.py`中导入上一层目录中其它文件夹中的包：
```python
import sys
sys.path.insert(0, '../utils')
from show_info import * 
```
目录结构如下：
```python
+--bgnet
\----bgnet.py
+--utils
\----show_info.py
```

</br>
报错`"conda\activate.py", line 1210, in main print(activator.execute(), end='') UnicodeEncodeError: 'gbk' codec can't encode charact`
问题：git bash作为终端时Python编码格式未设置为`utf-8`
解决方案：在`.bashrc/.zshrc/.bash_path`中添加环境变量：
```bash
export PYTHONIOENCODING=utf-8
export PYTHONLEGACYWINDOWSSTDIO=utf-8
```
> 参考资料：
> 1. [python 3.x - Conda: UnicodeEncodeError: 'charmap' codec can't encode character '\u2580' in position 644: character maps to undefined - Stack Overflow](https://stackoverflow.com/questions/59974715/conda-unicodeencodeerror-charmap-codec-cant-encode-character-u2580-in-po)
> 2. [Can not activate/deactivate conda environment due to cmder lambda character not handled in conda encoder/decoder · Issue #7445 · conda/conda · GitHub](https://github.com/conda/conda/issues/7445)

</br>
23.关闭`nvidia-smi`命令中运行在指定显卡上的进程：
```bash
nvidia-smi | grep 'python' | grep 19398 | awk '{ print $5 }' | xargs -n1 kill -9
```
> 参考资料：
> 1. [python - How to kill process on GPUs with PID in nvidia-smi using keyword? - Stack Overflow](https://stackoverflow.com/questions/50193538/how-to-kill-process-on-gpus-with-pid-in-nvidia-smi-using-keyword)







