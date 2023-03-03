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
10.扩充维度和复制：
```python
# 原始tensor x
x = torch.randn((366, 400)) # shape: 366, 400
# 扩充维度
x = x.unsqueeze(0) # shape: 1, 366, 400
# 复制通道
x = x.repeat(3, 1, 1) # shape: 3, 366, 400
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
> 1. [Site Unreachable](https://stackoverflow.com/questions/70193443/colab-notebook-cannot-import-name-container-abcs-from-torch-six)