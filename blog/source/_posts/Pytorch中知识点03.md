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
