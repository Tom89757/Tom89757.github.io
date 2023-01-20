---
title: PyTorch可视化实践
date: 2022-08-07 10:45:38
categories:
- 深度学习
tags:
- Pytorch
- 笔记

---

本文记录一下如何使用TensorBoard工具对深度学习模型及其数据进行可视化：

<!--more-->

此处以调用`add_graph()`展示模型的computation graph为例，说明如何使用TensorBoard对模型计算图进行可视化，并在本地浏览器可以查看远程服务器的tensorboard可视化情况。

其主要步骤如下：

1. 在`trian.py`文件中导入tensorboard的`SummaryWriter`类并创建对象：

   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter("add_graph") # 默认为runs文件夹，此处指定为add_graph文件夹
   ```

2. 在`epoch=0`且`idx=0`处调用`add_graph`：

   ```python
   if epoch == 0 and i == 0:
   	# add_graph
   	print("epoch=", epoch, ", i=", i)
   	writer.add_graph(model, input_to_model=images, verbose=False)
   ```

3. 运行`python train.py`，在训练成功开始后会发现add_graph文件夹中会出现如下所示类型的文件：

   ![image-20220808170232112](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220808170232112.png)

4. 远程登录服务器并指定与后续tensorboard使用端口对应的本地端口，以便后续在本地浏览器对应端口显示：

   ```bash
   ssh -L 16008:127.0.0.1:6008  user@server
   ```

   上述命令表示后续在服务器的6008端口打开tensorboard可视化文件时，可以在本地浏览器的16008端口查看可视化结果。

5. 在服务器上运行`tensorboard --logdir=add_graph --port=6008`，此时已经在服务器的6008端口打开tensorboard的可视化文件，并将其同步到本地的16008端口（报错不影响结果展示）：

   ![image-20220808170714479](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220808170714479.png)

6. 此时，在本地浏览器打开 http://localhost:16008 即可看到如下结果：

   ![image-20220808170919983](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220808170919983.png)
PS：当出现`you haven't written any data to your event files.`的情形时，不要急着调试错误。将`runs/add_graph`等events文件所在的文件夹换个位置，重新使用tensorboard加载。

> 参考资料：
>
> 1. [The Best Tools for Machine Learning Model Visualization](https://neptune.ai/blog/the-best-tools-for-machine-learning-model-visualization)
> 2. [Visualize PyTorch Model Graph with TensorBoard](https://liarsliarsliars.com/visualize-pytorch-model-graph-with-tensorboard/)
> 3. [VISUALIZING MODELS, DATA, AND TRAINING WITH TENSORBOARD](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
> 4. [how-to-use-tensorboard-with-pytorch](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md)
> 5. [Deep Dive Into TensorBoard: Tutorial With Examples](https://neptune.ai/blog/tensorboard-tutorial)
> 6. [本地浏览器使用tensorboard查看远程服务器训练情况](https://blog.csdn.net/u010626937/article/details/107747070)
> 7. [How can I run Tensorboard on a remote server?](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)