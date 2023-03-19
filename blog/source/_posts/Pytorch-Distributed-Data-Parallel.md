---
title: Pytorch Distributed Data Parallel
date: 2022-09-12 16:51:19
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下Pytorch中的核心操作之一——Distributed Data Parallel (分布式数据并行)
<!--more-->
训练时：
```python
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3, 5"
multi_gpu = True
model = Model(args)
if multi_gpu:
	print("training on multi_gpu: ")
	torch.cuda.empty_cache()
	model = nn.DataParallel(model)
model.train(True)
model.cuda()
```
测试时：
```python
model = Model(args)
if multi_gpu:
	print("testing on multi_gpu...")
	model = nn.DataParallel(model)
model.load_state_dict(torch.load(path))
model.train(False)
model.cuda()
```
> 参考资料：
> 1. [TRACER/trainer.py at main · Karel911/TRACER · GitHub](https://github.com/Karel911/TRACER/blob/main/trainer.py)
> 2. [Optional: Data Parallelism — PyTorch Tutorials 2.0.0+cu117 documentation](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
> 3. [DataParallel — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)


