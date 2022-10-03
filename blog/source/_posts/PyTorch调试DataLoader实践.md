---
title: PyTorch调试DataLoader实践
date: 2022-09-30 09:39:11
categories:
- 深度学习
tags:
- Pytorch
- 笔记
---

本文记录一下如何调试模型的数据加载类。
<!--more-->

下述代码为模型构建数据加载器对象的一般方法：
```python
def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
```
其中，`CamObjDataset`为定制化的数据集类，其继承`torch.utils.data.Dataset`，需要
- 进行`__init__`，对数据集的一些属性进行初始化，其中最重要的是提供数据变换`transforms.Compose`。
- 覆写`__getitem__`，使得可以通过索引如`dataset[0]`访问数据集数据。一般会调用初始化后的`self.transform`对数据进行变换后再返回。
- 覆写`__len__`，可通过`len(dataset)`返回数据集长度。

有时，我们会有访问加载的数据集中的单个数据以查看其形状、数据类型以及所含数据范围（如0\~255还是0\~1）的需求，此时就需要有简易的方法构建数据加载器对象并访问。其方法如下：
```python
train_path = '/storage/FT/data/TrainDataset'
batchsize = 16
trainsize = 416


image_root = '{}/Imgs/'.format(train_path)
gt_root = '{}/GT/'.format(train_path)
edge_root = '{}/Edge/'.format(train_path)

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=batchsize, trainsize=trainsize)
```
通过上述代码，我们已经实例化了数据加载器对象`train_loader`，此时可以通过在Python解释器窗口中进行如下调试访问数据：
```python
> loader = iter(train_loader) # 将train_loader转换为迭代器
> image, gt, edge = next(loader) # 取迭代器loader的下一个元素，此处为第一个
> image.shape
torch.Size([16, 3, 416, 416])
```
还可以通过以下定制化方法对image进行统计：
```python
# 进行图像数据属性统计，输入为二维张量，其尺寸为torch.Size([h, w])
def image_stat(image):
    min = 256
    max = -1
    delta = 0.00001 # 接近于零的值
    count = 0 # 大于delta的像素数量
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i][j]<min:
                min = image[i][j]
            if image[i][j]>max:
                max = image[i][j]
            if image[i][j]>delta:
                count += 1
    count_ratio = count / (h*w) # 大于零的像素数量所占比例
    print("min: ", min, " max: ", max, " count: ", count, " count_ratio: ", count_ratio)
    return min, max, count, count_ratio
```
