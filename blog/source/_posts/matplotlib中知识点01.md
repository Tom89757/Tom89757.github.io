---
title: matplotlib中知识点01
date: 2022-06-03 14:37:10
categories:
- 深度学习
tags:
- matplotlib
- python
---

本文记录一下在实现 [DDRQM](https://link.springer.com/article/10.1007/s11042-016-3392-4) 过程中的一些 matplotlib 包和 python 相关知识点。

<!--more-->

1.`matplotlib.pyplot.hist`或`plt.hist`：用于绘制直方图。

</br>

2.`matplotlib.pyplot.show`或`plt.show`：用于展示所有打开的图片。完整调用形式如下：

```python
matplotlib.pyplot.show(*, block=None)
```

- `block`：布尔类型，可选。表示在返回之前是否等待所有figures关闭。默认为True，通常在非交互模式使用；交互模式通常设为False。

> 参考资料：
>
> 1. [matplotlib.pyplot.show](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html)

</br>

3.`matplotlib.pyplot.imshow`或`plt.imshow`：用于将数据作为图像展示，例如以$2*2$的形式展示4张图片。输入要么是$RGB(A)$数据，要么是二维的标量数据，后者将被渲染成一张具有伪颜色的图像。显示灰度图时可以设置参数`cmap='gray'`。完整调用形式为：

```python
matplotlib.pyplot.imshow(X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None, data=None, **kwargs)
```

- `X`：数组形式或者PIL图像。支持的数组类型有：
  - (M, N)，具有标量数据的图像
  - (M, N, 3)，具有$RGB$值的图像
  - (M, N, 4)，具有$RGBA$值的图像，包括透明度

- `cmap`：用于将标量数据映射为colors，对$RGB(A)$数据该参数无效
- 略

> 参考资料：
>
> 1. [matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)

</br>

4.当使用matplotlib画有很多subplots的图时，改善subplots布局：

> 参考资料：
>
> 1. [Improve subplot size/spacing with many subplots in matplotlib](https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib)

</br>
5.使用matplot画散点图，利用scipy计算相关系数并利用sklearn计算回归：
```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import pearsonr

def point_plot(model, dataset, size=0.1):
    score_avgf_file = model + '_' + dataset + '.txt'
    score_avgf_pair = open('./txt/' + score_avgf_file).read().splitlines()
    score = []
    avgf = []
    for pair in score_avgf_pair:
        score.append(float(pair.split(' ')[0]))
        avgf.append(float(pair.split(' ')[1]))
    
    scores = np.array(score)
    avgfs = np.array(avgf)

    # plt.plot(scores, avgfs, 'o')
    # corrco = np.corrcoef(scores, avgfs)
    corrco = pearsonr(scores, avgfs)
    plt.scatter(scores, avgfs, s=size)
    plt.xlabel('image complexity')
    plt.ylabel('avg F')
    title = model + ' on ' + dataset + ', correlation coefficient=' + str(corrco[0])
    plt.title(title)
    save_fig = model + '_' + dataset + '.png'

    reg = LinearRegression().fit(scores.reshape(-1,1), avgfs)
    pred = reg.predict(scores.reshape(-1,1))
    plt.plot(scores, pred,linewidth=2, color='red', label='回归线')

    plt.savefig('./fig/' + save_fig)
    plt.show()
```
> 参考资料：
> 1. [从零开始学Python【15】--matplotlib(散点图) - 天善智能：专注于商业智能BI和数据分析、大数据领域的垂直社区平台](https://ask.hellobi.com/blog/lsxxx2011/10243)
> 2. [如何在 Matplotlib 中设置散点图的标记大小](https://www.delftstack.com/zh/howto/matplotlib/how-to-set-marker-size-of-scatter-plot-in-matplotlib/)
> 3. [Matplotlib 散点图 | 菜鸟教程](https://www.runoob.com/matplotlib/matplotlib-scatter.html)
> 4. [Python三种方法计算皮尔逊相关系数](https://blog.csdn.net/qq_40260867/article/details/90667462)

</br>
5.由于OpenCV读取的图片默认三通道顺序为BGR，所以在使用matplotlib进行画图时，需要对其通道顺序进行调整：
```python
from matplotlib import pyplot as plt
plt.subplot(1,1,1)
plt.imshow(result[:, :, [2, 1, 0]])
plt.title("result")
plt.show()
```

