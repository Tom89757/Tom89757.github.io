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

