---
title: OpenCV中知识点01
date: 2022-06-02 17:09:03
categories:
- 深度学习
tags:
- OpenCV
- python
---

本文记录一下在实现 [DDRQM](https://link.springer.com/article/10.1007/s11042-016-3392-4) 过程中的一些 OpenCV 框架和 python 相关知识点。

<!--more-->

1.`cv2.filter2D()`：该函数表示在一张图像上应用相应的卷积核，完整的函数调用形式如下：

```python
cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
```

该函数将任意的 linear filter 应用到一张图像上。当 filter 的部分孔径落在图像之外时，函数会根据 border 类型进行插值。该函数实际上计算的是相关（correlation），而不是卷积（convolution）。

例如，下面代码：

```python
retval = cv2.getGaborKernel(ksize=(111,111), sigma=10, theta=60, lambd=10, gamma=1.2)
image1 = cv2.imread('src.jpg')
result = cv2.filter2D(image1,-1,retval)
```

表示将对应的 $Gabor$ 滤波器应用在图像`src.jpg`上。

```
a=cv2.getGaborKernel((40,40), 1.69, 0, 3, 0.5)
b=Gabor_filter((40,40), 1.69, 0, 3, 0.5)
```

</br>

2.`cv2.getGaborKernel()`：该函数为 $Gabor$ 滤波器函数，其返回值为一个 $Gabor filter$，具体形式为一个二维数组，完整的函数调用形式如下：

```python
cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]) -> retval
```

</br>

3.代码为：

```python
result[x][y] = abs(depth_img[x][y] - 0.5*(depth_img[x1][y1]+depth_img[x2][y2]))
```

报错信息：`RuntimeWarning: overflow encountered in ubyte_scalars`。

分析：可能是将两个`unit8`类型的值相加并将其存入一个`unit8`，导致数值溢出

解决方案：将`unit8`类型数值转换为`int`类型。

```python
result[x][y] = abs(int(depth_img[x][y]) - 0.5*(int(depth_img[x1][y1])+int(depth_img[x2][y2])))
```

> 参考资料：
>
> 1. [RuntimeWarning: overflow encountered in ubyte_scalars](https://stackoverflow.com/questions/9384435/runtimewarning-overflow-encountered-in-ubyte-scalars)
>
> 2. [Why I am getting "RuntimeWarning: overflow encountered in ubyte_scalars error" in python?](https://stackoverflow.com/questions/59531334/why-i-am-getting-runtimewarning-overflow-encountered-in-ubyte-scalars-error-i)

</br>

4.以下代码：

```python
img = cv2.imread(path)
height, width, channels = img.shape
```

表示从路径`path`中读取图像文件并存入`img`对象中，通过`type(img)`可知该对象类型为`<class 'numpy.ndarray'>`。`img.shape`返回的是一个三元元组，其值分别对应图像的`height`、`width`和`channels`，即图像形状并不是我们熟悉的`width * height * channels `的形式

可以通过以下代码：

```python
img = cv2.transpose(img)
```

将`img`转置为`width * height * channels`形状，通过我们熟悉的方式访问。

PS：`img`中`channels`顺序为$B G R$。

> 参考资料：
>
> 1. [Opencv showing wrong width and height of image](https://stackoverflow.com/questions/55636318/opencv-showing-wrong-width-and-height-of-image)
> 2. [OpenCV-Python教程：几何空间变换~缩放、转置、翻转(resize,transpose,flip)](http://www.juzicode.com/opencv-python-resize-transpose-flip/)
> 3. [Image file reading and writing](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)

</br>

5.`PILLOW` vs `OpenCV`：

```python
# 读取和加载图片
# PILLOW
from PIL import Image
img = Image.open('NLPR1.jpg') # 打开图片
img.size # 获取图片尺寸 width * height
img_rgb = img.load() # 分配内存，并将图像像素值添加到img_rgb对象
img_rgb[width, height] # 通过width/height访问像素值，返回值为RGB模式
# OpenCV
import cv2
img = cv2.imread('NLPR1.jpg') # 打开并将图像像素值添加到img对象
img.shape # 获取图片尺寸 height * width * channels
img[height][width] #通过height/width访问像素值，返回值为BGR模式

# 读取和转换为灰度图
# PILLOW
img = Image.open('NLPR1.jpg') # 打开图片
img_gray = img.convert('L') # 转换为灰度图
pixels_gray = img_gray.load() # 分配内存，并将图像灰度值添加到img_gray对象
# OpenCV
# 直接读取灰度图
img = cv2.imread('NLPR1.jpg', 0) # 将图像转换为灰度图后读取
# 或者
img = cv2.imread('NLPR1.jpg', cv2.IMREAD_GRAYSCALE)
# 先读取彩色图，再转化为灰度图
img = cv2.imread('NLPR.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 将BGR模式的彩色图像转换为灰度图
```

> 参考资料：
>
> 1. [Image.load()](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=load#PIL.Image.Image.load)
> 2. [python opencv将图片转为灰度图](https://blog.csdn.net/sinat_29957455/article/details/84845016)
> 3. [imread()](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
> 4. [ImreadModes](https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)
> 5. [cvtColor](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)
> 6. [ColorConversionCodes](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0)

</br>

6.`cv2.imread()`详解：用于读取图像

该函数的完整调用形式为：`cv.imread(filename[, flags]) ->retval`，`retval`为返回值。下面对各个参数做具体说明（说明针对C++函数版本）：

- `filename`：要读取的图像路径，可以为绝对路径或相对路径

- `flags`：可以对图像采取的`ImreadModes`，常见的有`cv2.IMREAD_GRAYSCALE`、`cv.IMREAD_COLOR`。

调用实例：

```python
img = cv2.imread('NLPR1.jpg') # 打开并将图像像素值添加到img对象
```

PS：该函数对RGB图的默认读取顺序为$BGR$。


> 参考资料：
>
> 1. [imread()](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
> 2. [ImreadModes](https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)

</br>

7.`cv2.cvtColor`详解：用于转换图像

该函数的完整调用形式为：`cv.cvtColor(src, code[, dst[, dstCn]]) ->dst`，`dst`为返回值。下面对各个参数做具体说明（说明针对C++函数版本）：

- `src`：输入图像
- `dst`：输出图像，和`src`有相同的`size`和`depth`，即相同的`height`、`width`和`channles`
- `code`：颜色空间转换模式`ColorConversionCodes`，常见的有`cv2.BGR2GRAY`、`cv2.BGR2RGB`。

- `dstCn`：输出图像的`channels`数，该参数为0时，根据`src`和`dst`自动生成

调用实例：

```python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 将BGR模式的彩色图像转换为灰度图
```

> 参考资料：
>
> 1. [cvtColor](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)
> 2. [ColorConversionCodes](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0)

</br>

8.`cv2.resize()`详解：用于放缩图像

该函数的完整调用形式为：`cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) ->dst`，`dst`为返回值。下面对各个参数做具体说明（说明针对C++函数版本）：

- `src`：输入图像

- `dst`：输出图像，当`dsize`不为0（Python中为`None`）时，其size与`dsize`相同；`dsize`为零时，则与`src.size()`即`fx`和`fy`相同。其类型与`src`相同。

- `dsize`：输出图像尺寸，当`dsize`为0或`None`时，计算方式为：

  `dsize = Size(round(fx*sr.cols), round(fy*src.rows))`

  要么`dsize`非零，要么`fx`和`fy`非零

- `fx`：沿着水平轴的放缩因子，当为0时，可以通过`(double)dsize.width/src.cols`计算

- `fy`：沿着垂直轴的放缩因子，当为0时，可以通过`(double)dsize.height/src.rows`计算

- `interpolation`：插值方法，常见的插值方法有`cv2.INTER_NEAREST`、`cv2.INTER_LINEAR`，默认为`cv2.INTER_LINEAR`。

调用实例：

```python
img = cv2.imread('NLPR1.jpg')
img_resize = cv2.resize(img, (100, 100)) # 通过dsize 放缩
# 或者
img_resize = cv2.resize(img, None, fx=0.5, fy=0.3) # 通过 fx, fy 放缩
```

> 参考资料：
>
> 1. [resize()](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
> 2. [InterpolationFlags](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)

</br>

9.`cv2.transpose()`详解：用于转置矩阵（数组）

该函数的完整调用形式为：`cv.transpose(src[, dst]) ->dst`。下面对各个参数做具体说明（说明针对C++函数版本）：

- `src`：输入数组
- `dst`：和`src`相同类型的输出数组

其效果为：`dst(i, j) = src(j, i)`。调用实例：

```python
img = cv2.transpose(img)
```

>  参考资料：
>
>  1. [transpose()](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga46630ed6c0ea6254a35f447289bd7404)

</br>

10.`cv2.flip()`详解：用于翻转图像（数组）

该函数的完整调用形式为：`cv.flip(src, flipCode[, dst]) ->dst`。下面对各个参数做具体说明（说明针对C++函数版本）：

- `src`：输入数组
- `dst`：输出数组，和`src`具有相同类型和尺寸
- `flitCode`：用来指定怎样翻转数组，0表示沿x轴翻转，正值如1表示沿y轴翻转，-1表示同时沿x轴和y轴翻转，即绕(0,0)翻转

调用实例：

```python
img = cv2.flip(img, 0)
```

> 参考资料：
>
> 1. [flip()](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441)

</br>

11.`cv2.imshow()`详解：用于显示图像

该函数的完整调用形式为：`cv.imshow(winname, mat) ->None`。下面对各个参数做具体说明（说明针对C++函数版本）：

- `winname`：窗口名，显示在窗口顶栏
- `mat`：要显示的矩阵

调用示例：

```python
img = cv2.imread('NLPR.jpg')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> 参考资料：
>
> 1. [imshow()](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)

</br>

12.`cv2.waitKey()`详解：用于指定窗口打开时间

该函数的完整调用形式为：`cv.waitKey([, delay]) ->retval`。下面对各个参数做具体说明（说明针对C++函数版本）：

- `delay`：表示窗口的持续时间，为0时表示保持窗口打开，为其他正值时表示窗口持续的毫秒数

调用示例：

```python
cv2.imshow('img', img)
cv2.waitKey(1000)
```

>  参考资料：
>
>  1. [waitKey()](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7)

</br>

13.`cv2.destroyAllWindows()`详解：用于关闭所有 HightGUI 窗口

该函数的完整调用形式为：`cv.destroyAllWindows() ->None`。

调用实例：似乎不会关闭窗口

```python
cv2.imshow('img', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

PS：`cv2.destroyWindows(winname)`用于关闭指定窗口

> 参考资料：
>
> 1. [destroyWindow()](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga6b7fc1c1a8960438156912027b38f481)

</br>

14.python中获取当前目录路径和上级路径：

```python
import os

print('***获取当前目录***')
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__))) # __file__表示文件名

print '***获取上级目录***'
print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(os.path.abspath(os.path.dirname(os.getcwd())))
print(os.path.abspath(os.path.join(os.getcwd(), "..")))

print '***获取上上级目录***'
print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
```

>  参考资料：
>
>  1. [python获取当前目录路径和上级路径](https://blog.csdn.net/leorx01/article/details/71141643)
>  2. [Python获取当前文件路径](https://www.jianshu.com/p/bfa29141437e)

</br>

15.`cv2.GaussianBlur()`详解：用于给图片添加高斯模糊

该函数的完整调用形式为：

`cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) ->dst`

下面对各个参数做具体说明（说明针对C++函数版本）：

- `src`：输入图像，图像有不同的通道，会被分别处理
- `dst`：输出图像，类型和尺寸与`src`相同
- `ksize`：高斯核尺寸。`ksize.width`和`ksize.height`必须为正奇数；否则`ksize`为0，此时其值通过`sigmaX`和`sigmaY`计算
- `sigmaX`：在X方向上的 Gaussian kernel standard deviation
- `sigmaY`：在Y方向上的 Gaussian kernel standard deviation。为0时设为与`sigmaX`相等。
- `borderType`：像素外插值方法，常见的有`cv.BORDER_CONSTANT`、`cv.BORDER_REPLICATE`，不支持`cv.BORDER_WRAP`。

为了完全控制对图片的操作而不需要管OpenCV后续对语义的修改，建议对`ksize`/`sigmaX`/`sigmaY`都进行指定。

调用实例：

```python
depth_img = cv2.imread('NLPR2_depth.png', 0)
# 给深度图添加高斯模糊
depth_img = cv2.GaussianBlur(depth_img, (31, 31), sigmaX=0)
```

> 参考资料：
>
> 1. [GaussianBlur()](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
> 2. [BorderTypes](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)
> 2. [高斯模糊](https://zh.m.wikipedia.org/zh/%E9%AB%98%E6%96%AF%E6%A8%A1%E7%B3%8A)
> 2. [高斯模糊的原理是什么，怎样在界面中实现](https://www.zhihu.com/question/54918332)

</br>

16.`cv2.threshold()`：用于对每个数组元素应用固定水平的阈值。该函数通常用于从一个灰度图中得到二值图像或者去除噪音（即过滤掉太小或太大的值）。该函数支持几种阈值，通过类型参数来设置。

与此同时，`THRESH_OTSU`和`THRESH_TRIANGLE`可以联合上述的值来使用。这种情况下，函数通过 Otsu 或者 Triangle 算法来计算最优的阈值。

PS：目前，Otsu 算法和 Triangle 算法只在 8-bit 的单通道图像上实现

该函数的完整调用形式为：

`cv.threshold(src, thresh, maxval, type[, dst]) ->retval, dst`

- `src`：输入数组（多通道，8-bit 或者 32-bit 浮点数）
- `dst`：和`src`同尺寸、类型和通道的输出数组
- `thresh`：阈值
- `maxval`：在使用`THRESH_BINARY`和`THRESH_BINARY_INV`参数时的最大值
- `type`：阈值类型，常见的有`THRESH_BINARY`和`THRESH_BINARY_INV`。

> 参考资料：
>
> 1. [threshold()](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)
> 2. [ThresholdTypes](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)
> 3. [OpenCV-Python入门教程6-Otsu阈值法](https://www.cnblogs.com/gezhuangzhuang/p/10295181.html)
> 4. [OTSU算法（大津法）原理解析](https://zhuanlan.zhihu.com/p/395708037)

</br>

17.`cv2.imwrite()`：用于存储图片到指定文件，图片类型取决于文件后缀名。其完整声明形式如下：

```python
cv.imwrite(filename, img[, params]) -> retval
```

- `filename`：存储文件名
- `img`：图片数据对应的矩阵
- `params`：成对的指定存储格式的参数。

> 参考资料：
>
> 1. [imwrite()](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)

</br>

18.`cv2.Canny()`：用于通过$Canny$算法查找图像边缘。其完整声明形式如下：

```c++

void cv::Canny	(InputArray image,
	OutputArray edges,
	double threshold1,
	double threshold2,
	int apertureSize = 3,
	bool L2gradient = false 
	)
```

该函数在输入的图像中查找边缘并通过$Canny$算法在输出中标记出它们。在threshold1和threshold2中最小的值将用于edge linking，最大值将用于查找初始的更为强烈/显著的边缘。具体见 [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)

- `image`：8-bit的输入图片
- `edges`：输出的 edge map，8-bit单通道，和`image`尺寸相同
- `threshold1`：滞后过程（hysteresis procedure）的第一个阈值
- `threshold2`：滞后过程的第二个阈值
- `apertureSize`：$Sobel$操作子的孔径尺寸
- `L2gradient`：a flag。表明是否使用更准确的$L2$范数$\sqrt{(dI/dx)^2+(dI/dy)^2}$计算图像梯度大小（`L2gradient=true`），还是使用默认的$L1$范数$\sqrt{|dI/dx|+|dI/dy|}$（`L2gradient=false`）。

调用实例：

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
```

参考资料：

1. [Canny()](https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de)
2. [Canny Edge Detection in OpenCV](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html)
3. [Canny算子](https://zh.m.wikipedia.org/zh-hans/Canny%E7%AE%97%E5%AD%90)
4. [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
5. [Canny边缘检测](https://zj-image-processing.readthedocs.io/zh_CN/latest/opencv/code/[Canny]%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B/)
6. [Python实现Canny算子边缘检测](https://yueyue200830.github.io/2020/04/04/Python%E5%AE%9E%E7%8E%B0Canny%E7%AE%97%E5%AD%90%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B/)

</br>

19.`cv2.Sobel()`：使用$Sobel$算子计算图像导数。其完整声明形式如下：

```c++
void cv::Sobel(InputArray src,
	OutputArray dst,
	int ddepth,
	int dx,
	int dy,
	int ksize = 3,
	double scale = 1,
	double delta = 0,
	int borderType = BORDER_DEFAULT 
	)	
```

该函数通过用合适的核与函数做卷积来计算图像导数。通常，该函数通过`xorder=1, yorder=0, ksize=3`和`xorder=0, yorder=1, ksize=3`来计算图像的一阶$x$和$y$导数，分别对应：
$$
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix} 和
\begin{bmatrix}
-1 & -2 & 1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$
这两个核。

- `src`：输入图片
- `dst`：相同尺寸和通道数的输出图片
- `ddepth`：输出图片depth，见 [combinations](https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#filter_depths)。（depth指图片的数据类型，例如对应图像梯度你想要16bit而不是8bit。当输入图片为8-bit类型时，将导致梯度裁剪（精度不够）
- `dx`：梯度$x$的order
- `dy`：梯度$y$的order
- `ksize`：$Sobel$核尺寸，必须是1/3/5/7。
- `scale`：计算梯度值时的可选因子，默认不提供
- `delta`：在`dst`中存储结果之前默认加到结果上的可选delta值。
- `borderType`：像素插值类型。

参考资料：

1. [Soble()](https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)
2. [Sobel算子](https://zh.wikipedia.org/wiki/%E7%B4%A2%E8%B2%9D%E7%88%BE%E7%AE%97%E5%AD%90)
3. [边缘检测](https://www.cnblogs.com/zhuifeng-mayi/p/9563947.html)

</br>

20.`numpy.ndarray.flatten()`：返回一个坍缩成一维的数组的副本。

调用实例：

```python
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
```

参考资料：

1. [numpy.ndarray.flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

</br>

21.`math.atan(x)`vs`math.atan2(y, x)`：

`atan`返回`x`对应的arc tangent，其结果所属区间为$(-\pi/2, \pi/2)$；

`atan2(y, x)`则返回`atan(y/x)`，其结果所属区间为$(-\pi, \pi)$。

举例来说，`atan(1)=atan2(1,1)`$=\pi/4$；`atan2(-1,-1)`$=-3\pi/4$。

> 参考文献：
>
> 1. [math.atan(x)](https://docs.python.org/3.8/library/math.html?highlight=atan#math.atan)

</br>

22.`numpy.zeros`：用于返回给定`shape`和`type`的用零填充的新的数组。其完整调用形式为：

```python
numpy.zeros(shape, dtype=float, order='C', *, like=None)
```

- `shape`：指定数组形状，如`(2, )`或`(2,3)`。
- `dtype`：指定填充的数据类型，如`dtype=int`，默认为`numpy.float64`。
- `order`：指定行优先还是列优先，默认为`order='C'`，行优先，`order='F'`表示列优先。
- `like`：引用对象，用于创建非Numpy arrays类型的数组，可以兼容其他类型的数组。

调用实例：

```python
>>> np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])
>>> np.zeros((5,), dtype=int)
array([0, 0, 0, 0, 0])
```

PS：`numpy.ones`与之类似

> 参考资料：
>
> 1. [numpy.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
> 2. [numpy.ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html#numpy.ones)

</br>

23.在进行灰度图转为BGR图`img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)`之前，需要确保灰度图`gray`中数值的类型满足转换要求，否则会出现如下错误：

![image-20220626163339376](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220626163339376.png)

可以通过`gray = np.uint8(gray)`将灰度图转换为满足条件的格式：

> 参考资料：
>
> 1. [Opencv error -Unsupported depth of input image:](https://stackoverflow.com/questions/55179724/opencv-error-unsupported-depth-of-input-image)

</br>

24.`numpy.sum`：计算给定`axis`上数组的元素之和。其完整调用形式为：

```python
numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)[source]
```

- `a`：数组类型的对象
- `axis`：None、整数或整数类型的元组，可选。默认为None，将对所有元素求和。
- `dtype`：数据类型，可选。
- `out`：用于存放求和结果的输出数组，可选。
- 略

调用实例：

```python
>>> a = np.ones((10,2), dtype=int)
>>> a.sum()
20
>>> a.sum(axis=0)
array([10, 10])
>>> a.sum(axis=0)/6
array([1.66666667, 1.66666667])
```

> 参考资料：
>
> 1. [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)

</br>

























