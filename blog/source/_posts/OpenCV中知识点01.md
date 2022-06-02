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
- `ksize`：高斯核尺寸。`ksize.width`和`ksize.height`必须为正奇数；否则为0，此时通过`sigmaX`和`sigmaY`计算
- `sigmaX`：在X方向上的 Gaussian kernel standard deviation
- `sigmaY`：在Y方向上的 Gaussian kernel standard deviation。为0时设为与`sigmaX`相等。
- `borderType`：像素外插值方法，常见的有`cv.BORDER_CONSTANT`、`cv.BORDER_REPLICATE`，不支持`cv.BORDER_WRAP`。

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

