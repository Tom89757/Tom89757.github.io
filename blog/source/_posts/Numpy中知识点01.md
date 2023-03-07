---
title: Numpy中知识点01
date: 2023-02-11 17:40:10
categories:
- 深度学习
tags:
- Numpy
- python
---

本文记录一下Numpy相关知识点：
<!--more-->

1.numpy创建数组：
```python
import numpy as np
x = np.zeros(shape, dtype = np.uint8)
```
> 参考资料：
> 1. [NumPy 创建数组 | 菜鸟教程](https://www.runoob.com/numpy/numpy-array-creation.html)

</br>
2.numpy数组转换为dataframe：
```python
import numpy as np
import pandas as pd

my_array = np.array([[11,22,33],[44,55,66]])

df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

print(df)
print(type(df))
```
> 参考资料：
> 1. [How to Convert NumPy Array to Pandas DataFrame – Data to Fish](https://datatofish.com/numpy-array-to-pandas-dataframe/)

</br>
3.numpy矩阵乘法：
```python
>>> import numpy as np
>>> a = np.array([[1,1],[1,1]])
>>> a
array([[1, 1],
       [1, 1]])
>>> b = np.array([[1,2],[3,4]])
>>> b
array([[1, 2],
       [3, 4]])
>>> np.multiply(a,b)
array([[1, 2],
       [3, 4]])
>>> np.dot(a,b)
array([[4, 6],
       [4, 6]])
```
>参考资料：
>1. [NumPy矩阵乘法](http://c.biancheng.net/numpy/matrix-multiplication.html)









