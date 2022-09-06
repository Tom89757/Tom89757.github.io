---
title: Windows10常见操作
date: 2022-05-22 20:30:33
categories:
- 环境配置
tags:
- Windows10
---

本文记录一下使用 Windows10 过程中的常见操作：

<!--more-->

### 查看 Windows版本信息

- `Win + R`打开窗口
- 窗口中输入 `winver` 回车，即可查看 Windows10 版本信息

### 查看Windows .NET Framework版本

打开资源管理器，再地址栏输入：`%systemroot%\Microsoft.NET\Framework`，可以看到 .Net Framework 安装目录下列出的版本

> 参考资料：
>
> 1. [查看本机.NET Framework版本](https://blog.csdn.net/zyw_anquan/article/details/9873047)

### 在没有安装pip或pip被卸载的情况下安装pip

- 下载`get-pip.py`脚本，[here](https://bootstrap.pypa.io/get-pip.py)
- 运行`python get-pip.py`

>  参考资料：
>
> 1. [pip installation](https://pip.pypa.io/en/stable/installation/)

### 更新pip

`python -m pip install --upgrade pip`。

参考资料：

> 1. [pip installation](https://pip.pypa.io/en/stable/installation/)

### 键盘键位映射修改
问题：笔记本left shift键失灵，将Caps Lock键映射到left shift键然后作为left shift键使用
> 参考资料：
> 1. [键盘键位修改及管理（Windows篇）](https://zhuanlan.zhihu.com/p/29581818)
> 2. [Windows：修改键盘映射表](https://blog.csdn.net/qq_42191914/article/details/104840458)

