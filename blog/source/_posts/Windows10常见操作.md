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

### 网络问题排查
> 参考资料：
> 1. [网络出了问题，如何排查? 这篇文章告诉你](https://www.51cto.com/article/620620.html)

### 在不断开网线的情况下断网
控制面板->网络和共享中心->更改适配器设置->以太网右键禁用
> 参考资料：
> 1. [如何不拔网线断网，怎么断开电脑网络本地连接](https://jingyan.baidu.com/article/0964eca27410968285f53613.html)

### 如何使得VSCode可以有多个taskbar icon以方便切换
如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230112140219.png)
1. 方法1：
Windows 10打开Taskbar设置：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230112140315.png)
改为：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230112140336.png)
但这会对所有软件生效。
2. 方法2：使用7+ Taskbar Tweaker管理Taskbar，推荐
> 参考资料：
> 1. [VS Code multiple taskbar icon - Stack Overflow](https://stackoverflow.com/questions/63381934/vs-code-multiple-taskbar-icon)
> 2. [88：7+ Taskbar Tweaker 最新 5.1一款功能强大的Windows任务栏自定义设置工具使用教程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1VV411n7hW)
