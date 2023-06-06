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
> 2. [python - ModuleNotFoundError: No module named 'distutils.util' - Ask Ubuntu](https://askubuntu.com/questions/1239829/modulenotfounderror-no-module-named-distutils-util)

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
2. 方法2：使用7+ Taskbar Tweaker管理Taskbar，推荐。配置如下：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230203201506.png)
> 参考资料：
> 1. [VS Code multiple taskbar icon - Stack Overflow](https://stackoverflow.com/questions/63381934/vs-code-multiple-taskbar-icon)
> 2. [88：7+ Taskbar Tweaker 最新 5.1一款功能强大的Windows任务栏自定义设置工具使用教程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1VV411n7hW)

### 其中的文件夹或文件已在另一程序中打开

> 参考资料：
> 1. [想删除这个东西但它显示的是在另一文件夹中或程序中打开，无法删除，怎么办？ - 知乎](https://www.zhihu.com/question/453864187/answer/1827894565)


### Google同步多个标签页

> 参考资料：
> 1. [Sync OneTab tabs on Chrome across different PC's - Super User](https://superuser.com/questions/630975/sync-onetab-tabs-on-chrome-across-different-pcs)

### Windows重启资源管理器
重命名文件时，包含特殊字符会导致桌面卡住，无法进行桌面文件选择和点击操作：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230415122100.png)
此时需要重启windows资源管理器，但在任务管理器中有时又没有该任务。此时需要启动资源管理器然后对它进行重启。操作步骤如下：
- `Windows + R`键打开运行窗口，输入`expolorer.exe`回车启动资源管理器，（不要关闭打开的文件资源管理器窗口）：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230415122405.png)
- 此时会发现任务管理器中出现Windows Explorer任务，右键重启即可：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230415122528.png)
> 参考资料：
> 1. [Windows 10 计算机如何重启文件资源管理器 | 华为官网](https://consumer.huawei.com/cn/support/content/zh-cn00733776/)

### PPT (Powerpoint) 在多个窗口打开多个文件
视图->层叠窗口

### 同步时间
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230604143409.png)


### 添加系统http_proxy和https_proxy环境变量

> 参考资料：
> 1. [Using the cf CLI with a proxy server | Cloud Foundry Docs](https://docs.cloudfoundry.org/cf-cli/http-proxy.html) 