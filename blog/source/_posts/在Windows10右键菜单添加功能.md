---
title: 在Windows10右键菜单添加功能
date: 2022-05-15 16:44:03
categories:
- 环境配置
tags:
- Windows
---

本文记录一下在Windows10右键菜单中添加各项功能的流程：

<!--more-->

1.Windows10 右键新建菜单中添加`.md`文件。

步骤如下：

- 安装 Typora，一个`md`文件编辑器
- 键盘 `Win + R`快捷键打开窗口，输入`regedit`打开注册表编辑器 (registry editor)
- 定位到 `HKEY_CLASSES_ROOT\.md` 
- 点击 `.md` 文件夹，双击右侧 (默认，Default) 项编辑字符串，将数值数据 (value data) 改为 `Typora.md`
- 右键 `.md` 文件夹，新建项 (key) ，将新建项命名为 `ShellNew`
- 右键 `ShellNew` ，新建字符串项 (string value) ，把新建的字符串项改为 `NullFile`

结果如图：

![image-20220515202646881](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515202646881.png)

> 参考资料：[Win10下如何在右键新建菜单中添加.md文件](https://www.zhihu.com/question/267616299)
>
> PS：答案有误，应该将答案中的 `Typora.exe` 改为 `Typora.md`

</br>

2.Windows10 右键菜单中添加在此处打开WSL (Windows Subsystem for Linux)。

步骤如下：

- 键盘 `Win + R`快捷键打开窗口，输入`regedit`打开注册表编辑器 (registry editor)
- 定位到 `HKEY_CLASSES_ROOT\Directory\Background\shell`
- 点击`shell`文件夹，新建项 (key) ,将新建项命名为`Ubuntu`
- 点击`Ubuntu`文件夹，双击右侧 (默认，Default) 项编辑字符串，将数值数据 (value data) 改为 `Open Ubuntu here :-)`
- 右键`Ubuntu`文件夹，新建项 (key) ，并将其命名为`command`，点击`command`文件夹，双击右侧 (默认，Default) 项编辑字符串，将数值数据 (value data) 改为 `C:\Windows\System32\wsl.exe` (即wsl.exe文件所在位置)
- 右键`Ubuntu`文件夹，新建字符串项 (string value) ，名称设为 Icon，值改为`C:\Windows\System32\wsl.exe`。此操作用于给右键WSL添加图标

结果如图：

![image-20220515202553533](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515202553533.png)

> 参考资料：[将 Windows10 中的 WSL 添加至右键菜单](https://www.i4k.xyz/article/gulang03/79177500)

</br>
