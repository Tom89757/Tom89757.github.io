---
title: VSCode配置
date: 2022-05-21 17:09:43
categories:
- 环境配置
tags:
- VSCode
---



本文记录一下在 VSCode 编辑器中一些常用的配置方法。

<!--more-->

### VSCode 重新启用“错误波形提示”

有两种方式：

- 手动：
  1. 文件——首选项——设置（files->preferences->settings），搜索 `error Squiggles`。
  2. 在用户 (user) 和工作区 (workspace) 都选择`enable`。

- 快捷键：在命令面板通过快捷键 (Ctrl + Shift + P) 打开搜索栏，搜索 `Error Squiggles`，选择`enable`。

  PS：只对某些 extensions 有用，如 C/C++ 插件

> 参考资料：[VSCode重新启用“错误波形提示”](https://blog.csdn.net/HermitSun/article/details/103627053)

### VSCode 配置快捷键（shortcuts）

有两种方式：

- 手动：`File -> Preferences -> Keyboard Shortcuts`，即可打开窗口手动编辑。

- `json`文件：在上述打开的窗口中点击右上角的如图按钮编辑`json`文件：

  ![image-20220523101335662](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220523101335662.png)

### VSCode 在多个窗口中打开同一项目

- `File -> Add Folder to Workspace`，添加目标项目到 workspace：

  ![image-20220523101543021](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220523101543021.png)

- `File -> Dulplicate Workspace`，重复该工作区：

  ![image-20220523101705028](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220523101705028.png)

### VSCode显示85字符宽度提示线

- `File->Preferences->Settings`，搜索`editor.rulers`：

  ![image-20220625185036578](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220625185036578.png)

- 打开`settings.json`，编辑`editor.rulers`的值为85：

  ![image-20220625185247521](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220625185247521.png)

> 参考资料：
>
> 1. [vs code设置每行代码的垂直标尺](https://blog.csdn.net/qq_43406338/article/details/109397831)
