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
