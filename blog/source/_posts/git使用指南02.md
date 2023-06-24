---
title: git使用指南02
date: 2023-06-15 23:44:08
categories:
- 开发工具
tags:
- git
---

本文记录一下使用git时的常见操作。
<!--more-->

1.git更新版本并添加到Windows Terminal：

> 参考资料：
> 1. [How to Upgrade git to the Latest Version on Windows](https://linuxhint.com/upgrade-git-latest-version-windows/)
> 2. [Adding Git-Bash to the new Windows Terminal - Stack Overflow](https://stackoverflow.com/questions/56839307/adding-git-bash-to-the-new-windows-terminal)

</br>

2.git terminal在VSCode中行为很奇怪：
更新git版本。
> 参考资料：
> 1. [git bash terminal is acting wierd in vscode · Issue #184719 · microsoft/vscode · GitHub](https://github.com/microsoft/vscode/issues/184719)
> 2. [visual studio code - git bash terminal is acting wierd in vscode - Stack Overflow](https://stackoverflow.com/questions/76479076/git-bash-terminal-is-acting-wierd-in-vscode)


</br>
3.git bash终端中文显示乱码：
解决方案1（无效）：
- `git config --global core.quotepath false`
- 在设置里将`text`改为`zh-CN`和`UTF-8`。
解决方案2：
- `chcp.com 65001`：`65001`对应UTF-8，可以写在`.zshrc`文件中。
> 参考资料：
> 1. [Git Bash终端中文输出显示乱码解决方案 - lybingyu - 博客园](https://www.cnblogs.com/sdlz/p/13023342.html)
> 2. [windows - Unicode (utf-8) with git-bash - Stack Overflow](https://stackoverflow.com/questions/10651975/unicode-utf-8-with-git-bash)