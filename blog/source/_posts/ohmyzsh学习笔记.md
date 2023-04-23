---
title: ohmyzsh学习笔记
date: 2023-04-20 22:30:01
categories:
- 环境配置
tags:
- tmux 
---

本文记录一下在使用ohmyzsh过程中碰到的问题：
<!--more-->

### 报错`job table full or recursion limit exceeded`：
原因：在`.bash_aliases`中出现了如下代码，导致循环导入：
```bash
if [ -f ~/.bash_aliases ]; then
 . ~/.bash_aliases
fi
```
解决方案：注释掉即可

### 报错`[process exited with code 11 (0x0000000b)]`
原因：`.bashrc`中出现了问题，暂未排查出来
解决方案：将`.bashrc`被分为`.bashrc.bak`，新建`.bashrc`文件并写入配置。（建议直接使用zsh替代bash）。

### 查看commands历史并使用
- `history`：查看commands历史
- `!!`：选择最近命令
- `!12`：选择`history`中的第12条命令
> 参考资料：
> 1. [How to Search in My ZSH History](https://linuxhint.com/search-in-my-zsh-history/)

