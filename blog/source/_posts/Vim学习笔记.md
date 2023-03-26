---
title: Vim学习笔记
date: 2022-11-20 13:05:32
categories:
- 工具
tags:
- Vim
---
本文记录一下在学习Vim过程中的经验和总结：
<!--more-->

### Vim删除查找匹配的行
例如，要全局替换掉包含`file:`的行：
```vim
:g/file:/d
```
> 参考资料：
> 1. [vim 删除匹配行_中国风2012的博客-CSDN博客_vim 删除匹配行](https://blog.csdn.net/hanshileiai/article/details/50528505)

