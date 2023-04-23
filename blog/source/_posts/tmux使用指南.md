---
title: tmux使用指南
date: 2023-04-19 17:24:29
categories:
- 环境配置
tags:
- tmux 
---

本文记录一下tmux的学习和使用过程：
<!--more-->

### 基础使用
主要参考资料4，资料5中包含了较多进阶配置。
> 参考资料：
> 1. [Tmux Cheat Sheet & Quick Reference](https://tmuxcheatsheet.com/)
> 2. [手把手教你使用终端复用神器 tmux_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1KW411Z7W3)
> 3. [GitHub - tmux/tmux: tmux source code](https://github.com/tmux/tmux)
> 4. [Tmux 使用教程 - 阮一峰的网络日志](https://www.ruanyifeng.com/blog/2019/10/tmux.html)
> 5. [Tmux使用手册 | louis blog](http://louiszhai.github.io/2017/09/30/tmux/)


### 报错`sessions should be nested with care, unset $TMUX to force`
tmux布建议在一个已经active的tmux session嵌套另一个session。
> [Tmux sessions should be nested with care, unset $TMUX to force](https://koenwoortman.com/tmux-sessions-should-be-nested-with-care-unset-tmux-to-force/)