---
title: tmux学习笔记
date: 2023-04-23 13:35:33
categories:
- 环境配置
tags:
- tmux 
---
本文记录一下学习tmux过程中的经验和总结：
<!--more-->

### alias配置
```bash
alias tl='tmux list-sessions'
alias tkss='tmux kill-session -t'
alias ta='tmux attach -t'
alias td='tmux detach'
alias ts='tmux new-session -s'
```
> 参考资料：
> 1. [mac上使用oh my zsh有哪些必备的插件推荐？ - 知乎](https://www.zhihu.com/question/49284484)
