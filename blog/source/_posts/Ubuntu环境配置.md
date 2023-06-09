---
title: Ubuntu环境配置
date: 2023-06-09 16:12:41
categories:
- 环境配置
tags:
- Ubuntu 
---
本文记录一下Ubuntu系统的环境配置：
<!--more-->

### tldr
以简洁的方式说明各个常用命令如`tar`的用法。
安装：
- `npm install -g tldr`：node.js中安装
- `pip3 install tldr`：linux中安装
使用：
- `tldr git`：即可查看git的用法，第一次使用时需要建立索引
PS：根据参考资料2和3，在gitbash中`tldr`的输出没有颜色，此时可以在`.zshrc`中添加`export FORCE_COLOR=2`。
> 参考资料：
> 1. [GitHub - tldr-pages/tldr: 📚 Collaborative cheatsheets for console commands](https://github.com/tldr-pages/tldr)
> 2. [How to get colors? · Issue #1262 · tldr-pages/tldr · GitHub](https://github.com/tldr-pages/tldr/issues/1262)
> 3. [tldr doesnt pick color config from .tldrrc file · Issue #276 · tldr-pages/tldr-node-client · GitHub](https://github.com/tldr-pages/tldr-node-client/issues/276)
