---
title: Obsidian配置
date: 2022-08-14 12:25:34
categories:
- 环境配置
tags:
- VSCode
---

本文记录一下在Obsidian中一些常用的配置和使用技巧
<!--more-->
### 使用教程

### 主题
- Minimal：不太好用，短代码不会高亮
- Obsidian-Typora-Vue：[Obsidian-Typora-Vue-Theme](https://github.com/ZekunC/Obsidian-Typora-Vue-Theme)，正在使用
### 插件
- Image auto upload Plugin：和PicGO协同上传图片到github图床
- Editor Syntax Highlight：代码语法高亮
- cMenu：用于支持word的富文本编辑，可自行配置
- Title Serial Number Plugin：用于自动给标题编号
- Auto Title Link：用于自动根据复制的链接拉取网页标题
#### Vim配置
> 参考资料：
> 1. [Obsidian 中使用 Vim 模式并配置 Vimrc | Verne in GitHub](https://einverne.github.io/post/2022/07/obsidian-vim-and-vimrc.html)
> 2. [Obsidian中使用Vim的最佳实践？ · Discussion #10 · obsidianzh/forum · GitHub](https://github.com/obsidianzh/forum/discussions/10)
> 3. [从零配置Obsidian-像Vim一样跳转_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ZN4y1j73m)
### 问题
#### 4.1 将VSCode代码粘贴到代码块中时出现多于空行
解决方案：
1. 右键以纯文本形式粘贴
参考资料：
1.  [【已解决】在阅读模式下，代码块中的代码不高亮](https://forum-zh.obsidian.md/t/topic/7496)
2. [格式化你的笔记](https://publish.obsidian.md/help-zh/%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/%E6%A0%BC%E5%BC%8F%E5%8C%96%E4%BD%A0%E7%9A%84%E7%AC%94%E8%AE%B0)
3. [求助：关于在Obsidian上粘贴代码每行会出现多余空行如何解决？](https://forum-zh.obsidian.md/t/topic/7432)
### 4.2 有时会出现打开某个仓库安装插件全部消失的情况
这是因为Obsidian的每个仓库的配置独立，即每个仓库文件夹下都有独立的`.obsidian`文件夹，对应不同的配置。为了解决该问题，可以将某仓库下的`.obsidian`复制到另一仓库下以实现配置迁移的目的。