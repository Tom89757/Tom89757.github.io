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
在笔记库根目录添加`.obsidian.vimrc`文件，并在里面添加如下按键映射：
```json
" I like using H and L for beginning/end of line
nmap H ^
nmap L $
"map j j to Esc"
imap jj Esc
" Have j and k navigate visual lines rather than logical ones
nmap j gj
nmap k gk
```

> 参考资料：
> 1. [Obsidian 中使用 Vim 模式并配置 Vimrc | Verne in GitHub](https://einverne.github.io/post/2022/07/obsidian-vim-and-vimrc.html)
> 2. [Obsidian中使用Vim的最佳实践？ · Discussion #10 · obsidianzh/forum · GitHub](https://github.com/obsidianzh/forum/discussions/10)
> 3. [从零配置Obsidian-像Vim一样跳转_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ZN4y1j73m)
> 4. [obsidian-vimrc-support](https://github.com/esm7/obsidian-vimrc-support)
> 5. [jj imap to Esc not work ? · Issue #4 · esm7/obsidian-vimrc-support · GitHub](https://github.com/esm7/obsidian-vimrc-support/issues/4)
### 问题
#### 4.1 将VSCode代码粘贴到代码块中时出现多于空行
解决方案：右键以纯文本形式粘贴
参考资料：
1.  [【已解决】在阅读模式下，代码块中的代码不高亮](https://forum-zh.obsidian.md/t/topic/7496)
2. [格式化你的笔记](https://publish.obsidian.md/help-zh/%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/%E6%A0%BC%E5%BC%8F%E5%8C%96%E4%BD%A0%E7%9A%84%E7%AC%94%E8%AE%B0)
3. [求助：关于在Obsidian上粘贴代码每行会出现多余空行如何解决？](https://forum-zh.obsidian.md/t/topic/7432)
### 4.2 有时会出现打开某个仓库安装插件全部消失的情况
这是因为Obsidian的每个仓库的配置独立，即每个仓库文件夹下都有独立的`.obsidian`文件夹，对应不同的配置。为了解决该问题，可以将某仓库下的`.obsidian`复制到另一仓库下以实现配置迁移的目的。
### 4.3 使用Vim模式编辑时代码缩进快捷键失效
一般模式缩进快捷键为`Ctrl+[`和`Ctrl+]`。使用Vim编辑时可以使用`Tab`和`Shift+Tab`替代。