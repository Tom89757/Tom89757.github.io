---
title: VSCode和Idea快捷键记录
date: 2023-04-07 00:02:41
categories:
- 开发工具
tags:
- VSCode
- IDEA
---
本文记录一下在VSCode和IDEA中所配置的快捷键以便查阅：
<!--more-->
### 编辑 
1.`Ctrl+Shift+D`：选中光标位置所在的单词出现。
2.`Ctrl+Shift+L`：选中光标位置所在单词的所有出现。此时使用`i`或`c/d`可以实现多光标编辑（VSCode和IDEA中的Vim插件使用`i`有点问题，可使用`c`作为替代）
3.`Ctrl+Right/Left`定位选中单词的下/上一个出现。（VSCode，IDEA中配置快捷键失效）
4.`Ctrl+D/U`上下移动整个屏幕。
5.`gd`跳转到函数/变量定义处。
6.`ge/gE`跳转到下/上一个error处。
7.`gm/gM`跳转到下/上一个method处。（IDEA）
8.`<leader><leader>w/b`快速定位光标前后单词。（通过easymotion实现）
9.`s+目标单词的前两个字母`快速定位光标所在行的目标单词。（通过sneak实现）
10.`Ctrl+;`输入字母快速定位文件中该字母的出现位置。（通过ace jump插件实现）。

### 切换
1.`<leader>h`和`<leader>l`切换tab页。
2.`<leader>n`定位explorer，通过`j/k`等选择、新建、重命名文件。（通过NERDTree插件实现）
3.`Ctrl+P`根据文件名搜索explorer中的所有文件并通过enter打开。
4.`~ + 1`定位文件进行编辑，`~ + 2`定位终端窗口。


### 其它
1.`Ctrl+Shift+F`全局搜索。
2.`Ctrl+,`打开settings。
3.`F2`搜索commands（VSCode）。
4.`Ctrl+Shift+R`重载窗口（VSCode）。
5.`Ctrl+T`打开终端。
6.`Ctrl+w`或者`<leader>q`关闭当前文件。
7.`Ctrl+K+W`关闭所有文件。
