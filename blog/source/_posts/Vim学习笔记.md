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

### Vim复制、粘贴然后替换
以下面代码为例：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191509.png)
我们的目的是将`pass`替换为`if __name__ == "__main__"`。步骤如下：
1.在visual模式下选择`if __name__ == "__main__"`，然后在normal模式下按下`y`复制。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191750.png)
2.在visual模式下选择`pass`，按下`y`即可进行替换：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191812.png)
便可得到下述结果：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191848.png)
> 参考资料：
> 1. [Copy, delete, then paste in Vim - Super User](https://superuser.com/questions/371160/copy-delete-then-paste-in-vim)


### Vim移动
1.`w`、`b`、`e`。`e`用于移动到光标所在位置的下一个单词的末尾。
2.`W`、`B`、`E`。跳转到下一个空格。
3.`%`。跳到对应的括号。

### Vim Tab页切换
1.`gt`：右边的tab页。已经映射为`<leader>l`。
2.`gT`：左边的tab页。已经映射为`<leader>h`。

### leetcode网页端Vim模式
问题1：按下`Esc`键会跳出编辑窗口。
解决方案：编辑前使用`:imap jj <Esc>`进行映射，或者使用`Ctrl+C`进入normal模式。
> 参考资料：
> 1. [VIM Command Mode](https://leetcode.com/discuss/general-discussion/446981/vim-command-mode)

### Vim配置特定主题

> 参考资料：
> 1. [Best vim color schemes and how to install](https://linuxhint.com/best_vim_color_schemes/)


### Vim禁止闪烁

> 参考资料：
> 1. [vim - Disable blinking at the first/last line of the file - Stack Overflow](https://stackoverflow.com/questions/5933568/disable-blinking-at-the-first-last-line-of-the-file)
> 2. [terminal - How to prevent Vim from making a flashy screen effect when pressing `ESC` or `^[` in normal mode? - Vi and Vim Stack Exchange](https://vi.stackexchange.com/questions/22547/how-to-prevent-vim-from-making-a-flashy-screen-effect-when-pressing-esc-or)


### git bash Vim主题显示和对应主题不一致

> 参考资料：
> 1. [Reddit - Dive into anything](https://www.reddit.com/r/vim/comments/ebylxb/vim_from_git_bash_showing_weird_colorscheme/)

### Vim复制到剪切板 clipboard
参考资料3设置有效
> 参考资料：
> 1.[What is difference between Vim's clipboard "unnamed" and "unnamedplus" settings? - Stack Overflow](https://stackoverflow.com/questions/30691466/what-is-difference-between-vims-clipboard-unnamed-and-unnamedplus-settings) 
> 2.  [Windows Subsystem Linux - Make VIM use the clipboard? - Super User](https://superuser.com/questions/1291425/windows-subsystem-linux-make-vim-use-the-clipboard)
> 3. [Vim Wsl Clipboard](https://waylonwalker.com/vim-wsl-clipboard/)