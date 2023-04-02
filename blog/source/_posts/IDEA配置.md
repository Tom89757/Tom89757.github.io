---
title: IDEA配置
date: 2022-08-15 10:20:35
categories:
- 环境配置
tags:
- IDEA
---

本文记录一下在IDEA中一些好用的配置：
<!--more-->

### 将快捷键更改为VSCode keymap
1. 安装VSCode keymap插件：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220815102337.png)
2.打开`File->Settings->Keymap`，设置Keymap为VSCode：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220815102443.png)

### IDEA返回上一次光标所在位置
同 [Android Studio返回上一次光标所在位置](https://tom89757.github.io/2022/07/27/Android-Studio%E9%85%8D%E7%BD%AE)

### 解决IDEA打开某个项目卡住（白屏）
其解决思路为删除IDEA在本地保存的该项目的状态文件；
其代价为需要重新导入该项目的各个模块。
> 参考资料：
> 1. [解决 idea 打开某个项目卡住 (白屏)](http://digtime.cn/articles/534/jie-jue-idea-da-kai-mou-ge-xiang-mu-ka-zhu-bai-ping)

### 解决IDEA无法指定compile output path的问题
如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220916000324.png)
有时上图中的Compile output路径无法通过浏览目录指定，此时可以直接复制目录完整路径到该选项，如上图中的`D:\Desktop\CS61B\out`，应用后重启项目即可。

### 对ideaVim中的键进行重新映射
1. 在`C:\Users\user\`目录下创建`.ideavimrc`文件。
2. 在其中添加如下内容：
```python
" I like ujsing H and L for beginning/end of line
:nmap H ^
:nmap L $
"map j j to Esc"
imap jj <Esc>
" Have j and k navigate visual lines rather than logical ones
:nmap j gj
:nmap k gk

" Yank to system clipboard"
:set clipboard=unnamed

“ set <leader> to <space>
let mapleader = " "
set easymotion
set sneak
```
> 参考资料：
> 1. [vim - Intellij IdeaVim change keys - Stack Overflow](https://stackoverflow.com/questions/10149187/intellij-ideavim-change-keys)
> 2. [dotfiles/.ideavimrc at master · fdietze/dotfiles · GitHub](https://github.com/fdietze/dotfiles/blob/master/.ideavimrc)
> 3. [GitHub - AlexPl292/IdeaVim-EasyMotion: EasyMotion emulation plugin for IdeaVim](https://github.com/AlexPl292/IdeaVim-EasyMotion)
> 4. [let mapleader = "\<Space>" not working! : vim](https://www.reddit.com/r/vim/comments/2dpihg/let_mapleader_space_not_working/)
> 5. [Site Unreachable](https://www.andreasoverland.no/vim)

### IDEA插件推荐
1. IdeaVimExtension：在切换到normal模式时，自动切换为英文输入
2. IdeaVim-EasyMotion：用于配合`<leader> <leader> w`快速跳转
3. IdeaVim-Sneak：行内快速跳转