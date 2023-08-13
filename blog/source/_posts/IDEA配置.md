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
" basic settings
set ignorecase
set scrolloff=30
set history=200

set number
set relativenumber

set incsearch
set hlsearch
set keep-english-in-normal

" Plug
Plug 'preservim/nerdtree' 
set easymotion
set sneak
" nmap j j to Esc"
" imap jj <Esc>

" I like using H and L for beginning/end of line
" Have j and k navigate visual lines rather than logical ones
nmap H ^
nmap L $
nmap j gj
nmap k gk

" Yank to system clipboard"
set clipboard+=unnamed

" No Leader Keymaps
nmap gd <Action>(GotoDeclaration)
nmap ge <Action>(GotoNextError)
nmap gE <Action>(GotoPreviousError)
nmap gm <Action>(MethodDown)
nmap gM <Action>(MethodUp)

" Use <leader><Right> and <leader><Left> to locate Next/Previous Occurence
" nmap <leader><Right> <Action>(PreviousOccurrenceoftheWordatCaret)
" nmap <leader><Left> <Action>(NextOccurrenceoftheWordatCaret)
nmap <C-Right> <C-F3>
nmap <C-Left> <C-S-F3>

# set leader to space 
let mapleader=" " 

" Use <leader>l and <leader>h to switch tabs
nmap <leader>l gt
nmap <leader>h gT

" Use <leader>b to locate bracket
nmap <leader>b %

" Use <leader>q to close the current tab
nmap <leader>q :tabclose<CR>


" Use <leader>n to focus explorer 
nmap <leader>n :NERDTreeToggle<CR> 
```
PS：其中`<Action>`只能用于IDEA自带的actions。Plugins actions需要使用`:`。
> 参考资料：
> 1. [vim - Intellij IdeaVim change keys - Stack Overflow](https://stackoverflow.com/questions/10149187/intellij-ideavim-change-keys)
> 2. [dotfiles/.ideavimrc at master · fdietze/dotfiles · GitHub](https://github.com/fdietze/dotfiles/blob/master/.ideavimrc)
> 3. [GitHub - AlexPl292/IdeaVim-EasyMotion: EasyMotion emulation plugin for IdeaVim](https://github.com/AlexPl292/IdeaVim-EasyMotion)
> 4. [let mapleader = "\<Space>" not working! : vim](https://www.reddit.com/r/vim/comments/2dpihg/let_mapleader_space_not_working/)
> 5. [Vim & IdeaVim shortcuts, keystroke combos and commands](https://www.andreasoverland.no/vim)
> 6. [Vim keyboard shortcuts for project navigator / structure / tool window](https://youtrack.jetbrains.com/issue/VIM-1042/Vim-keyboard-shortcuts-for-project-navigator-structure-tool-window)

### IDEA插件推荐
1. IdeaVimExtension：在切换到normal模式时，自动切换为英文输入（切换为美式键盘，不好用）
2. IdeaVim-EasyMotion：用于配合`<leader> <leader> w`快速跳转
3. IdeaVim-Sneak：行内快速跳转
4. NERDTree：定位explorer并进行文件选择/重命名/新建等操作。[NERDTree support · JetBrains/ideavim Wiki · GitHub](https://github.com/JetBrains/ideavim/wiki/NERDTree-support)
5. Material Theme UI：IDEA界面主题美化
其中NERDTree快捷键如下：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407195931.png)

> 参考资料：[NERDTree support · JetBrains/ideavim Wiki · GitHub](https://github.com/JetBrains/ideavim/wiki/NERDTree-support)


### IDEA文件颜色配置

> 参考资料：
> 1. [File status highlights | IntelliJ IDEA Documentation](https://www.jetbrains.com/help/idea/file-status-highlights.html)

### Maven依赖项
 报错`Dependency 'org.springframework.boot:spring-boot-starter-web:' not found`。
解决方案：`Settings->Maven->Maven home directory->D:/Develop/Java/Maven/apache-maven-3.8.3`，详细见参考资料
> 参考资料：
> 1. [解决报错project 'org.springframework.boot:spring-boot-starter-parent:1.5.9.RELEASE' not found问题_叫我天真的博客-CSDN博客](https://blog.csdn.net/LJH_laura_li/article/details/104850229)

### 创建文件时自动生成作者信息
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230717111930.png)

> 参考资料：
> 1. [javadoc - Autocompletion of @author in Intellij - Stack Overflow](https://stackoverflow.com/questions/18736986/autocompletion-of-author-in-intellij)


### IDEA focus in sidebar

> 参考资料：
> 1. [Shortcut to toggle focus between project sidebar and editor? – IDEs Support (IntelliJ Platform) | JetBrains](https://intellij-support.jetbrains.com/hc/en-us/community/posts/206887115-Shortcut-to-toggle-focus-between-project-sidebar-and-editor-)


### IDEA将`dotfiles/idea/.ideavimrc`映射到`C:\Users\A\.ideavimrc`
打开`cmd`运行如下指令：
```cmd
mklink C:\Users\A\.ideavimrc D:\Desktop\dotfiles\idea\.ideavimrc
```

### IDEA将action映射到vim shortcuts

> 参考资料：
> 1. [IntelliJ IDEA Action List - Google Sheets](https://docs.google.com/spreadsheets/d/17GvVbsLc48iM-vpKgBTwz5ByvsMmmw0dqIenzemDcXM/edit#gid=0)
> 2. [IdeaVim actionlist · GitHub](https://gist.github.com/zchee/9c78f91cc5ad771c1f5d)


