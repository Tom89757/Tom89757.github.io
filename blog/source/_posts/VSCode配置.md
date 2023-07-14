---
title: VSCode配置
date: 2022-05-21 17:09:43
categories:
- 环境配置
tags:
- VSCode
---

本文记录一下在 VSCode 编辑器中一些常用的配置方法。

<!--more-->

### VSCode 重新启用“错误波形提示”

有两种方式：

- 手动：
  1. 文件——首选项——设置（files->preferences->settings），搜索 `error Squiggles`。
  2. 在用户 (user) 和工作区 (workspace) 都选择`enable`。

- 快捷键：在命令面板通过快捷键 (Ctrl + Shift + P) 打开搜索栏，搜索 `Error Squiggles`，选择`enable`。

  PS：只对某些 extensions 有用，如 C/C++ 插件

> 参考资料：[VSCode重新启用“错误波形提示”](https://blog.csdn.net/HermitSun/article/details/103627053)

### VSCode 配置快捷键（shortcuts）

有两种方式：

- 手动：`File -> Preferences -> Keyboard Shortcuts`，即可打开窗口手动编辑。

- `json`文件：在上述打开的窗口中点击右上角的如图按钮编辑`json`文件：

  ![image-20220523101335662](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220523101335662.png)

### VSCode 在多个窗口中打开同一项目

- `File -> Add Folder to Workspace`，添加目标项目到 workspace：

  ![image-20220523101543021](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220523101543021.png)

- `File -> Dulplicate Workspace`，重复该工作区：

  ![image-20220523101705028](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220523101705028.png)

### VSCode显示85字符宽度提示线

- `File->Preferences->Settings`，搜索`editor.rulers`：

  ![image-20220625185036578](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220625185036578.png)

- 打开`settings.json`，编辑`editor.rulers`的值为85：

  ![image-20220625185247521](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220625185247521.png)

> 参考资料：
>
> 1. [vs code设置每行代码的垂直标尺](https://blog.csdn.net/qq_43406338/article/details/109397831)

### VSCode设置默认Python Interpreter

依次打开`File->Preferences->Settings`，搜索`python.default`，配置如下：

![image-20220808165221633](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220808165221633.png)

> 参考资料：
>
> 1. [Using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments)

### VSCode关闭Editor: Enable Preview

VSCode默认打开Workbench的Preview，其表现为，在单击一个左边的一个文件后，如果在不对其进行modify操作后单击另一个文件，会直接切换为另一个文件而不是在新tab页打开另一个文件，这不符合本人的习惯。

此时，可以通过打开`settings`搜索`preview`关键词然后关闭`Workbench>Editor: Enable preview`来实现：

![image-20220729153253563](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220729153253563.png)

### VSCode出现Unexpected Indentation in Visual Studio Code with Python

> 参考资料：
>
> 1. [Unexpected Indentation in Visual Studio Code with Python](https://stackoverflow.com/questions/52224313/unexpected-indentation-in-visual-studio-code-with-python)

### VSCode配置默认Terminal

- 搜索`Terminal: Select Default Profile`

![image-20220810185654267](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220810185654267.png)

- 点击目标Terminal如cmd：

  ![image-20220810185807233](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220810185807233.png)

> 参考资料：
>
> 1. [VSCode Change Default Terminal](https://stackoverflow.com/questions/44435697/vscode-change-default-terminal)

### VSCode在运行或者编译C++文件出现问题
launch: workingDirectory 'D:\\Develop\\msys2\\mingw64\\bin' does not exist
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220917002755.png)
此时，为`.vscode`文件夹下的json配置文件出错：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220917002916.png)
需要将上图中的`msys2`改为正确的安装目录`msys64`。
### 彻底卸载VSCode

> 参考资料：
> 1. [彻底卸载VSCode](https://bbs.huaweicloud.com/blogs/254150)

### VSCode配置Git终端
- 打开Settings
- 搜索`shell:windows`，打开`settings.json`：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016110802.png)
- 添加如下GitBash终端配置：
```json
"GitBash": {
            "path": "D:\\Develop\\Git\\bin\\bash.exe",
            "args": [],
            "icon": "terminal-bash"
        },
```
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016110936.png)
- 重启VSCode，可以发现配置生效
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016111046.png)
PS：上述的GitBash不要添加空格使用"Git Bash"，会导致配置无法生效
> 参考资料：
> 1. [GitBash not showing up as a terminal option in Visual Studio Code](https://stackoverflow.com/questions/68068359/gitbash-not-showing-up-as-a-terminal-option-in-visual-studio-code)

### VSCode leetcode插件
出现`command 'leetcode.sign in' not found`。
解决方案：
- 在everything中搜索`leetcode`文件夹，删除如下安装的leetcode插件文件夹
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221031222539.png)
- 搜索`.lc`文件夹，同样删除
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221031222640.png)
- 重新安装leetcode插件。
当进行代码Test时出现如下问题：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221031225401.png)
解决方案：登录leetcode官网后在设置里进行邮箱验证
> 参考资料：
> 1. [Can't load and login: command 'leetcode.signin' not found · Issue #622 · LeetCode-OpenSource/vscode-leetcode · GitHub](https://github.com/LeetCode-OpenSource/vscode-leetcode/issues/622)
> 2. [vscode LeetCode显示sign in 成功，但是[ERROR] session expired, please login again [code=-1]的问题！！！_LIQIANDI的博客-CSDN博客](https://blog.csdn.net/qq_41521512/article/details/115199293)
> 3. [登录不上，[ERROR] session expired, please login again [-1] #94](https://github.com/skygragon/leetcode-cli/issues/94)


### VSCode Vim插件配置
```json
//插入模式设置，将双击"j"键映射为"<Esc>"键
"vim.insertModeKeyBindings": [
        {
            "before": ["j", "j"],
            "after": ["<Esc>"]
        }
    ],
//通过键<C-h>等设置为false保留VSCode原生快捷键
"vim.handleKeys": {
	"<C-h>": false,
	"<C-a>": false,
	"<C-f>": false,
	"<C-n>": false,
	"<C-p>": false,
	"<C-x>": false
},
//使vim不会捕获Ctrl键，从而可以使用所有Ctrl+Key VSCode快捷键，此时可不设置上面
"vim.useCtrlKeys": false,
//"vim.leader"可以看作类似Ctrl的前缀键，可以对Vim做很多的个性化设置
"vim.leader": "<space>",
"vim.commandLineModeKeyBindings": [
],
//正常模式设置，此处将"H"映射为"^"（行首），将"L"映射为"$"（行尾）
"vim.normalModeKeyBindings": [
	{
		"before": ["H"],
		"after": ["^"]
	},
	{
		"before": ["L"],
		"after": ["$"]
	}
]
```
> 参考资料：
> 1. [在VSCode里面配置Vim的正确姿势（细节解析） - 知乎](https://zhuanlan.zhihu.com/p/188499395)
> 2. [vscode + vim : vscode 全键盘使用方案_vim_lmymirror_InfoQ写作社区](https://xie.infoq.cn/article/654e137365b09e217f57bc965)
> 3. [指尖飞舞：vscode + vim 高效开发（easymotion）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Ry4y1H7zH)
> 4. [写给 VS Code 用户的 Vim 入坑指南](https://www.yuexun.me/blog/the-vim-guide-for-vs-code-users/)
> 5. [vim 使用技巧](https://www.pengfeixc.com/blogs/developer-handbook/vim-shortcuts)
> 6. [vs code 使用 vim 插件 快捷键问题 - V2EX](https://www.v2ex.com/t/703444)
> 7. [vscode + vim 全键盘操作高效搭配方案 - 知乎](https://zhuanlan.zhihu.com/p/430603620)
> 8. [vim 文本插入 - 在光标的前面，后面，行尾，行首插入字符 - vim使用入门 | 宅学部落](https://www.zhaixue.cc/vim/vim-insert.html#:~:text=%E5%B8%B8%E7%94%A8%E7%9A%84vim%E6%8F%92%E5%85%A5%E5%91%BD%E4%BB%A4%EF%BC%9A&text=a%EF%BC%9A%E5%9C%A8%E5%BD%93%E5%89%8D%E5%85%89%E6%A0%87%E7%9A%84,%E7%9A%84%E7%BB%93%E5%B0%BE%E5%A4%84%E6%B7%BB%E5%8A%A0%E6%96%87%E6%9C%AC)
> 9. [How do you avoid key binding collisions between VS Code and vscodevim? - Stack Overflow](https://stackoverflow.com/questions/62405783/how-do-you-avoid-key-binding-collisions-between-vs-code-and-vscodevim)
> 10. [Use in Visual Studio Code (Vim extension) CAPS instead of ESC - Stack Overflow](https://stackoverflow.com/questions/48369303/use-in-visual-studio-code-vim-extension-caps-instead-of-esc)

### VSCode Vim打开EasyMotion和Sneak实现文件内任意跳转
#### Sneak
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230325225209.png)
- `s{char}{char}`跳转到从光标开始，第一个`{char}{char}`出现的位置
- `;`跳转到下一个出现位置，`,`跳转到上一个
- `S{char}{char}`：反向查找，即方向相反
#### EasyMotion
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230325225459.png)
EasyMotion使你摆脱需要`5j`或者`5k`这样数数的麻烦。
- `<leader><leader>w`：会使用字母的排列组合标注当前行以及之后的行的单词。
标注前：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230325225622.png)
标注后：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230325225650.png)
按下对应键即可实现跳转。
- `<leader><leader>f'`，EasyMotion将会标注所有`'`字符在当前行和之后行出现的位置。
> 参考资料：
> 1. [Moving Even Faster with Vim Surround and EasyMotion | Barbarian Meets Coding](https://www.barbarianmeetscoding.com/boost-your-coding-fu-with-vscode-and-vim/moving-even-faster-with-vim-sneak-and-easymotion/)

### VSCode配置在终端和编辑器之间切换的快捷键
> 参考资料：
> 1. [VScode在终端和编辑器之间切换的快捷键_Xu小亿的博客-CSDN博客](https://blog.csdn.net/Jeffxu_lib/article/details/86651173)


### VSCode打开new tab
在VSCode中单击左侧的侧边栏中的file brower中的文件或者通过`Ctrl-p`搜索跳转或者通过`Ctrl+左键`在代码中跳转到文件时，会在新的tab打开对应文件，但如果不对该文件进行编辑，文件处于 preview mode (文件名为意大利斜体)：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230114000110.png)
此时如果通过上述三种方式打开或跳转到新的文件，之前处于preview mode的文件会被替换掉：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230114000131.png)
解决方法：
1. 双击左侧侧边栏的文件名或者上方tab的文件名，此时文件会进入edit mode：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230114000313.png)
2. 通过设置配置文件`settings.json` (通过`Ctrl-Shift-P`搜索`User Settings`打开) 在里面关闭preview mode：
```json
"workbench.editor.enablePreview": false
"workbench.editor.enablePreviewFromQuickOpen": false
```
> 参考资料：
> 1. [visual studio code - Open files always in a new tab - Stack Overflow](https://stackoverflow.com/questions/38713405/open-files-always-in-a-new-tab)


### VSCode配置defaultFormatter

> 参考资料：
> 1. [Installing the Ruby Plugin for Prettier in VS Code - DEV Community](https://dev.to/drayeleo/installing-the-ruby-plugin-for-prettier-in-vs-code-2m6c#:~:text=You%20should%20now%20be%20able,for%20macOS)
> 2. [html - How do I auto format Ruby or .erb files in VS Code? - Stack Overflow](https://stackoverflow.com/questions/41051423/how-do-i-auto-format-ruby-or-erb-files-in-vs-code)


### VSCode配置Fira Code字体
```json
"editor.fontLigatures": true,
"editor.fontFamily": "Fira Code",
```
>参考资料：
>1. [Fira Code: 一个有趣而实用的编程字体 - 知乎](https://zhuanlan.zhihu.com/p/38605932)
>2. [visual studio/vscode 使用Fira code字体 - KizunaT - 博客园](https://www.cnblogs.com/kizuna1314/p/15423673.html)
>3. [FiraCode/README_CN.md at master · tonsky/FiraCode · GitHub](https://github.com/tonsky/FiraCode/blob/master/README_CN.md)

### VSCode管理Java项目

> 参考资料：
> 1. [Java project management in Visual Studio Code](https://code.visualstudio.com/docs/java/java-project)


### VSCode插件推荐
1. [Set up CodeGPT in Visual Studio Code](https://blog.openreplay.com/set-up-codegpt-in-visual-studio-code/)
> 参考资料：
> 1. [有什么推荐的vs code插件？ - 知乎](https://www.zhihu.com/question/380933740/answer/1554048933)


### VSCode C++环境配置
> 参考资料：
> 1. [C++ programming with Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp)
> 2. [MSYS2](https://www.msys2.org/)
> 3. [Package: mingw-w64-x86_64-gcc - MSYS2 Packages](https://packages.msys2.org/package/mingw-w64-x86_64-gcc)

### VSCode ssh免密码连接远程服务器

> 参考资料：
> 1. [Developing on Remote Machines using SSH and Visual Studio Code](https://code.visualstudio.com/docs/remote/ssh)
> 2. [Visual Studio Code Remote SSH Tips and Tricks](https://code.visualstudio.com/blogs/2019/10/03/remote-ssh-tips-and-tricks)


### VSCode跳转配置
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230312161241.png)
通过勾选上述选项，可以实现在当前文件窗口更改代码，而无需跳转。
> 参考资料：
> 1. [VSCode #46 - A Better Code Folding Extension](https://mailchi.mp/vscode/46)

### VSCode刷题主题
1.Gruvbox Them较为简洁，且变量和关键字区分较为明显，比较适合刷题
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327235910.png)

### VSCode搜索文件
快捷键：Ctrl+P
> 参考资料：
> 1. [keyboard shortcuts - How do I search for files in Visual Studio Code? - Stack Overflow](https://stackoverflow.com/questions/30095376/how-do-i-search-for-files-in-visual-studio-code)


### VSCode Copilot使用
1.`tab`键补全。
2.`Ctrl + Enter`右侧打开Copilot窗口。
> 参考资料：
> 1. [Github Copilot 的使用方法和快捷键 | 教程 | Tinkink](https://tutorials.tinkink.net/zh-hans/vscode/copilot-usage-and-shortcut.html#%E4%BD%BF%E7%94%A8)


### VSCode快捷键
1.`Ctrl+P`：按文件名搜索文件
2.`Ctrl+W`：关闭当前文件窗口
3.`Ctrl+K+W`：关闭所有文件窗口
4.`Ctrl+shif+l`：选中当前光标所在单词的所有单词
5.`Ctrl+,`：打开settings。
>参考资料：
>1. [VS Code 的常用快捷键 - 知乎](https://zhuanlan.zhihu.com/p/44044896)

### VSCode配置插入模式下的Vim映射
在`settings.json`中添加：（经使用，不太好用，无法输入H/L/J/K，故取消）
```json
"vim.insertModeKeyBindings": [
        {
            "before": ["H"],
            "after": ["<Esc>", "^"]
        },
        {
            "before": ["L"],
            "after": ["<Esc>", "$"]
        },
        {
            "before": ["J"],
            "after": ["<Esc>", "g", "j"]
        },
        {
            "before": ["K"],
            "after": ["<Esc>", "g", "k"]
        },
    ]
```
> 参考资料：
> 1. [Moving to the beginning of line within Vim insert mode - Super User](https://superuser.com/questions/706674/moving-to-the-beginning-of-line-within-vim-insert-mode)

### VSCode查找光标处单词的所有出现插件
Find Word At Cursor。其快捷键配置如下：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230330105601.png)
使用步骤为：
1. 使用快捷键`Ctrl+D`选中光标所在单词的所有出现。
2. 使用快捷键`Ctrl+->`定位下一个出现，使用快捷键`Ctrl+<-`定位上一个出现k。
PS：需要覆盖或删除与`Ctrl+->`和`Ctrl+<-`冲突的系统默认快捷键。

### VSCode使用`.vimrc`文件配置Vim
在`settings`中打开`.vimrc`文件配置，并在指定位置添加`.vimrc`文件：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230402235543.png)
`.vimrc`内容如下（参考资料2）：
```json
" Pick a leader key
" Use space as leader key
let mapleader = " "

nmap H ^
nmap L $
nmap j gj
nmap k gk

set textwidth=79
```
> 参考资料：
> 1. [vim - How to modify/change the vimrc file in VsCode? - Stack Overflow](https://stackoverflow.com/questions/63017771/how-to-modify-change-the-vimrc-file-in-vscode)
> 2. [A basic .vimrc file that will serve as a good template on which to build. · GitHub](https://gist.github.com/simonista/8703722)

### VSCode配置80字符和120字符垂直线
在`settings.json`中添加：
```json
"editor.rulers": [
    80, 120
  ]
```
效果如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230402235455.png)

> 参考资料：
> 1. [configuration - Vertical rulers in Visual Studio Code - Stack Overflow](https://stackoverflow.com/questions/29968499/vertical-rulers-in-visual-studio-code)


### VSCode中Vim配置复杂命令映射
PS：VSCode中跳出文件编辑窗口后`<leader>`键和其它键不起作用。
> 参考资料：
> 1. [Elevating Your Workflow With Custom Mappings | Barbarian Meets Coding](https://www.barbarianmeetscoding.com/boost-your-coding-fu-with-vscode-and-vim/elevating-your-worflow-with-custom-mappings/)
> 2. [Visual Studio Code User Interface](https://code.visualstudio.com/docs/getstarted/userinterface)
> 3. [Built-in Commands | Visual Studio Code Extension API](https://code.visualstudio.com/api/references/commands)
> 4. [How do I assign a mapping to VSCode commands · Issue #2542 · VSCodeVim/Vim · GitHub](https://github.com/VSCodeVim/Vim/issues/2542)

### VSCode中NERDTree插件实现文件切换
快捷键如下：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230404222129.png)

### VSCode中Find Word At Cursor插件实现定位所有光标处单词并快速移动到下一个
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230404003727.png)
1.`Ctrl+D`定位所有光标位置单词。
2.`Ctrl+->`和`Ctrl+<-`跳转到上一个/下一个。

### VSCode中Code Ace Jumper插件实现任意跳转
按下`Ctrl + ;`后输入对应字母即可跳转到以该字母为首字母的单词出现位置。

### VSCode中设置`justMyCode=false`失效
当前调试配置文件`launch.json`为：
```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```
根据参考资料3在上述`launch.json`中添加：
```json
"purpose": ["debug-in-terminal"]
```
即更新为：
```json
"configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "purpose": ["debug-in-terminal"]
        }
    ]
```
> 参考资料：
> 1. ["justMyCode" does not enable standard library debugging · Issue #7347 · microsoft/vscode-python · GitHub](https://github.com/microsoft/vscode-python/issues/7347)
> 2. [Testing Python in Visual Studio Code](https://code.visualstudio.com/docs/python/testing#_debug-tests)
> 3. [VsCode justMyCode: false无效 - 知乎](https://zhuanlan.zhihu.com/p/440413830)

### VSCode Insiders vs VSCode
二者的配置是独立的，但是VSCode Insiders可以同步VSCode的配置。通过关闭对VSCode的同步，然后打开VSCode Insiders的同步便可以从VSCode复制一份配置。
> 参考资料：
> 1. [Visual Studio Code Frequently Asked Questions](https://code.visualstudio.com/docs/supporting/FAQ)
> 2. [Download Visual Studio Code Insiders](https://code.visualstudio.com/insiders/)


### VSCode Python错误提示和类型检查
问题：插件Pylance会提供对Python文件的类型检查，但是有时候其类型检查较为严格，会爆出很多我们想忽视的错误
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230408141703.png)
解决方案：在settings里关闭
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230408141832.png)

### VSCode设置选取候选词切换键为tab
在快捷键中查找selectNextSuggestion设置
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230409110425.png)
> 参考资料：
> 1. [How to select the non-first item of the candidate selector by using the shortcut key? · Issue #33645 · microsoft/vscode · GitHub](https://github.com/microsoft/vscode/issues/33645)

### VSCode在settings sync (设置同步) 中排除某些设置
```json
{
    "settingsSync.ignoredSettings": [
	    "editor.fontSize",
		"-python.defaultInterpreterPath",
		"remote.SSH.configFile",
    ]
}
```
快捷键：
```json
{
   "settingsSync.keybindingsPerPlatform": True
}
```

> 参考资料：
> 1. [Can I exclude certain settings with VS Code built-in settings sync feature? - Stack Overflow](https://stackoverflow.com/questions/64603768/can-i-exclude-certain-settings-with-vs-code-built-in-settings-sync-feature)

### VSCode显示tab和space并设置颜色

> 参考资料：
> 1. [Show whitespace characters in Visual Studio Code - Stack Overflow](https://stackoverflow.com/questions/30140595/show-whitespace-characters-in-visual-studio-code)
> 2. [Setting to change the color and opacity of whitespace characters when made visible… · Issue #25956 · microsoft/vscode · GitHub](https://github.com/Microsoft/vscode/issues/25956)

### VSCode中使用显示/隐藏terminal窗口
使用
```
Ctrl + `
```

> 参考资料：
> 1. [How to switch between terminals in Visual Studio Code? - Stack Overflow](https://stackoverflow.com/questions/48440673/how-to-switch-between-terminals-in-visual-studio-code) 


### VSCode中切换terminal tab
- `Ctrl + pageDown`：

### VSCode出现  "Error while fetching extensions. XHR failed"

> 参考资料：
> 1. [marketplace - Visual Studio Code "Error while fetching extensions. XHR failed" - Stack Overflow](https://stackoverflow.com/questions/70177216/visual-studio-code-error-while-fetching-extensions-xhr-failed)


### VSCode配置终端字体
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230615115430.png)

> 参考资料：
> 1. [Nerd Fonts - Iconic font aggregator, glyphs/icons collection, & fonts patcher](https://www.nerdfonts.com/font-downloads)

### VSCode配置终端图标
```json
"Git Bash": {
  "source": "Git Bash",
  "icon": "github"
},
"Ubuntu (WSL)": {
  "path": "C:\\WINDOWS\\System32\\wsl.exe",
  "args": ["-d", "Ubuntu"],
  "icon": "terminal-ubuntu"
}
```
>参考资料：
>1. [Terminal Appearance in Visual Studio Code](https://code.visualstudio.com/docs/terminal/appearance)
>2. [Terminal Profiles in Visual Studio Code](https://code.visualstudio.com/docs/terminal/profiles)


### VSCode Neovim
- 映射`init.vim`文件：
```cmd
mklink C:\Users\26899\AppData\Local\nvim\init.vim D:\Desktop\dotfiles\nvim-win\init.vim
```
> 参考资料：
> 1. [vscode-nvim - 云崖君 - 博客园](https://www.cnblogs.com/YunyaSir/p/15523927.html)
> 5. [Settings sync is not available · Issue #482 · VSCodium/vscodium · GitHub](https://github.com/VSCodium/vscodium/issues/482)
> 6. [Neovim IDE Crash Course](https://www.chrisatmachine.com/posts/01-ide-crash-course)

### VSCode设置隐藏文件

> 参考资料：
> 1. [如何在VSCode设置/取消隐藏文件_zwkkkk1的博客-CSDN博客](https://blog.csdn.net/zwkkkk1/article/details/93742821)