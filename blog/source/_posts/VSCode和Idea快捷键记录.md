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
11.`Alt+Left/Right`跳转到上/下一个跳转位置。
12.`gb`选中光标所在单词，再次按`gb`选中和前面单词的下一个出现。按下`c`或者`i`可以进行多光标编辑。（IDEA中不起作用）

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


### 文本选择（包括选择、删除、替换、复制、缩进）
选择、删除、替换、复制、缩进分别对应`v`, `d`, `c`, `y`, `>`。（visual/delete/change/yank/indent）
首先先根据参考资料1理解text object的概念：
以选择模式(`v`)为例：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407182521.png)
解释：
`viw`：visual select inner a word
`di"`：delete the content which surrounded by `""`
1.`viw`：选中光标所在单词（不包括单词末尾空格/space）。如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407172636.png)
2.`vaw`：选中光标所在单词（包括单词末尾空格/space）。如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407172754.png)
3.`viW/vaW`：选中更广义的单词（`,`、`;`等不会中断识别）：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407173106.png)
上述`viw`会被`,`中断。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407173146.png)
使用`viW`则不会被`,`中断。
4.`vis`：选中光标所在句子，句子通过`.`分隔。`vas`则会多选中末尾一个空格（对代码来说不太好用，但理解`s`表示句子的概念对理解Vim-surround中的操作有帮助。没有`viS/vaS`）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407184135.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407184204.png)
5.`vip`：选中光标所在的段落，段落通过空行分割。`vap`会多选中末尾空行直至下一个段落。（没有`viP/vaP`）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407184257.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407184346.png)
6.`vi{`：选中光标所在位置外部的第一个`{}`内的所有内容，不包括`{}`本身。（`vi(`, `vi[`, `vi"`, `vi'`同理）：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407184721.png)
7.`va{`：选中光标所在位置外部的第一个`{}`内的所有内容，包括`{}`本身。（`vi(`, `vi[`, `vi"`, `vi'`同理）：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407184839.png)
8.`vit`：选中光标所在的HTML tag中的内容。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407191410.png)

PS：将上述的`v`替换为`c/d/y/>`可以实现相同范围的选择。
>参考资料：
> 1. [Vim中的重要概念 Text Object_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Ze4y1E7Sk/?spm_id_from=333.999.0.0&vd_source=71b57f2bb132ac1f88ed255cad4a06a6)

### Vim-surround
add/change/delete surrounding for content
add：`ys [text object] Mark`
change：`ds Mark`
delete：`cs srcMark dstMark`
**add**
1.`ysiw"`：给光标所在单词添加`""`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407190108.png)
2.`yss"`：给光标所在行添加`""`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407190233.png)
3.`ysip"`：给光标所在代码段添加`""`。（注意此处需要`i`，上述VSCode中`yss`不需要`i`，IDEA中需要写`ysis`，但根据参考资料5该特性并没有在IDEA中实现）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407190720.png)
4.`ysiw(`：给光标所在单词添加`()`，额外添加空格。（`yss(`和`ysip(`同理）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407190909.png)
5.`ysiw)`：给光标所在单词添加`()`，不额外添加空格。（`yss)`和`ysip)`同理）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407190957.png)
6.`ysiwt p`：给光标所在单词添加HTML tag `<p></p>`。（`ysst p`和`ysipt p`同理）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407191812.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407191934.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407191949.png)

**change**
1.`cs"'`：将光标所在单词最近的`""`变为`''`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407192147.png)
2.`cs"{`：将光标所在单词最近的`""`变为`{}`，并添加额外空格。（`cs"}`不会添加额外空格）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407192308.png)
3.`cs"t p`：将光标所在单词最近的`""`变为对应HTML tag `<p></p>`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407192819.png)

**delete**
1.`ds"`：删除光标所在单词最近的`""`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407192441.png)
2.`ds{`：删除光标所在单词最近的`{}`，并删除周围空格，如果没有则不删除。（`ds}`不会删除周围空格）
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407192615.png)
3.`dst`：删除光标所在单词最近的HTML tag。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407193033.png)

> 参考资料：
> 1. [好用的Vim插件Vim-Surround介绍_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Le4y1i7Uz/?spm_id_from=333.788&vd_source=71b57f2bb132ac1f88ed255cad4a06a6)
> 2. [指尖飞舞：vscode + vim 高效开发（vim-surround）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1YA411u78P/?spm_id_from=333.337.search-card.all.click&vd_source=71b57f2bb132ac1f88ed255cad4a06a6)
> 3. [GitHub - tpope/vim-surround: surround.vim: Delete/change/add parentheses/quotes/XML-tags/much more with ease](https://github.com/tpope/vim-surround)
> 4. [https://youtrack.jetbrains.com/issue/VIM-769/inbuilt-vim-surround-support-in-ideavim](https://youtrack.jetbrains.com/issue/VIM-769/inbuilt-vim-surround-support-in-ideavim)
> 5. [https://youtrack.jetbrains.com/issue/VIM-2004/Cannot-surround-full-line-using-yss-command](https://youtrack.jetbrains.com/issue/VIM-2004/Cannot-surround-full-line-using-yss-command)


### VSCode插件NERDTree
操作如下：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407195849.png)









