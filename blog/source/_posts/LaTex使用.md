---
title: LaTex使用
date: 2023-04-04 19:45:24
categories:
- 工具
tags:
- Latex 
---
本文记录一下使用LaTex过程中的经验和教训：
<!--more-->
1.编译时出现`Fatal Package fontspec Error: The fontspec package requires either XeTeX or lualatex`报错。
原因：使用的编译器不匹配
解决方案：在Overleaf中的menu中将编译器设置为XeTex或LuaLatex。
> 参考资料：
> 1. [Fontspec {cannot-use-pdftex} on overleaf - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/400825/fontspec-cannot-use-pdftex-on-overleaf)


