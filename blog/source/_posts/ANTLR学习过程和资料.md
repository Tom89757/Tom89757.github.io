---
title: ANTLR学习过程和资料
date: 2022-10-21 10:50:22
categories:
- 资料
tags:
- ANTLR
---

本文记录以下学习ANTLR过程中所使用的资料和学习经验：
<!--more-->

### 学习资料
1. [ANTLR4 笔记](https://abcdabcd987.com/notes-on-antlr4/)：记录了ANTLR4的使用技巧
2. [如何学习ANTLR? - 知乎](https://www.zhihu.com/question/437337408/answer/1661188049)：记录了学习ANTLR的经验
3. [grammars-v4](https://github.com/antlr/grammars-v4)：ANTLR4官方仓库，收集了大量ANTLR v4的形式化语法
4. [antlr4/grammars.md](https://github.com/antlr/antlr4/blob/master/doc/grammars.md)：ANTLR4官方开源仓库的语法文档，描述了语法结构
5. [ANTLR 4权威指南 (豆瓣)](https://book.douban.com/subject/27082372/)

### 配置教程
1. [ANTLR4在windows10下的安装 - solvit - 博客园](https://www.cnblogs.com/solvit/p/10097234.html)

### 快速上手
1. [语法解析器ANTLR4从入门到实践 - 掘金](https://juejin.cn/post/7018521754125467661)
2. [用ANTLR解析领域特定语言 | 陈颂光](https://www.chungkwong.cc/antlr.html)
3. [Antlr4简易快速入门 - 知乎](https://zhuanlan.zhihu.com/p/114982293)


### 所遇问题
1. 出现`Cannot resolve symbol 'antlr'`。如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221021115827.png)
原因：虽然已经在本地下载了antlr包并将其添加进了CLASSPATH，但IDEA并不能识别出。
解决方案：在Project Settings对应Module中添加依赖
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221021120414.png)
