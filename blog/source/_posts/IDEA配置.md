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
