---
title: everything搜索技巧
date: 2023-05-31 11:29:07
categories:
- 开发工具
tags:
- everything 
---
本文记录一下使用everything的搜索技巧。
<!--more-->
### everything配置search history (搜索历史)
可搭配PowerToys映射快捷键`Ctrl+j/k`到`downarrow/uparrow`。
> 参考资料：
> 1. [Search History - voidtools](https://www.voidtools.com/support/everything/search_history/)
> 2. [How to Remap Any Key or Shortcut on Windows 10](https://www.howtogeek.com/710290/how-to-remap-any-key-or-shortcut-on-windows-10/)


### everything搜索技巧
1. 可以自定义筛选器（例如指定图片后缀）
2. `*.png 1234`：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230531121936.png)
3. `123 | 456`：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230531122052.png)
4. 排除文件夹：在索引里面设置
5. `file:123`：只搜索包含`123`的文件
6. `folder:123`：只搜索包含`123`的文件夹
7. `D:\Desktop 123`：指定搜索路径
8. `123??`：通配符，`?`代表任意一个字符
9. `123*`：`*`代表任意多个字符
10. `vimrc$`：正则表达式，以`vimrc`结尾的文件
11. 文件服务器：在浏览器中搜索、查看和下载文件
> 参考资料：
> 1. [高效搜索神器 Everything 搜索技巧汇总 - 知乎](https://zhuanlan.zhihu.com/p/165142586)