---
title: 常见bug记录
date: 2022-05-13 01:37:45
tags:
---



1.`github`中网址前缀为`raw.githubusercontent.com`的资源（图片、文档等）无法访问。

解决方案：

根据 [解决 raw.githubusercontent.com 无法访问的问题](https://learnku.com/articles/43426)，可能是由于某些原因导致 DNS 被污染，Windows系统上可以通过修改 `hosts` 文件解决该问题，步骤如下：

1. 通过 [IPAddress.com](https://www.ipaddress.com/) 查询域名 `raw.githubusercontent.com`所在网址：

![image-20220513014334969](C:\Users\A\AppData\Roaming\Typora\typora-user-images\image-20220513014334969.png)

2. 在路径 `C:\Windows\System32\drivers\etc`下的`hosts`文件最后一行添加如下信息：

   `185.199.108.133 raw.githubusercontent.com`

   PS：可能`hosts`文件为只读文件，此时需要右键单击`hosts`文件修改其访问权限。
