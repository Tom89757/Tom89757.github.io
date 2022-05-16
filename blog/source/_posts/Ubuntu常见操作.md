---
title: Ubuntu常见操作
date: 2022-05-15 00:22:55
categories:
- 开发工具
tags:
- Ubuntu
---

本文记录一下使用Ubuntu操作系统时的常见操作：

<!--more-->

1.`echo $PATH | tr ":" "\n"`：在`bash`终端中分行展示环境变量。

> 参考资料：[How to split the contents of $PATH into distinct lines](https://stackoverflow.com/questions/33469374/how-to-split-the-contents-of-path-into-distinct-lines)

</br>

2.`httping -x localhost:1080 -g http://google.com -c 3`：在Ubuntu终端中测试通过代理是否能访问`google.com`。之所以使用`httping`是因为`ping`无法通过代理访问。具体步骤如下：

- 1）通过`sudo apt install httping`安装工具`httping`。
- 2）（在代理开启的情况下）运行上述命令。`-x`表示代理服务器地址；`localhost:1080`表示代理服务器为本机，监听`1080`端口；`-g`表示对其发送请求的URL，本例中为`http://google.com`；`-c`表示在结束请求前代理服务器会向目标URL发送多少 probe，此处为3。

运行结果如下：

![image-20220515111109604](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515111109604.png)

> 参考资料：[can not ping google using proxy](https://askubuntu.com/questions/428408/can-not-ping-google-using-proxy)

</br>



