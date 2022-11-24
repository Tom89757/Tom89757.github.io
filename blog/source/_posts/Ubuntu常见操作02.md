---
title: Ubuntu常见操作02
date: 2022-11-22 20:49:01
categories:
- 开发工具
tags:
- Ubuntu
---
本文记录一下使用Ubuntu操作系统时的常见操作：
<!--more-->
1.`sed`命令中的`.`字符。`.`用于匹配除换行符之外的任意单个字符，它必须匹配一个字符。通过这种形式加上正则表达式的贪婪匹配（匹配符合模式的最长字符串）可以进行如下替换操作：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221122221549.png)
其命令为`head scores2.txt | sed 's#.*/##*'。
> 参考资料：
> 1. 《Linux命令行于shell脚本编程大全》第三版20.2.4点号字符

</br>
2.删除文件时出现`cannot remove:device or resource busy`：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221123001238.png)
解决方案：
- `lsof +D /path`：查看当前路径下哪些进程占用文件
- `kill -9 $PID`：关闭对应进程id
- `rm -rf ./*`：重新尝试删除文件
一行命令实现：
`lsof +D ./ | awk '{print $2}' | tail -n +2 | xargs -r kill -9`
> 参考资料：
> 1. [files - How to get over "device or resource busy"? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/11238/how-to-get-over-device-or-resource-busy)


