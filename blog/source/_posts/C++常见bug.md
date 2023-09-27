---
title: C++常见bug
date: 2023-09-28 00:20:42
categories:
- 资料
tags:
- C++
---
本文记录一下C++编程过程中常见的bug：
<!--more-->

### 报错`can't open source file "sys/socket.h"`等
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230928002225.png)
问题：这些头文件为Linux系统中对应头文件
解决方案：导入Windows系统中对应头文件
```c++
#include <Winsock2.h>
#include <sys/types.h>
```
> 参考资料：
> 1. [c++ - How can I use sys/socket.h on Windows? - Stack Overflow](https://stackoverflow.com/questions/67726142/how-can-i-use-sys-socket-h-on-windows)
> 2. [c - Cannot open include file: 'arpa/inet.h': - Stack Overflow](https://stackoverflow.com/questions/23730455/cannot-open-include-file-arpa-inet-h)

### 报错`SOCKLEN_T' : UNDECLARED IDENTIFIER`
问题：windows中`socklen_t`定义在`ws2tcpip.h`中
解决方案：导入该头文件
```c++
#include <ws2tcpip.h>
```
> 参考资料：
> 1. [[Solved]-error C2065: 'socklen_t' : undeclared identifier-C++](https://www.appsloveworld.com/cplus/100/508/error-c2065-socklen-t-undeclared-identifier)

### 报错`argument of type int is incompatible with parameter of type const char *`
问题：该函数接收参数需要为`const char*`
解决方案：类型转换
```c++
setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&val, sizeof(val));
```