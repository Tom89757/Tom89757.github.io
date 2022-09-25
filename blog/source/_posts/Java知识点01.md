---
title: Java知识点01
date: 2022-08-15 11:17:01
categories:
- 资料
tags:
- Java
---

本文记录一下与Java相关的一些知识点：
<!--more-->

### 属性和方法签名
>参考资料：
>1. [关于属性与方法的签名](https://morrisware01.gitbooks.io/android-learning-note/content/ndkkai-fa-zhi-lv/jniji-chu/shu-xing-yu-fang-fa-qian-ming.html?q=)
### IDEA将 Java 项目打包成 Jar 包
> 参考资料：
> 1. [IDEA 将普通 Java 项目打包成 Jar 包并运行](https://juejin.cn/post/7031717860003020814)
### exe4j将jar包转换为可执行文件
> 参考资料：
> 1. [exe4j安装及注册](https://www.cnblogs.com/jepson6669/p/9211208.html)
> 1. [idea打包java项目成exe可执行文件](https://blog.csdn.net/weixin_45149355/article/details/106839486)
### 从pom.xml导入项目
>参考资料：
>1. [Lab 2 Setup: Library Setup](https://sp21.datastructur.es/materials/lab/lab2setup/lab2setup)
### JUnit debugging
> 参考资料：
> 1. [Lab3: Unit Testing with JUnit, Debugging](https://sp19.datastructur.es/materials/lab/lab3/lab3)
### IntelliJ IDEA使用Java visualizer可视化
>参考资料：
>1. [Java Visualizer Tutorial](https://examples.javacodegeeks.com/java-visualizer-tutorial/)
### java 错误，不支持发行版本5
> 参考资料：
> 1. [Error java 错误 不支持发行版本5 ( 完美解决版）](https://blog.csdn.net/qq_51263533/article/details/120209830)
### Java内存使用评估
> 参考资料：
> 1. [How to calculate the memory usage of Java objects](https://www.javamex.com/tutorials/memory/object_memory_usage.shtml)
> 2. [Memory Usage Estimation in Java](http://blog.kiyanpro.com/2016/10/07/system_design/memory-usage-estimation-in-java/ "Memory Usage Estimation in Java")
### 多个模块之间的依赖
有时一个项目可能包含多个模块，在想要在一个模块中引用另一个模块的类或方法时，会发现IDEA并没有弹出提示，其原因为另一个模块中的类或方法没有声明为`public`。
### Error running 'Remote Debugger'
使用Remote JVM Debug时，运行debug出现：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220924174528.png)


> 参考资料：
> 1. [Error running 'Remote Debugger': Unable to open debugger port (localhost:5005): java.net.ConnectException "Connection refused (Connection refused)"](https://stackoverflow.com/questions/53327701/error-running-remote-debugger-unable-to-open-debugger-port-localhost5005)

### javac编译详解
> 参考资料：
> 1. [第1期：抛开IDE，了解一下javac如何编译](https://imshuai.com/using-javac#)
### Java外部包配置
有两种方式：
- 在IDEA Project Structure里配置
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220925001124.png)
- 在环境变量里配置`CLASSPATH`，可能需要重启后生效（无效，似乎IDEA不会自动导入`CLASSPATH`变量）。
> 参考资料：
> 1. [Algorithms, 4th Edition（算法-第四版）源码使用系统配置](https://zhuanlan.zhihu.com/p/25551032)
> 2. [Java Algorithms and Clients](https://algs4.cs.princeton.edu/code/)
### 出现ClassNotFoundException
原因：未将所依赖的jar包配置到`CLASSPATH`环境变量
解决方案：
- 临时生效：`java -cp .;D:\\Develop\\Java\\jdk11.0.11\\lib\\algs4.jar BinarySearch largeW.txt < largeT.txt`
- 永久生效：在`CLASSPATH`中添加`.;D:\\Develop\\Java\\jdk11.0.11\\lib\\algs4.jar`。`.`表示将当前目录加入class path的检索路径。
> 参考资料：
> 1. [Java.lang.classnotfoundexception - HelloWorld.class [duplicate]](https://stackoverflow.com/questions/52386085/java-lang-classnotfoundexception-helloworld-class)
> 2. [What does "Could not find or load main class" mean?](https://stackoverflow.com/questions/18093928/what-does-could-not-find-or-load-main-class-mean)
> 3. [PATH and CLASSPATH](https://docs.oracle.com/javase/tutorial/essential/environment/paths.html)
> 4. [Java Algorithms and Clients](https://algs4.cs.princeton.edu/code/)


