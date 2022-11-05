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
### 出现乱码
当直接使用`javac Evaluate.java`编译文件时出现以下乱码：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221002095435.png)
此时可以通过指定编码格式解决该问题：
```bash
javac -encoding utf8 p3/Evaluate.java
```

> 参考资料：
> 1. [Javac和JVM的字符编码问题](https://www.cnblogs.com/jayson-jamaica/p/12695427.html)

## Linux下编译和运行Java引用多个外部包
以算法第四版第三章的3.1.3的用例举例为例进行说明。在windows中已经根据上述的**出现ClassNotFoundException**进行外部库的依赖配置。
此时在windows中只需在cmd窗口按照以下步骤运行用例类：
- `javac p1/TestST.java`：编译生成`TestST.class`文件
- `java p1/TestST <tinyTale.txt`：将`tinyTale.txt`文件作为标准输入运行
PS：
1. 由于cmd极弱的自动补全，每次运行都需要手敲完整文件名，很难受
2. powershell虽然有适当的自动补全，但在命令行窗口中`<`符号不能作为标准输入的重定向符。
因此，对在WSL的Linux系统中编译和运行java文件的需求应运而生。
其步骤如下：首先确保在WSL系统中已经安装好java环境，`javac`和`java`可正常运行。
- `javac -cp "/mnt/d/Develop/Java/jdk11.0.11/lib/*" p1/TestST.java`：`/mnt/d/Develop/Java/jdk11.0.11/lib/*`为外部库所在路径，其中包含`algs.jar`这个外部依赖库。
- `java -cp $CLASSPATH:"/mnt/d/Develop/Java/jdk11.0.11/lib/*" p1.TestST <p1/tinyTale.txt`：运行`.class`文件。
为了避免上述命令中繁琐的输入，可以设置`alias`别名：
```bash
alias javac="javac -cp '/mnt/d/Develop/Java/jdk11.0.11/lib/*'"
alias java="java -cp $CLASSPATH:/mnt/d/Develop/Java/jdk11.0.11/lib/*"
```
> 参考资料：
> 1. [linux 下编译和运行 java 引用多个外部包](https://blog.csdn.net/onebigday/article/details/123266336)
> 2. [linux alias 命令 查看系统设置的命令别名](https://www.cnblogs.com/mingerlcm/p/10791074.html)

### Ubuntu上安装Java

> 参考资料：
> 1. [如何在Ubuntu 18.04上安装Java(JDK11)](https://www.jianshu.com/p/5a25b9535016)


### Java中的HashMap、TreeMap和HashSet、TreeSet

> 参考资料：
> 1. [哈希表和有序表的简单介绍 - 掘金](https://juejin.cn/post/6978052334911766558)
> 2. [Map的有序和无序实现类，与Map的排序 - 龙昊雪 - 博客园](https://www.cnblogs.com/chen-lhx/p/8432422.html)
> 3. [Java HashSet | 菜鸟教程](https://www.runoob.com/java/java-hashset.html)
> 4. [Java TreeSet - Java教程 - 菜鸟教程](https://www.cainiaojc.com/java/java-treeset.html)
> 5. [Java HashMap | 菜鸟教程](https://www.runoob.com/java/java-hashmap.html)
> 6. [Java TreeMap - Java教程 - 菜鸟教程](https://www.cainiaojc.com/java/java-treemap.html)

### Java中溢出处理

> 参考资料：
> 1. [java int溢出总结 | njuCZ's blog](https://njucz.github.io/2017/08/16/java-int%E6%BA%A2%E5%87%BA%E6%80%BB%E7%BB%93/)


