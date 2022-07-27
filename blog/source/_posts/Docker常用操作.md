---
title: Docker常用操作
date: 2022-07-27 11:10:02
categories:
- 开发工具
tags:
- Docker
---

本文记录一下使用Docker容器时的常见操作：

<!--more-->

1.查看docker中安装部署的所有容器：`docker ps -a`。

![image-20220727111314196](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727111314196.png)

</br>

2.Docker配置镜像加速：

- 从 [阿里云](https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors) 获取镜像加速地址，会得到如下个人镜像加速地址

  ![image-20220727111855809](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727111855809.png)

- 在Windows 10上安装的Docker Desktop中，进入`settings->Docker Engine`。编辑右侧json格式的configuration文件如下：

  ![image-20220727112129610](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727112129610.png)

- 点击`Apply & Restart`。此时系统会要求注销当前账号，然后重新登录，同意即可，此时镜像即配置成功。

- 验证：在任意地方打开`git bash`或者`cmd`窗口，输入`docker info`命令回车，可以看到如下信息：

  ![image-20220727112341200](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727112341200.png)

  表示配置成功。

> 参考资料：
>
> 1. [Docker 镜像加速](https://www.runoob.com/docker/docker-mirror-acceleration.html)

</br>

