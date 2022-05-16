---
title: MySQL使用指南
date: 2022-05-15 21:51:14
categories:
- 开发工具
tags:
- MySQL
---

本文记录一下使用 MySQL 时的常见配置和操作。

<!--more-->

1.启动 MySQL 服务。启动 MySQL 有两种方式：

- 以管理员 (administrator) 方式运行 `cmd`，在`cmd`终端中运行`net start mysql`。显示如下界面表示启动成功：

  ![image-20220515220235630](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515220235630.png)

  PS：关闭该终端 MySQL 仍然保持运行

  此时可能出现以下界面：

  ![image-20220515215652080](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515215652080.png)

  原因：Windows 系统中 MySQL 服务被禁用

  解决方案：`Win + R`打开`run`窗口，输入`services.msc`打开`Services`窗口，查看其中的 MySQL服务并将其 status 由 `Disabled` 改为`Manual`。如下图所示：

  ![image-20220515215940920](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515215940920.png)

- 定位到 MySQL 安装目录的 `bin`文件夹，如`D:\Develop\MySQL\bin`。在此处打开 `cmd` 终端，并运行 `mysqld --console`，显示如下界面表示启动成功：

  ![image-20220515220421568](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515220421568.png)

  PS：关闭该终端或者`Ctrl + C`都会使得 MySQL 服务关闭

</br>

2.关闭 MySQL 服务。与启动对应，同样有两种方式：

- 以管理员 (administrator) 方式运行 `cmd`，在`cmd`终端中运行`net stop mysql`。显示如下界面表示关闭成功：

  ![image-20220515220939116](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515220939116.png)

- 定位到 MySQL 安装目录的 `bin`文件夹，如`D:\Develop\MySQL\bin`。在此处打开 `cmd` 终端（在上面的启动终端之外另开一个终端），并运行`mysqladmin -uroot -p shutdown`并输入对应密码：

  ![image-20220515221417975](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515221417975.png)

  在启动终端中显示如下界面表示关闭成功（也可在启动终端中通过`Ctrl+C`快捷键关闭，不推荐）：

  ![image-20220515221316977](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515221316977.png)

PS：MySQL 服务的启动方式和关闭方式相对应

</br>

3.登录MySQL 用户：在启动 MySQL 后，在终端运行 `mysql -u root -p`后输入对应密码登录 root 用户（可更改用户名登录其他用户）。出现如下界面表示登录成功：

![image-20220515221935109](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515221935109.png)

</br>

4.退出 MySQL 用户：在登录成功后，可通过`quit`或者`Ctrl + C`退出当前用户登录。出现如下界面表示退出成功：

![image-20220515222217664](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220515222217664.png)

> 以上操作参考资料为：
>
> 1. [restart mysql server on windows7](https://stackoverflow.com/questions/12972434/restart-mysql-server-on-windows-7)
> 2. [MySQL 教程 - 菜鸟教程](https://www.runoob.com/mysql/mysql-tutorial.html)

</br>
