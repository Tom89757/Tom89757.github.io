---
title: Obsidian+OneDrive+FolderSync多端文件同步方案
date: 2023-09-12 20:54:50
categories:
  - 资料
tags:
  - 坚果云
---
本文记录一下使用Obsidian+OneDrive+FolderSync同步手机端和电脑端文件的使用指南：
<!--more-->
主要参考资料3：
```
mklink/d E:\OneDrive\CoursesNotes D:\Notes\CoursesNotes
mklink/d E:\OneDrive\_post D:\Notes\CoursesNotes
```
> 参考资料：
> 1. [坚果云手机客户端FAQ | 坚果云帮助中心](https://help.jianguoyun.com/?page_id=864)
> 2. [Obsidian通过Remotely save插件实现坚果云webdav同步 - 经验分享 - Obsidian 中文论坛](https://forum-zh.obsidian.md/t/topic/5367?page=4)
> 3. [Android平台上obsidian如何和onedrive联动? - 知乎](https://www.zhihu.com/question/475280128/answer/2422842141)

另一种同步方案是通过Obsidian+Termux+Git实现多段同步。
> 参考资料：
> 1. [Andriod 使用 Obsidian 的客户端 | 程序员的喵](https://catcoding.me/p/obsidian-andriod-client-sync-git/)