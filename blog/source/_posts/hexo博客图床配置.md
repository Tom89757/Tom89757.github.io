---
title: hexo博客图床配置
date: 2022-05-08 20:24:35
categories:
- 环境配置
tags:
- hexo博客
---

本文记录一下如何将本地`.md`文件中图片上传到 github 仓库或腾讯云对象存储中，并在本地`.md`文件中将图片地址替换为对应的 github 或腾讯云存储链接。

<!--more-->

### 上传至 github 仓库

该方法的思路是在 github 中新建一个仓库（必须为public仓库，否则无法访问）作为图床，并通过 PicGo 软件将本地图片上传至对应的 github 仓库。具体步骤如下：

#### 准备工作

- Typora 0.11.18 (beta)：[下载地址](https://typora.en.uptodown.com/windows/versions)
- PicGo 2.3.0：[下载地址](https://github.com/Molunerfinn/PicGo/releases)
- nodejs v16.13.0

#### 软件配置

- 打开 PicGo 在图床配置里选择 GitHub 图床进行配置，并设为默认图床

  ![image-20220516220239293](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516220239293.png?token=AKWAGW46DCB4MTDZJIFCFR3CQJMZU)

- 打开 Typora 偏好设置并进行图像设置：

  ![image-20220516220428369](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516220428369.png?token=AKWAGW6CJAWAMQRBNJQZ4RLCQJNAO)

- 编辑 markdown 文档会发现插入文档的图片会自动上传至 github 仓库并将图片地址更新为对应 github 仓库地址。也可右键图像手动上传：

  ![image-20220516220638096](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516220638096.png?token=AKWAGW6R34UTLRBW7F5KINLCQJNIS)

> 参考资料：[为Typora配置图床,实现图片自动上传](https://www.jianshu.com/p/4740993c5843)

### 上传至腾讯云对象存储

