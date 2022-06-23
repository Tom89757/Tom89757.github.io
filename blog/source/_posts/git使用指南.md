---
title: git使用指南
date: 2022-05-12 17:54:26
categories:
- 开发工具
tags:
- git
---

本文记录一下使用git时的常见操作。

<!--more-->

1.`git remote -v`：查看当前项目远程地址。

</br>

2.`echo $PATH | tr ":" "\n"`：在`git bash`终端中分行展示环境变量。

> 参考资料：[How to split the contents of $PATH into distinct lines](https://stackoverflow.com/questions/33469374/how-to-split-the-contents-of-path-into-distinct-lines)

</br>

3.在未将修改的本地文件推送到远程仓库的状态下，通过 `git pull` 拉取远程仓库的内容时会出现`conflicts`，如下图所示：

![image-20220516212248506](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516212248506.png?token=AKWAGW4TWAIV3UZMNZVZDRLCQJLUY)

此时可以通过如下步骤解决`conflicts`：

- `git fetch origin`：下载远程分支的所有变动，但不与本地分支合并

- `git pull origin hexo`：下载远程`hexo`分支的所有变动并与本地分支合并。此时会出现如下情形

  ![image-20220516212734022](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516212734022.png?token=AKWAGWYZ35VBES4T4KXFT5LCQJLUG)

- `git status`：查看当前分支状态。

  ![image-20220516212832926](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516212832926.png?token=AKWAGWYTCJTJYHQH53NVMFDCQJLUS)

- `git add ./`：添加指定目录到暂存区，包括子目录
- `git commit -m "update"`：提交暂存区到仓库区
- `git pull origin hexo`：发现`Already up to date.`
- `git status`：发现`nothing to commit, working tree clean`

> 参考资料：[How do I resolve merge conflicts in a Git repository?](https://stackoverflow.com/questions/161813/how-do-i-resolve-merge-conflicts-in-a-git-repository)

</br>

4.在文件或者文件夹已经存在在仓库中时，将这些文件或者文件夹加入`.gitignore`文件后，git 并不会将这些文件或者文件夹删除。此时可以通过以下步骤使`.gitignore`中的改动生效：

- `git rm -rf --cached path_to_file`：将对应路径的文件从仓库缓存中删除；
- `git add ./`
- `git commim -m "add a commit"`
- `git push origin main`

> 参考资料：
>
> 1. [Gitignore not working](https://stackoverflow.com/questions/25436312/gitignore-not-working)
> 2. [Can you have multiple Gitignore files?](https://www.quora.com/Can-you-have-multiple-Gitignore-files)
>
> PS：可以为仓库中的每个文件夹创建一个`.gitignore`，但是并不建议，因为不方便查询和管理。

</br>

5.给 git 配置代理：

- 配置 socks 协议代理：

  1. 设置代理：

     ```
     git config --global http.proxy 'socks5://127.0.0.1:1080' 
     git config --global https.proxy 'socks5://127.0.0.1:1080'
     ```

  2. 查看代理：

     ```
     git config --global --get http.proxy
     git config --global --get https.proxy
     ```

  3. 取消代理：

     ```
     git config --global --unset http.proxy
     git config --global --unset https.proxy
     ```

- 配置 http 协议代理：

  1. 设置代理：

     ```
     git config --global http.proxy 'http://127.0.0.1:1080' 
     git config --global https.proxy 'https://127.0.0.1:1080'
     ```

  2. 查看代理：

     ```
     git config --global --get http.proxy
     git config --global --get https.proxy
     ```

  3. 取消代理：

     ```
     git config --global --unset http.proxy
     git config --global --unset https.proxy
     ```

  PS：1080 为在 ShadowSocksR 或者 V2rayN 客户端中设置的代理的端口；Windows V2rayN 客户端似乎不支持 http 协议代理（注意在 git 更新 personal access token 时不支持 socks 协议）
  
  > 参考文献：
  >
  > 1. [git设置、查看、取消代理](https://www.cnblogs.com/yongy1030/p/11699086.html)