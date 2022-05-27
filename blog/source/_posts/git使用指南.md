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

- `git rm -rf --cached`：将所有文件从仓库缓存中删除
- `git add ./`
- `git commimt -m "add a commit"`
- `git push origin main`

> 参考资料：[Gitignore not working](https://stackoverflow.com/questions/25436312/gitignore-not-working)
