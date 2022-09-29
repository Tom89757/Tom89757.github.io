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

6.有时需要在`.gitignore`文件中添加仓库中所有的名为`folder_name`文件夹或文件，此时可以通过在`.gitignore`中添加如下内容实现：

```bash
*folder_name*
```

例如，要忽略所有名为`datafile`和`pth`的文件夹或文件，可以在`.gitignore`中添加：

```python
*datafile*
*pth*
```

PS：此种方式由于使用了极其宽松的正则表达式，凡是文件夹名或文件名中包含`datafile`或`pth`的连续字符串都将被忽略。

</br>

7.同时进行多个项目的开发时，对`git commmit -m "msg"`中的`msg`没有过多要求的情况下，可以通过Windows `.bat`脚本对多个项目进行批量的`git pull`和`git push`。

- `git pull`：

  ```bash
  @echo off
  echo "batch git pull"
  D:
  
  echo "moving to D:\Projects\AndroidProjects"
  cd D:\Projects\AndroidProjects
  echo git pull D:\Projects\AndroidProjects >>D:\Desktop\pull.txt
  git pull >>D:\Desktop\pull.txt
  echo "D:\Projects\AndroidProjects git pull finish"
  echo= >>D:\Desktop\pull.txt
  echo=
  
  ...
  
  pause
  ```

  上面演示了对`AndroidProjects`项目进行`git pull`操作，并将相关输出记录到`pull.txt`文件。最后的`pause`命令使得执行`.bat`脚本执行完毕后停留在`cmd`页面。

- `git push`：

  ```bash
  @echo off
  echo "batch git push"
  D:
  
  echo "moving to D:\Desktop\Tom89757.github.io"
  cd D:\Desktop\Tom89757.github.io
  echo git push D:\Desktop\Tom89757.github.io >>D:\Desktop\push.txt
  git add ./ >>D:\Desktop\push.txt
  git commit -m "update" >>D:\Desktop\push.txt
  git push origin hexo >>D:\Desktop\push.txt
  echo "D:\Desktop\Tom89757.github.io git push finish"
  echo= >>D:\Desktop\push.txt
  echo= >>D:\Desktop\push.txt
  echo=
  cd D:\Desktop\Tom89757.github.io\blog
  hexo d -g
  
  pause
  ```

  上面演示了对`Tom89757.github.io`项目进行`git push`操作，并将相关输出记录到`push.txt`文件。并在完成`git push`操作后，对`blog`中的内容进行生成和部署。

> 参考资料：
>
> 1. [Windows 下bat脚本git提交代码到github](https://blog.csdn.net/Ep_Little_prince/article/details/108895103)
> 2. [bat 批处理教程](https://www.w3cschool.cn/dosmlxxsc1/wvqyr9.html)
> 3. [bat脚本echo命令输出空行的11种方法和效率](https://blog.csdn.net/justlpf/article/details/120077423)

</br>

8.当`git push`较大文件（大于50M）时，会出现如下`warning`。
此时可以通过`git lfs`来解决，其步骤如下：
1. 安装git bash，运行`git lfs install`。注意对每个user account只运行一次
2. 在每个你想要使用`Git LFS`的仓库，训责你想要用`Git LFS`管理的文件类型（或者直接编辑你的`.gitattributes`文件），可以在任何使用配置额外的扩展文件类型。例如：`git lfs track ".pptx"`。将`.pptx`文件类型添加到`Git LFS`管理的文件类型中。
3. 上述操作会在当前仓库根目录下添加`.gitattributes`文件，其内容如下：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220903153435.png)
4. 使用`git add .gitattributes`便可使`Git LFS`生效。
> 参考资料：
> 1. [Git Large File Storage](https://git-lfs.github.com/)

</br>
5.使用git bash进行javac编译时出现中文乱码：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220929170956.png)
解决方案：在选项里将字符集设为GBK，重启git bash：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220929171040.png)

> 参考资料：
> 1. [在 git bash 里运行 java 命令，打印出的中文显示乱码](https://blog.csdn.net/qq_21260033/article/details/78786608)
> 2. [解决 Git Bash 在 windows 下中文乱码的问题](https://minsonlee.github.io/2020/11/how-to-set-utf8-with-git-bash)（不针对此问题）