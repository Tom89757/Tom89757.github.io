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
9.使用git bash进行javac编译时出现中文乱码：

![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220929170956.png)

解决方案：在选项里将字符集设为GBK，重启git bash：

![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220929171040.png)

> 参考资料：
> 1. [在 git bash 里运行 java 命令，打印出的中文显示乱码](https://blog.csdn.net/qq_21260033/article/details/78786608)
> 2. [解决 Git Bash 在 windows 下中文乱码的问题](https://minsonlee.github.io/2020/11/how-to-set-utf8-with-git-bash)（不针对此问题）

</br>
10.有时需要给git终端设置别名。给git设置别名分为两种：
- 给git本身的命令设置别名，此时可以通过`git config --global alias.co checkout`设置全局别名，这样可以`git co`等同于`git checkout`，该配置会写入`~/.gitconfig`文件。故也可以直接编辑该文件来设置别名
- 给git终端运行的其他命令设置别名，如`javac -encoding utf8`简化为`javac`，此时可以编辑`~/.bashrc`文件（如果没有则创建），在里面写入`alias javac='javac -encoding utf8'`。设置后每次启动git终端窗口后该文件中的配置都会生效

> 参考资料：
> 1. [2.7 Git 基础 - Git 别名](https://git-scm.com/book/zh/v2/Git-%E5%9F%BA%E7%A1%80-Git-%E5%88%AB%E5%90%8D)
> 2. [玩转 Git 别名](https://segmentfault.com/a/1190000023541589)

</br>
11.出现报错`error: RPC failed; curl 92 HTTP/2 stream 7 was not closed cleanly before end of the underlying stream`：

![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221108182409.png)
解决方案：
1. 配置http版本：`git config --global http.version HTTP/1.1`
2. 配置`http.postBuffer：git config --global http.postBuffer 157286400`

> 参考资料：
> 1. [git - error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly: PROTOCOL_ERROR (err 1) - Stack Overflow](https://stackoverflow.com/questions/59282476/error-rpc-failed-curl-92-http-2-stream-0-was-not-closed-cleanly-protocol-erro)


</br>
12.在linux中使用git时，出现如下情况：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230203180529.png)
解决方案：在Linux的git中配置personal token
- `git config --global credential.helper cache`：在`.gitconfig`中添加配置，使得系统记住后续输入的token，避免重复输入
- `git clone https://github.com/Tom89757/dotfiles.git`：clone对应仓库，并输入对应用户名和token。由于上述配置，此次输入后后续无需再次输入。

>参考资料：
>1. [git - Message "Support for password authentication was removed. Please use a personal access token instead." - Stack Overflow](https://stackoverflow.com/questions/68775869/message-support-for-password-authentication-was-removed-please-use-a-personal#:~:text=From%202021%2D08%2D13%2C,a%20PAT%20on%20your%20system.)
>

</br>
13.clone仓库时出现：
```bash
Host key verification failed. fatal: The remote end hung up unexpectedly
```
解决方案：
```bash
git config --global user.name "你的github账户名"
git config --global user.email "你的github账户默认的邮箱地址"
ssh-keygen -t rsa -b 4096 -C "你的github账户默认的邮箱地址"
cat ~/.ssh/id_rsa.pub # 添加到git ssh
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts 
```
> 参考资料：
> 1. [我的现代化Neovim配置 - 知乎](https://zhuanlan.zhihu.com/p/382092667)
> 2. [ssh - Git error: "Host Key Verification Failed" when connecting to remote repository - Stack Overflow](https://stackoverflow.com/questions/13363553/git-error-host-key-verification-failed-when-connecting-to-remote-repository)

</br>
14.下载仓库中某个文件夹：
工具：[Download GitHub directory](https://download-directory.github.io/)

</br>

15.在运行`git push origin main`之后，在出现`Total`行之后卡住
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230415230020.png)
问题：推送的objects较大
解决方案：参考下方参考资料
> 参考资料：
> 1. [bitbucket - git push hangs after Total line - Stack Overflow](https://stackoverflow.com/questions/15843937/git-push-hangs-after-total-line)

</br>

16.查看分支的最早提交时间：
切换到对应分支例如`w6-complete`然后运行如下命令：
```bash
git log --reverse --pretty=format:"%cd" --date=iso w6-complete | tail -1
```
PS：上述命令只能查看`w6-complete`的最早提交时间，而不是分支的创建时间
> 参考资料：VSCode Copilot

</br>
17.`git log`显示commits时间并倒序输出到`commits.txt`：
```bash
git log --pretty=format:"%h %ad | %s%d [%an]" --graph --date=short | tac >commits.txt
```

</br>
18.git添加PATH：
> 参考资料：
> 1. [node.js - How come npm install doesn't work on git bash - Stack Overflow](https://stackoverflow.com/questions/35915832/how-come-npm-install-doesnt-work-on-git-bash)
> 2. [node.js - Http-Server Command Not Found After Install in NPM - Stack Overflow](https://stackoverflow.com/questions/31694646/http-server-command-not-found-after-install-in-npm#:%7E:text=you%20should%20add%20the%20npm,%22%2C%20run%20the%20below%20command.&text=bashrc%20file%20in%20your%20home,commands%20on%20your%20git%20bash)

</br>

19.git配置pager：
GitHub Copilot: Git pager是指Git命令行工具中用于分页显示输出的程序。Git pager通常用于显示长时间运行的命令的输出，例如git log或git diff。当输出超过终端窗口的大小时，Git pager会自动将输出分页显示，以便您可以逐页查看输出。

默认情况下，Git pager使用less程序作为分页程序。您可以使用`git config`命令来配置Git pager的选项。例如，您可以使用以下命令将Git pager配置为使用more程序而不是less程序：
```bash
git config --global core.pager more
```
您还可以使用以下命令来禁用Git pager：
```bash
git config --global core.pager ''
```
这将使Git命令在终端窗口中直接输出所有内容，而不是分页显示。
> 参考资料：
> 1. [How do I prevent 'git diff' from using a pager? - Stack Overflow](https://stackoverflow.com/questions/2183900/how-do-i-prevent-git-diff-from-using-a-pager)

</br>
20.git bash使用`git diff`，不显示任何东西。
问题：错误的使用方式
解决方案：按照下述流程使用：
```bash
git add file_name

git diff --cached file_name
```
> 参考资料：
> 1. ['git status' shows changed files, but 'git diff' doesn't - Stack Overflow](https://stackoverflow.com/questions/14564946/git-status-shows-changed-files-but-git-diff-doesnt) 

</br>
21.git显示警告：
```bash
warning: lF will be replaced by CRLF.
```
解决方案：
```bash
git config --global core.autocrlf false
```
> 参考资料：
> 1. [Windows git "warning: LF will be replaced by CRLF", is that warning tail backward? - Stack Overflow](https://stackoverflow.com/questions/17628305/windows-git-warning-lf-will-be-replaced-by-crlf-is-that-warning-tail-backwar)

</br>
22.git设置符号链接（软链接 `ln -s`）：
- 打开Windows 10中的开发者模式（"Developer Mode"），从而给`mklink`权限
- 使得git中symbol links生效：
```bash
git config --global core.symlinks true # 全局生效
git config core.symlinks true # 当前仓库生效
```
- 添加symbol link链接：
```bash
mklink C:\Users\26899\.bash_aliases D:\Desktop\dotfiles\git\.bash_aliases
```
> 参考资料：
> 1. [Symbolic link does not work in Git over Windows - Super User](https://superuser.com/questions/1713099/symbolic-link-does-not-work-in-git-over-windows)
> 2. [Git symbolic links in Windows - Stack Overflow](https://stackoverflow.com/questions/5917249/git-symbolic-links-in-windows)

</br>
23.git bash配置定制（custom）的`git-prompt.sh`文件：
1. 找到`git-prompt.sh`文件，在Git安装目录下。
2. 将其复制到`~/.config/git/`目录下。
3. 即可编辑上述复制的`git-prompt.sh`文件对git prompt进行定制。
PS：同理，可以对使用oh-my-zsh的git prompt进行定制，其文件位于`/c/Users/26899/.oh-my-zsh/plugins/gitfast/git-prompt.sh`。
> 参考资料：
> 1. [How to change the display name in Git bash prompt](https://www.brainstormcreative.co.uk/git-bash/how-to-change-the-display-name-in-git-bash/)

















