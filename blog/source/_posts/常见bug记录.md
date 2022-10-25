---
title: 常见bug记录
date: 2022-05-13 01:37:45
categories:
- 环境配置
tags:
- bug
---

本文记录在操作系统、浏览器中常见的bug及其解决方案。

<!--more-->

1.`github`中网址前缀为`raw.githubusercontent.com`的资源（图片、文档等）无法访问。

解决方案：

根据 [解决 raw.githubusercontent.com 无法访问的问题](https://learnku.com/articles/43426)，可能是由于某些原因导致 DNS 被污染，Windows系统上可以通过修改 `hosts` 文件解决该问题，步骤如下：

- 1）通过 [IPAddress.com](https://www.ipaddress.com/) 查询域名 `raw.githubusercontent.com`所在网址：

![image-20220513014334969](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-14/image-20220513014334969.png)

- 2）在路径 `C:\Windows\System32\drivers\etc`下的`hosts`文件最后一行添加如下信息：

  `185.199.108.133 raw.githubusercontent.com`

  PS：可能`hosts`文件为只读文件，此时需要右键单击`hosts`文件修改其访问权限。

</br>

2.在 Windows10 系统中，当默认浏览器设置为谷歌浏览器时，偶尔会出现点击其他应用（如飞书，TIM等）的链接无法跳转到谷歌浏览器打开此链接的情况。

解决方案：重启电脑

</br>

3.在Windows 10系统中，当使用`git bash`终端进行package的安装和初始化时，无法在选项之间切换，如下图所示，无法在 No/Yes 的两个选项之间切换：

![image-20220515153036885](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515153036885.png)

查询资料可知，`git bash`终端并不提供交互功能，因此无法进行选项切换。

解决方案：使用`cmd`或者其他的终端

>  参考资料：
>
> 1. [Can't use arrow keys in Git Bash (Windows)](https://stackoverflow.com/questions/55753151/cant-use-arrow-keys-in-git-bash-windows)

</br>

4.在 [Convertio](https://convertio.co/zh/document-converter/) 进行文件转换时，转换完成后在点击文件下载按钮后会跳转到 403 Forbidden 页面，如下图所示：

![image-20220515162921157](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515162921157.png)

推测原因：在通过 ssr 挂代理后，Convertio 不允许代理 ip 进行文件下载

解决方案：关闭 ssr 代理

</br>

5.在使用 VSCode 编辑器进行开发时，常常会遇到编辑环境变量并保存后在 VSCode 终端中环境变量并未同步更新的情况，即使关闭 VSCode 后重新打开也是如此，如下图所示，在将`gcc.exe`所在`bin`添加到环境变量后在 VSCode 终端中并没有更新：

![image-20220516004330258](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220516004330258.png)

猜测：VSCode 将之前的环境变量缓存起来，在重新启动 VSCode 后环境变量并未及时更新

解决方案：安装 `Chocolatey` 并运行`refreshenv` 命令更新环境变量

步骤：

- 关闭代理（在打开代理的情况下安装策略不同）

- 以管理员身份运行`cmd`并运行如下命令：

  ```
  @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
  ```

- 关闭`cmd`终端后重新打开以更新`Chocolatey`配置

- 在打开的`cmd`终端中运行`refreshenv`更新环境变量：

  ![image-20220516004936734](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220516004936734.png)

- 在`cmd`终端中运行`code`启动 VSCode

- 在打开的 VSCode 的`cmd`终端中运行`gcc`发现环境变量更新成功：

  ![image-20220516005208707](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-16/image-20220516005208707.png)

PS：遗憾的是，通过右键文件夹打开 VSCode 环境变量仍然未更新

>  参考资料：
>
> 1. [VS Code Refresh Integrated Terminal Environment Variables without Restart/Logout](https://stackoverflow.com/questions/54653343/vs-code-refresh-integrated-terminal-environment-variables-without-restart-logout)
> 2. [Installing Chocolatey](https://docs.chocolatey.org/en-us/choco/setup#installing-chocolatey)
> 3. [Is there a command to refresh environment variables from the command prompt in Windows?](https://stackoverflow.com/questions/171588/is-there-a-command-to-refresh-environment-variables-from-the-command-prompt-in-w)

</br>

6.powershell启动时出现以下报错信息：

```po
File D:\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1 cannot be loaded because running scripts is disabled on this system
```

- 以管理员身份打开powershell
- 运行`set-executionpolicy remotesigned`
- 键入`A` (Yes to All) 并回车

> 参考资料：
>
> 1. [What's Wrong With My Windows PowerShell](https://answers.microsoft.com/en-us/windows/forum/all/whats-wrong-with-my-windows-powershell/f05e72f2-a429-4ee0-81fb-910c8c8a1306)

7.IDM 总是在浏览器中下载`f.txt`文件。

原因：

在五年之前，在全世界的许多网站上都会显示Google 的广告，但由于谷歌广告系统的一个小 error，广告可能不会显示，而是将 code 下载到一个 txt 文件中，Google 已经采取措施解决了这个问题；五年前在 Firefox 浏览器上也出现了这个问题，但过几天就被解决了。对使用各种浏览器的用户来说，当浏览各种类型的网站时这个问题偶尔会发生，但是没有什么风险。

解决方案：

- 清楚浏览器缓存
- 更新浏览器到最新版本
- 进行全面的病毒扫描

> 参考资料：
>
> 1. [F.txt Help! Computer keeps downloading “f.txt” file…](https://howtofix.guide/f-txt-not-virus/)
> 2. [Google Chrome forcing download of "f.txt" file](https://stackoverflow.com/questions/28535603/google-chrome-forcing-download-of-f-txt-file)

</br>

8.打开 Microsoft Store 时出现 0x800704cf 错误，如下图所示：

![image-20220608231526381](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220608231526381.png)

> 参考资料：
>
> 1. [0x800704cf](https://answers.microsoft.com/zh-hans/windows/forum/all/0x800704cf/0949346c-ed7c-40d6-a72d-3dd2fd3d0306)

</br>
9.当默认浏览器为google时，点击其他应用内的链接无法跳转到浏览器。
解决方案：在系统设置默认应用里将Web browser改为其他浏览器，发现可以正常跳转链接，然后改回为google，重启电脑
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220917112759.png)

> 参考资料：
> 1. [win10点击超链接无法跳转到浏览器](https://blog.csdn.net/xichengqc/article/details/102988258)

</br>
10.点击Windows10 OneDrive应用没有反应
> 参考资料：
> 1. [OneDrive无法打开的原因](https://zhuanlan.zhihu.com/p/343335173)

</br>
在git bash终端运行java程序无法通过`Ctrl+Z`终止`hasNext()`输入，如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221001224702.png)
其原因在于git bash不支持通过`Ctrl+Z`发送Signal 18即`SIGTSTP`信号。在cmd中运行会发现`Ctrl+Z`后回车可以正常终止输入：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221001225100.png)

> 参考资料：
> 1. [python error Suppressing signal 18 to win32](https://stackoverflow.com/questions/50110571/python-error-suppressing-signal-18-to-win32)

</br>
11.使用conda创建新环境时出现以下报错：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016160914.png)
1. `conda clean -i`
2. 删除`.condarc`文件，关闭代理
3. 重新打开Anaconda Prompt窗口，运行命令创建新环境。（此时删除了所有镜像，使用官方源进行安装，未尝试镜像源安装）
> 参考资料：
> 1. [conda install packages error: Collecting package metadata (current_repodata.json): failed](https://stackoverflow.com/questions/61134985/conda-install-packages-error-collecting-package-metadata-current-repodata-json)
> 2. [Managing channels](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html)

</br>
12.安装`pydensecrf`pip包时出现如下报错：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221025102938.png)
提示`Microsoft Visual C++ 14.0 or greater is required`。
解决方案：
1. 按照参考资料3安装对应Visual Studio工具
安装后使用`pip install pydensecrf`安装出现新的报错：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221025111316.png)
2. 参照参考资料4将`pydensecrf-1.0rc2-cp38-cp38-win32.whl`下载到本地安装后出现如下报错：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221025111520.png)
更换为`amd64.whl`版本后解决。
> 参考资料：
> 1. [Error "Microsoft Visual C++ 14.0 is required (Unable to find vcvarsall.bat)"](https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat)
> 2. [Pip error: Microsoft Visual C++ 14.0 is required](https://stackoverflow.com/questions/44951456/pip-error-microsoft-visual-c-14-0-is-required)
> 3. [Error "Microsoft Visual C++ 14.0 is required (Unable to find vcvarsall.bat)"](https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat)
> 4. [Pydensecrf 安装报错_MenahemLi的博客-CSDN博客](https://blog.csdn.net/qq_36978986/article/details/108130879)
> 5. [Archived: Python Extension Packages for Windows - Christoph Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pydensecrf)



















