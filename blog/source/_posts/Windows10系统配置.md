---
title: Windows10系统配置
date: 2022-05-21 17:10:48
categories:
- 环境配置
tags:
- Windows10
---

本文记录一下在使用 Windows10 系统过程中的一些常见配置：

<!--more-->

## PowerShell

### 环境变量

`ls env:`：查看所有环境变量

`ls env:NODE*`：搜索环境变量

`$env:Path`：查看单个环境变量

> 参考资料：
>
> 1. [Powershell下设置环境变量](https://www.cnblogs.com/liuyt/p/5677781.html)
> 2. [关于环境变量 - PowerShell](https://docs.microsoft.com/zh-cn/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.2)

## WSL

### 安装流程

#### 前置条件

Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11

> 可以 [查看 WIndows 版本信息](https://tom89757.github.io/2022/05/22/Windows10%E5%B8%B8%E8%A7%81%E6%93%8D%E4%BD%9C/#%E6%9F%A5%E7%9C%8B-windows%E7%89%88%E6%9C%AC%E4%BF%A1%E6%81%AF)

#### 安装

- 以管理员身份运行 powershell 或 cmd

- 在打开的终端窗口中运行 `wsl --install`。

  - 当第一次使用wsl时运行该命令会默认安装 Ubuntu 系统。
  - 已经安装过wsl时运行该命令会列出帮助信息

  > 第一次打开新安装的 Linux distribution (即wsl系统)时会打开一个控制窗口，并需要等待文件解压和存储在本机上，等待即可

- 可以通过`wsl --list --online`列出可以在线安装的wsl发行版本：

  ![image-20220522210152668](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220522210152668.png)

- 可以通过`wsl --install -d kali-linux`指定安装的wsl为kali发行版：

  ![image-20220522210402831](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220522210402831.png)
### WSL代理配置
配置代理后测试代理不能使用ping，因为ping使用ICMP协议，而代理只支持TCP/UDP数据包。应该使用curl：
```bash
curl https://www.youtube.com | head -1
```
> 参考资料：
> 1. [Install Linux on Windows with WSL](https://docs.microsoft.com/en-us/windows/wsl/install)
> 2. [WSL的基本命令](https://docs.microsoft.com/zh-cn/windows/wsl/basic-commands)
> 3. [WSL2配置代理 - Leaos - 博客园](https://www.cnblogs.com/tuilk/p/16287472.html)
> 4. [为 WSL2 一键设置代理 - 知乎](https://zhuanlan.zhihu.com/p/153124468)
> 5. [httping：测量网站延迟_Linux教程_Linux公社-Linux系统门户网站](https://www.linuxidc.com/Linux/2016-05/131448.htm)
> 6. [windows上使用代理，使用ping命令会超时](https://blog.csdn.net/zhongliangtang/article/details/81280460)
> 7. [WSL2-解决无法ping通主机/配置使用主机代理_wsl 使用代理_会伏丘的博客-CSDN博客](https://blog.csdn.net/fur_pikachu/article/details/127973376)

### WSL普通用户无法登录
`wsl -u root`：
> 参考资料：
> 1. [windows subsystem for linux - [process exited with code 1], can't open WSL, zsh - Stack Overflow](https://stackoverflow.com/questions/67261530/process-exited-with-code-1-cant-open-wsl-zsh)
## cmd
### 别名设置
在windows系统中可以使用类似Linux中`alias`的别名设置。
- 编写`cmd_auto.bat`脚本文件，内容如下：
```bash
@doskey javac=javac -encoding utf8
@doskey ls=dir /b
@doskey grep=findstr
```
`@doskey`即等价于Linux中的`alias`。
- 在注册表中的`Computer\HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Command Processor`添加`AutoRun`字符串项，并设置`cmd_auto.bat`的路径，其等价于Linux中的`/etc/profile/`或`~/.bashrc`文件设置，会在cmd启动时自动运行。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221014235536.png)

> 参考资料：
> 1. [在cmd中为命令设置别名以及启动cmd时自动执行bat](https://www.cnblogs.com/fstang/archive/2013/04/06/3002006.html)
> 2. [window alias给cmd命令起别名](https://www.cnblogs.com/onelikeone/p/10783497.html)
> 3. [Windows版Grep – findstr示例](https://blog.csdn.net/cyan20115/article/details/106551191)


### cmd中文乱码
- `chcp 65001`：临时改为UTF-8编码。
> 参考资料：
> 1. [CMD中文乱码 - 码农在路上 - 博客园](https://www.cnblogs.com/lzmrex/articles/12553666.html)

### Scoop
Scoop是Windows系统下一款流行的包管理工具
推荐包：
1. [GitHub - solidiquis/erdtree: A multi-threaded file-tree visualizer and disk usage analyzer.](https://github.com/solidiquis/erdtree)：目录结构树状展示
2. [GitHub - Slackadays/Clipboard: Your external brain that remembers anything, anytime, anywhere. Save time and effort the 𝗲𝗮𝘀𝘆 way! 😄](https://github.com/Slackadays/Clipboard)：实现终端复制、剪切和粘贴。
> 参考资料：
> 1. [Scoop](https://scoop.sh/)

### cmd自动补全和命令高亮

> 参考资料：
> 1. [GitHub - chrisant996/clink: Bash's powerful command line editing in cmd.exe](https://github.com/chrisant996/clink)

### cmd默认以管理员身份运行并关闭UAC prompt

> 参考资料：
> 1. [How To Always Open Command Prompt As Administrator](https://www.intowindows.com/how-to-always-run-command-prompt-as-administrator/)
> 2. [Use the Windows Task Scheduler to run apps without UAC prompts, with admin rights](https://www.digitalcitizen.life/use-task-scheduler-launch-programs-without-uac-prompts/)
> 3. [How to Run Programs as Administrator (Elevated) without UAC Prompt » Winhelponline](https://www.winhelponline.com/blog/run-programs-elevated-without-getting-the-uac-prompt/)

### cmd配置Fira Code字体

> 参考资料：
> 1. [Font not in Windows 10 console properties list - Super User](https://superuser.com/questions/1337310/font-not-in-windows-10-console-properties-list)
> 2. [Ligatures not working with Fira Code Light in console · Issue #941 · tonsky/FiraCode · GitHub](https://github.com/tonsky/FiraCode/issues/941)
> 3. [cmd.exe support · Issue #431 · tonsky/FiraCode · GitHub](https://github.com/tonsky/FiraCode/issues/431)
> 4. [Detect path of font on Windows - Super User](https://superuser.com/questions/1658678/detect-path-of-font-on-windows)
> 5. [How to Customize the Font in Windows Command Prompt: 15 Steps](https://www.wikihow.com/Customize-the-Font-in-Windows-Command-Prompt#:~:text=Right%2Dclick%20on%20the%20top,%2C%20Font%2C%20Layout%2C%20Colors.&text=When%20you%20open%20the%20%22Options,check%20%22Quick%20Edit%20Mode%22.)

