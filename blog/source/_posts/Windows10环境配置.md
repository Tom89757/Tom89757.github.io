---
title: Windows10环境配置
date: 2022-09-10 17:53:28
categories:
- 环境配置
tags:
- Windows10
---
本文记录一些Windows10系统中的配置和对应工具：
<!--more-->

### Windows多屏使用技巧
安装软件：
1. PowerToys -> FancyZones：屏幕分区
2. Twinkle Tray：多屏亮度调节

> 参考资料：
> 1. [在 Windows 下使用多块屏幕的你，可以收下这些建议](https://sspai.com/post/66381)

### Sumatra PDF和Everything的冲突
在使用Everything搜索pdf文件后，如果直接双击打开pdf文件，此时Sumatra PDF以管理员身份运行。如果直接在文件夹中双击pdf文件，此时会报错。
解决方案：Everything -> 工具 -> 选项 -> 取消勾选以管理员身份运行 -> 启动Everything服务 -> 重启Everything
> 参考资料：
> 1. [Why "SumatraPDF is running as admin and cannot open files from a non-admin process" error? · Discussion #2316 · sumatrapdfreader/sumatrapdf · GitHub](https://github.com/sumatrapdfreader/sumatrapdf/discussions/2316)
> 2. [FAQ - voidtools](https://www.voidtools.com/faq/#how_do_i_prevent_the_uac_prompt_when_running_everything)

### Golden配置屏幕取词
安装AutoHotKey1.1
> 参考资料：
> 1. [GoldenDict和AutoHotKey的安装和使用_SANGF_的博客-CSDN博客](https://blog.csdn.net/sangfengcn/article/details/75731410)
> 2. [[转载]GoldenDict 上的那些精美版权词典（附下载地址）（英语、_细草_微风_新浪博客](https://blog.sina.com.cn/s/blog_797a6edf0102wteg.html)


### 将任意键映射到Shortcuts

> 参考资料：
> 1. [How to Remap Any Key or Shortcut on Windows 10](https://www.howtogeek.com/710290/how-to-remap-any-key-or-shortcut-on-windows-10/)


### 关闭特定快捷键 (specific shortkeys)

> 参考资料：
> 1. [How to Disable Keyboard Shortcuts on Windows 10](https://www.makeuseof.com/windows-10-disable-keyboard-shortcuts)
> 2. [How to Disable Specific Windows Key Shortcut in Windows | Password Recovery](https://www.top-password.com/blog/disable-specific-windows-key-shortcut/)
> 3. [Disable Win+Space keyboard-layout switch in Windows 10 - Super User](https://superuser.com/questions/1000678/disable-winspace-keyboard-layout-switch-in-windows-10#:~:text=go%20to%20Edit%20language%20and,Layout%22%20to%20%22Not%20Assigned%22)

### PowerToys Keyboards Manager只在某些应用里生效
参考资料1给出如何进行映射；参考资料2给出如何查看进程名（process name）
> 参考资料：
> 1. [PowerToys Keyboard Manager utility for Windows | Microsoft Learn](https://learn.microsoft.com/en-us/windows/powertoys/keyboard-manager)
> 2. [Finding the Process ID - Windows drivers | Microsoft Learn](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/finding-the-process-id)
> 3. [Exclude Target apps for Keyboard Manager!!!!!!!!! · Issue #10800 · microsoft/PowerToys · GitHub](https://github.com/microsoft/PowerToys/issues/10800)

### 谷歌浏览器（Chrome）使用Vimium技巧
**指定搜索引擎搜索**
1. 按下`o`打开多功能搜索框。（`o`搜索内容在当前标签页，`O`在新标签页打开）
2. 按下`b+空格`指定搜索引擎为百度（搜索引擎可以进行设置）
3. 键入搜索内容即可用百度进行搜索。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230606104139.png)
> 参考资料：
> 1. [vimium插件设置快速搜索引擎_hampeter的博客-CSDN博客](https://blog.csdn.net/hampeter/article/details/81940035)
> 2. [vimium完全教程，各类技巧大全 - 知乎](https://zhuanlan.zhihu.com/p/30263616)

### Windows10查看所有环境变量和指定环境变量
- **指定环境变量**。`echo %VARIABLE%`
- **所有环境变量**。`SET | more`
- 使用配置文件配置环境变量
> 参考资料：
> 1. [windows - List all environment variables from the command line - Stack Overflow](https://stackoverflow.com/questions/5327495/list-all-environment-variables-from-the-command-line)
> 2. [go - Using .env files to set environment variables in Windows - Stack Overflow](https://stackoverflow.com/questions/48607302/using-env-files-to-set-environment-variables-in-windows)

### Vimium-C配置
重新映射快捷键：
```bash
map F LinkHints.activate
map f LinkHints.activateOpenInNewTab
map O Vomnibar.activate
map o Vomnibar.activateInNewTab
map B Vomnibar.activateBookmarks
map b Vomnibar.activateBookmarksInNewTab
```
> 参考资料：
> 1. [GitHub - gdh1995/vimium-c: A keyboard shortcut browser extension for keyboard-based navigation and tab operations with an advanced omnibar](https://github.com/gdh1995/vimium-c)
> 2. [shortcut - How to Remap a Key to Another Key in Vimium - Stack Overflow](https://stackoverflow.com/questions/66280656/how-to-remap-a-key-to-another-key-in-vimium)

### PowerShell安装oh-my-posh

> 参考资料：
>1.  [Oh My Posh：Windows 下的Oh my zsh - 掘金](https://juejin.cn/post/7210596158934433853)
>2. [I am not able to install a minimal theme for oh-my-posh · Issue #3756 · JanDeDobbeleer/oh-my-posh · GitHub](https://github.com/JanDeDobbeleer/oh-my-posh/issues/3756)
>3. [Home | Oh My Posh](https://ohmyposh.dev/)
>4. [GitHub - devblackops/Terminal-Icons: A PowerShell module to show file and folder icons in the terminal](https://github.com/devblackops/Terminal-Icons)
>5. [GitHub - PowerShell/PSReadLine: A bash inspired readline implementation for PowerShell](https://github.com/PowerShell/PSReadLine#upgrading)
>6. [icons don't display in VS code integrated terminal after setting terminal.integrated.fontFamily · Issue #671 · romkatv/powerlevel10k · GitHub](https://github.com/romkatv/powerlevel10k/issues/671)







