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
> 参考资料：
> 1. [GoldenDict和AutoHotKey的安装和使用_SANGF_的博客-CSDN博客](https://blog.csdn.net/sangfengcn/article/details/75731410)
> 2. [[转载]GoldenDict 上的那些精美版权词典（附下载地址）（英语、_细草_微风_新浪博客](https://blog.sina.com.cn/s/blog_797a6edf0102wteg.html)