---
title: Ubuntu常见操作
date: 2022-05-15 00:22:55
categories:
- 开发工具
tags:
- Ubuntu
---

本文记录一下使用Ubuntu操作系统时的常见操作：

<!--more-->

1.`echo $PATH | tr ":" "\n"`：在`bash`终端中分行展示环境变量。

> 参考资料：[How to split the contents of $PATH into distinct lines](https://stackoverflow.com/questions/33469374/how-to-split-the-contents-of-path-into-distinct-lines)

</br>

2.`httping -x localhost:1080 -g http://google.com -c 3`：在Ubuntu终端中测试通过代理是否能访问`google.com`。之所以使用`httping`是因为`ping`无法通过代理访问。具体步骤如下：

- 1）通过`sudo apt install httping`安装工具`httping`。
- 2）（在代理开启的情况下）运行上述命令。`-x`表示代理服务器地址；`localhost:1080`表示代理服务器为本机，监听`1080`端口；`-g`表示对其发送请求的URL，本例中为`http://google.com`；`-c`表示在结束请求前代理服务器会向目标URL发送多少 probe，此处为3。

运行结果如下：

![image-20220515111109604](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515111109604.png)

> 参考资料：[can not ping google using proxy](https://askubuntu.com/questions/428408/can-not-ping-google-using-proxy)

</br>

3.`ls`

</br>

4.`wc`

</br>

5.`zip`和`unzip`：用于压缩解压缩 zip 文件。

`zip -r filename.zip /path/to/folder1`：把`folder1`压缩到`filename.zip`。

`unzip /path/to/file.zip -d foldername`：把`file.zip`解压缩到`foldername`。

> 参考资料：
>
> 1. [Linux unzip 命令](https://www.runoob.com/linux/linux-comm-unzip.html)
> 2. [Zip all files in directory?](https://unix.stackexchange.com/questions/57013/zip-all-files-in-directory)
> 3. [How to extract a zip file to a specific folder?](https://askubuntu.com/questions/520546/how-to-extract-a-zip-file-to-a-specific-folder)

</br>

6.`rename`：用于重命名文件。

- `000001_left.png -> 000001_left_depth.png `：`rename -v 's/.png/_depth.png/' *.png `。
- `000001_left.png -> 000001_left_gt.png`：`rename -v 's/.png/_gt.png/' *.png`。
- `000001_left_GT.png -> 000001_left_gt.png`：`rename -v 's/GT.png/gt.png/' *.png`。

> 参考资料：
>
> 1. [Batch renaming files](https://unix.stackexchange.com/questions/1136/batch-renaming-files)

</br>

7.在 Ubuntu 终端中隐藏当前工作目录：

![image-20220609103802742](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220609103802742.png)

- 当前终端中生效：在当前终端中运行`export PS1='\u@\h$ '`

  ![image-20220609103954302](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220609103954302.png)

- 永久生效：在`~/.bashrc`文件末尾添加`export PS1='\u@\h$ '`，并运行`source ~/.bashrc`使之生效

- 设置颜色：`export PS1='\e[34m\u\e[0m@\e[35m\h\e[0m$ ' `。

  ![image-20220609113641179](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220609113641179.png)

PS：`\e`表示`Esc`即转义，`\e[34m`和`\e[35m`表示颜色，`\e[0m`表示清除前面的格式，故`@`和`$`符号无格式。

参考资料：

1. [Hide current working directory in terminal](https://askubuntu.com/questions/16728/hide-current-working-directory-in-terminal)

2. [How to Change / Set up bash custom prompt (PS1) in Linux](https://www.cyberciti.biz/tips/howto-linux-unix-bash-shell-setup-prompt.html)
3. [Bash tips: Colors and formatting (ANSI/VT100 Control sequences)](https://misc.flogisoft.com/bash/tip_colors_and_formatting)

</br>
