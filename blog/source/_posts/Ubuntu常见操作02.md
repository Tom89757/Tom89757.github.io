---
title: Ubuntu常见操作02
date: 2022-11-22 20:49:01
categories:
- 开发工具
tags:
- Ubuntu
---
本文记录一下使用Ubuntu操作系统时的常见操作：
<!--more-->
1.`sed`命令中的`.`字符。`.`用于匹配除换行符之外的任意单个字符，它必须匹配一个字符。通过这种形式加上正则表达式的贪婪匹配（匹配符合模式的最长字符串）可以进行如下替换操作：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221122221549.png)
其命令为`head scores2.txt | sed 's#.*/##'。
> 参考资料：
> 1. 《Linux命令行于shell脚本编程大全》第三版20.2.4点号字符

</br>
2.删除文件时出现`cannot remove:device or resource busy`：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221123001238.png)
解决方案：
- `lsof +D /path`：查看当前路径下哪些进程占用文件
- `kill -9 $PID`：关闭对应进程id
- `rm -rf ./*`：重新尝试删除文件
一行命令实现：
`lsof +D ./ | awk '{print $2}' | tail -n +2 | xargs -r kill -9`
类似命令：
`ps -ef | grep FT | awk '{print $2}' | xargs -r kill -9`
> 参考资料：
> 1. [files - How to get over "device or resource busy"? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/11238/how-to-get-over-device-or-resource-busy)

</br>
3.bash脚本中的字符串比较：
```bash
if [[ "$dir" == "image" ]]; then
	img=DUTS/$dir/$line".jpg"
else
	img=DUTS/$dir/$line".png"
fi
```
> 参考资料：
> 1. [Bash shell字符串比较 | myfreax](https://www.myfreax.com/how-to-compare-strings-in-bash/#:~:text=%E5%9C%A8Bash%E4%B8%AD%E6%AF%94%E8%BE%83%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%97%B6%EF%BC%8C%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8%E4%BB%A5%E4%B8%8B,%E4%BD%A0%E5%BA%94%E8%AF%A5%E4%B8%8E%20%5B%20%E9%85%8D%E5%90%88%E4%BD%BF%E7%94%A8%E3%80%82)
> 2. [Bash Shell字符串比较入门_Linux教程_Linux公社-Linux系统门户网站](https://www.linuxidc.com/Linux/2019-05/158678.htm)

</br>
4.由于Windows系统和Linux系统中换行符的差异（前者为\n\r，后者为\n），在WSL进行批量处理时可能出现以下问题：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221203144329.png)
其解决方式是，在遍历时进行`sed`替换操作替换掉`\r`：
```bash
cat img_1000.txt | sed 's/\r//' | while read line
do
    img=./image/$line".jpg"
    cp $img ./DUTS1000/
    # echo "$img"
done
```

</br>
5.`awk`指令。
- 打印第4列：`cat test.txt | awk '{print $4}'`
- 打印第1，3列：`cat test.txt | awk '{print $1, $3}'`
> 参考资料：
> 1. [Fetching Title#90mq](https://www.runoob.com/linux/linux-comm-awk.html)

</br>
6.sed指令匹配模式，并替换模式中的一部分，保留剩余部分（向后查找，回溯）：
```bash
sed 's/hello \(world\)/hi \1/' file.txt
```
可以将`hello world`替换为`hi world`。

</br>
7.由于Linux和Windows系统中换行符的差异，有时需要将`\r`替换为空字符。
```bash
$ cat test.txt | sed 's/\r//' >a.txt
$ sort b.txt a.txt a.txt | uniq -u > b-a.txt
```

</br>
8.使用FreePic2Pdf给书制作目录时，从豆瓣或使用OCR对应的目录txt文本往往如下图所示：
```txt
第1章 基础：逻辑和证明 1
1.1 命题逻辑 1
1.1.1 引言 1
1.1.2 命题 1
1.1.3 条件语句 4
1.1.4 复合命题的真值表 7
1.1.5 逻辑运算符的优先级 8
```
根据FreePic2Pdf要求，需要将末尾的" "(空格)+数字转换为"\t"(tab键)+数字。此时可以使用vim中的替换（使用子模式匹配），命令如下：
```vim
/\v( )([0-9]**)$
```
先使用上述正则语法，然后回车找到末尾的`( )(页码)$`模式；
```vim
:%s//\t\2
```
再使用上述的替换命令，将页码前的空格替换为`\t`。
同样可以使用类似命令在行首添加`\t`：
```vim
/\v^([0-9]*).([0-9]*) #末尾包含空格
:%s//\t\1.\2 #末尾包含空格
```
更精准的做法是：
```vim
/\v^([0-9]{1,2}.[0-9]{1.2} ) #末尾包含空格，{1,2}表示匹配1或2次
:%s//\t\1
```
还可以使用`^`排除匹配：
```vim
/\v^([^第\t]) #[^第\t]表示排除对"第"和"\t"的匹配
:%s//\t\1
```
> 参考资料：
> 1. 《Vim实用技巧》技巧94
> 2. [VIM学习笔记 正则表达式-进阶 (Regular Expression Advanced)](http://yyq123.github.io/learn-vim/learn-vi-82-RegularExpressionAdv.html)

</br>
9.当使用pip或者conda安装新的package时，可能出现"No Space Left on Device"。此时推荐的做法是本文档的第2点，杀掉自己的所有进程然后重新连接服务器。下面查找到的其它的方法均需要root权限，并不推荐。
> 参考资料：
> 1. [Top 3 Ways to Fix “No Space Left on Device” Error in Linux](https://helpdeskgeek.com/linux-tips/top-3-ways-to-fix-no-space-left-on-device-error-in-linux/)
> 2. [How to Fix the "No Space Left on Device" Error on Linux - Make Tech Easier](https://www.maketecheasier.com/fix-linux-no-space-left-on-device-error/)
> 3. [privileges - lsof: WARNING: can't stat() fuse.gvfsd-fuse file system - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/171519/lsof-warning-cant-stat-fuse-gvfsd-fuse-file-system)


</br>
10.当使用`grep`指令时，搭配正则表达式，例如或运算：
```bash
grep 'fatal\|error\|critical' /var/log/nginx/error.log # 使用\转义
grep -E 'fatal|error|critical' /var/log/nginx/error.log
```
> 参考资料：
> 1. [Linux grep中的正则表达式Regex | myfreax](https://www.myfreax.com/regular-expressions-in-grep/)

</br>
11.Ubuntu卸载软件：
```bash
sudo apt-get remove lua5.3 #只去除lua5.3
sudo apt-get remove --auto-remove lua5.3 #去除lua5.3及其依赖packages
sudo apt-get purge lua5.3 # 使用purge，所有配置和依赖packages将被移除
sudo apt-get purge --auto-remove lua5.3 # 使用auto remove选项时，将根据该package来去除，在你想要重装时很有用
```
> 参考资料：
> 1. [在ubuntu系统中删除软件的三种最佳方法_51CTO博客_ubuntu 卸载软件](https://blog.51cto.com/u_168360/2407085)
> 2. [server - How to completely remove virtual packages? - Ask Ubuntu](https://askubuntu.com/questions/207505/how-to-completely-remove-virtual-packages)
> 3. [How to uninstall or remove lua5.3 software package from Ubuntu 17.04 (Zesty Zapus)](https://www.thelinuxfaq.com/ubuntu/ubuntu-17-04-zesty-zapus/lua5.3?type=uninstall)


</br>
12.Ubuntu源配置：
1. 备份sources.list文件，然后删除
```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bat
```
2. 新建sources.list，将下述参考资料2中的源地址复制到其中：
```bash
# 默认注释了源码仓库，如有需要可自行取消注释
deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.ustc.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```
3. 运行`sudo apt-get update`对源进行更新
> 参考资料：
> 1. [Ubuntu 20.04系统下更改apt源为阿里源 - 知乎](https://zhuanlan.zhihu.com/p/251009600)
> 2. [USTC Open Source Software Mirror](http://mirrors.ustc.edu.cn/)
> 3. [Ubuntu 源使用帮助 — USTC Mirror Help 文档](http://mirrors.ustc.edu.cn/help/ubuntu.html)

</br>
13.修改用户名和密码
1. 使用`wsl -u root`登录root用户
2. 执行如下命令：
```bash
 usermod -l <newname> -d /home/<newname> -m <oldname>
 usermod -c "newfullname" <newname>
 groupmod -n <newgroup> <oldgroup>
```
> 参考资料：
> 1. [permissions - How do I change my username? - Ask Ubuntu](https://askubuntu.com/questions/34074/how-do-i-change-my-username)
> 2. [Linux usermod user is currently used by process - Stack Overflow](https://stackoverflow.com/questions/28972503/linux-usermod-user-is-currently-used-by-process)
> 3. [linux - How to set default user for manually installed WSL distro? - Super User](https://superuser.com/questions/1566022/how-to-set-default-user-for-manually-installed-wsl-distro)

</br>
14.查看文件和文件夹大小：
- 查看文件大小：`ls -l filename`
- 查看文件夹大小：`du -sh folder`
- 查看磁盘使用情况：`df -h`

> 参考资料：
> 1. [Ubuntu查看文件大小或文件夹大小_jackliang的博客-CSDN博客_ubuntu 文件大小](https://blog.csdn.net/xiqingchun/article/details/42466267)


</br>
15.卸载java
```bash
dpkg-query -W -f='${binary:Package}\n' | grep -E -e '^(ia32-)?(sun|oracle)-java' -e '^openjdk-' -e '^icedtea' -e '^(default|gcj)-j(re|dk)' -e '^gcj-(.*)-j(re|dk)' -e '^java-common' | xargs sudo apt-get -y remove

sudo apt-get -y autoremove
```
> 参考资料：
> 1. [How to completely uninstall Java? - Ask Ubuntu](https://askubuntu.com/questions/84483/how-to-completely-uninstall-java#)

</br>
16.改造`rm`命令，删除文件至回收站：

> 参考资料：
> 1. [linux - 改造rm命令，删除文件至回收站 - 不足 - SegmentFault 思否](https://segmentfault.com/a/1190000018464527)
> 2. [GitHub - andreafrancia/trash-cli: Command line interface to the freedesktop.org trashcan.](https://github.com/andreafrancia/trash-cli)

</br>
17.定期清空某个文件夹

> 参考资料：
> 1. [linux定时删除N天前的文件（文件夹） - 腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1849092)
> 2. [command line - Removing files older than 7 days - Ask Ubuntu](https://askubuntu.com/questions/589210/removing-files-older-than-7-days)

</br>
18.Ubuntu配置ssh免输密码：
- `ssh-keygen`：在本地机器上生成密钥对，`id_rsa.pub`和`id_rsa`。在`~/.ssh/`目录下。更改`id_rsa` 私钥权限，`chmod 600 id_rsa`。
- `ssh-copy-id -i ~/.ssh/id_rsa.pub user@host`：将本地公钥`id_rsa.pub`写入远程host的`~/.ssh`目录下的`authorized_keys`文件。如果远程host没有`.ssh`目录手动进行创建。更改`authorized_keys`文件权限，`chmod 755 authorized_keys`。
- 尝试本地登录，如果无法免密码登录，更高远程host上`/home/user`目录权限，`chmod 700 /home/user`。
> 参考资料：
> 1. [Getting Started With SSH in Linux](https://linuxhandbook.com/ssh-basics/)
> 2. [How to Add SSH Public Key to Server](https://linuxhandbook.com/add-ssh-public-key-to-server/)
> 3. [Why am I still getting a password prompt with ssh with public key authentication? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/36540/why-am-i-still-getting-a-password-prompt-with-ssh-with-public-key-authentication)
> 4. [ssh-copy-id succeeded, but still prompt password input - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/407394/ssh-copy-id-succeeded-but-still-prompt-password-input)
> 5. [VSCode远程连接服务器 免密登录（ssh key） | 烟雨平生](https://i007it.com/2022/07/14/VSCode%E8%BF%9C%E7%A8%8B%E8%BF%9E%E6%8E%A5%E6%9C%8D%E5%8A%A1%E5%99%A8-%E5%85%8D%E5%AF%86%E7%99%BB%E5%BD%95/)

</br>
19.vnc viewer和vnc server搭配使用。
- 先使用`vncserver`在远程服务器上生成对应端口号
- 在本地机器上建立`host:port`的连接
> 参考资料：
> 1. [linux中如何开启vnc服务端口,Linux下vnc配置及启动_听亭亭的博客-CSDN博客](https://blog.csdn.net/weixin_30125993/article/details/116636925?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-116636925-blog-107807058.pc_relevant_3mothn_strategy_recovery&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

</br>
20.使用`rm`删除文件时排除某个文件：

> 参考资料：
> 1. [Remove all files/directories except for one file](https://unix.stackexchange.com/questions/153862/remove-all-files-directories-except-for-one-file)

</br>
21.Ubuntu复制命令行输出到剪切板的工具。
工具：xclip
安装过程：
- 从参考资料4下载并解压得到xclip文件夹
- 进入xclip文件夹，运行`./configure --prefix=/storage/FT/.local`指定安装文件夹。
- `make`进行编译
- 通过`su`切换到root用户
- `make install`和`make install.man`安装xclip和man page，分别安装在`/storage/FT/.local/bin`和`/storage/FT/.local/man`目录下。
- 编辑`.bashrc`将该路径加入到PATH中：
```bash
export PATH="/storage/FT/.local/bin:$PATH"
```
- 编辑`.bashrc`添加alias：
```bash
alias clip='xclip -se c'
```
- 此时即可通过`pwd | clip`复制当前路径。
PS：但在VSCode终端中会出现如下报错。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230418225219.png)

> 参考资料 ：
> 1. [software recommendation - A command-line clipboard copy and paste utility? - Ask Ubuntu](https://askubuntu.com/questions/11925/a-command-line-clipboard-copy-and-paste-utility)
> 2. [xclip/INSTALL at master · milki/xclip · GitHub](https://github.com/milki/xclip/blob/master/INSTALL)
> 3. [How to install xclip on Ubuntu](https://howtoinstall.co/en/xclip)
> 4. [xclip download | SourceForge.net](https://sourceforge.net/projects/xclip/)
> 5. [linux - Make install, but not to default directories? - Stack Overflow](https://stackoverflow.com/questions/3239343/make-install-but-not-to-default-directories)

</br>
22.VSCode中使用bash连接远程服务器时，运行上述`pwd | clip`出现如下报错（MobaXterm中不报错）：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230418225219.png)
根据参考资料4中下述描述：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230419002605.png)
在MobaXterm中`$DISPLAY`生效，但在VSCode中打开的bash终端中`$DISPLAY`并未生效。只需：
- 在MobaXterm中通过`echo $DISPLAY`查看`$DISPLAY`环境变量的值。(若没有，通过`cat /etc/resolv.conf`查看`nameserver`的值)
- 在`.bashrc`中添加：
```bash
export DISPLAY='localhost:29.0'
```
- `source ~/.bashrc`使之生效。
> 参考资料：
> 1. [xclip fails with Error: Can't open display: (null) · Issue #4933 · microsoft/WSL · GitHub](https://github.com/microsoft/WSL/issues/4933)
> 2. [How do I fix a "cannot open display" error when opening an X program after ssh'ing with X11 forwarding enabled? - Super User](https://superuser.com/questions/310197/how-do-i-fix-a-cannot-open-display-error-when-opening-an-x-program-after-sshi)
> 3. [linux - Error: Can't open display: (null) when using Xclip to copy ssh public key - Stack Overflow](https://stackoverflow.com/questions/18695934/error-cant-open-display-null-when-using-xclip-to-copy-ssh-public-key)
> 4. [linux - Error: Can't open display: (null) when using Xclip to copy ssh public key - Stack Overflow](https://stackoverflow.com/questions/18695934/error-cant-open-display-null-when-using-xclip-to-copy-ssh-public-key)


</br>
22.Ubuntu系统复制粘贴：
VSCode终端窗口中可以使用`Ctrl+Shift+C/V`进行复制粘贴；
MobaXterm可以使用`Ctrl+C`进行复制，鼠标右键进行粘贴。

</br>
23.Ubuntu从源码编译安装，安装到指定文件夹：
```bash
./configure --prefix=/storage/FT/.local
```
> 参考资料：
> 1. [linux - Make install, but not to default directories? - Stack Overflow](https://stackoverflow.com/questions/3239343/make-install-but-not-to-default-directories)

</br>
24.`tar`解压文件夹：

> 参考资料：
> 1. [Linux tar 命令 | 菜鸟教程](https://www.runoob.com/linux/linux-comm-tar.html)

</br>
25.设置alias
> 参考资料：
> 1. [Bash alias with piping - Super User](https://superuser.com/questions/407104/bash-alias-with-piping)

</br>
26.设置时区。·
`ls -lh`和`date`命令显示的时间不同。
> 参考资料：
> 1. [linux - ls and date showing different file dates - Super User](https://superuser.com/questions/908157/ls-and-date-showing-different-file-dates)
> 2. [How to set or change timezone in linux](https://linuxize.com/post/how-to-set-or-change-timezone-in-linux/)
> 3. [关于Linux中ls -l显示时间不全的问题_nui111的博客-CSDN博客](https://blog.csdn.net/nui111/article/details/42275481)

</br>
27.使用`ln -s`将`dotfiles`仓库中的文件链接到`~`。
```bash
ln -s /mnt/d/Desktop/dotfiles/wsl/.zshrc ~/.zshrc
ln -s /mnt/d/Desktop/dotfiles/wsl/.bash_aliases ~/.bash_aliases
ln -s /mnt/d/Desktop/dotfiles/wsl/.bash_path ~/.bash_path
ln -s /mnt/d/Desktop/dotfiles/wsl/.bashrc ~/.bashrc
ln -s /mnt/d/Desktop/dotfiles/wsl/.vimrc ~/.vimrc
ln -s /mnt/d/Desktop/dotfiles/wsl/.vim/colors/monokai.vim ~/.vim/colors/monokai.vim
```
</br>
28.Linux的内存统计：

> 参考资料：
> 1. [聊聊 Linux 的内存统计 | 浅墨的部落格](https://www.0xffffff.org/2019/07/17/42-linux-memory-monitor/) 






