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
`ps -ef | grep FT | awk '{print $2}' | xargs -r kill -9'`
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
6.sed指令匹配模式，并替换模式中的一部分，保留剩余部分：
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

</br>
13.修改用户名和密码

> 参考资料：
> 1. [修改Ubuntu用户名及其密码、主机名、主目录名 - 直木 - 博客园](https://www.cnblogs.com/yxqxx/p/12319130.html)

</br>
14.查看文件和文件夹大小：
- 查看文件大小：`ls -l filename`
- 查看文件夹大小：`du -sh folder`
- 查看磁盘使用情况：`df -h`

> 参考资料：
> 1. [Ubuntu查看文件大小或文件夹大小_jackliang的博客-CSDN博客_ubuntu 文件大小](https://blog.csdn.net/xiqingchun/article/details/42466267)