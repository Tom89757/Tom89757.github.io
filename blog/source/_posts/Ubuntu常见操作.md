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
解压命令大全见：[linux下解压命令大全](https://www.cnblogs.com/eoiioe/archive/2008/09/20/1294681.html)

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

也可用于批量给文件名添加前缀或后缀，例如给多个文件批量添加`.cpp`后缀或`1`前缀：

- `helloworld -> helloworld.cpp`：`rename -v 's/$/.cpp/' *`。
- `helloworld.cpp -> 1helloworld.cpp`：`rename -v 's/^/1/' *`。

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

8.`ls | sed 's/.jpg//'   `：用于对`ls`的输出做处理。

- 处理之前：

  ```python
  FT@node2$ ls | head
  ILSVRC2012_test_00000004.jpg
  ILSVRC2012_test_00000018.jpg
  ILSVRC2012_test_00000019.jpg
  ILSVRC2012_test_00000022.jpg
  ILSVRC2012_test_00000030.jpg
  ILSVRC2012_test_00000072.jpg
  ILSVRC2012_test_00000082.jpg
  ILSVRC2012_test_00000108.jpg
  ILSVRC2012_test_00000130.jpg
  ILSVRC2012_test_00000172.jpg
  ```

- 处理之后：

  ```python
  FT@node2$ ls | head | sed 's/.jpg//'
  ILSVRC2012_test_00000004
  ILSVRC2012_test_00000018
  ILSVRC2012_test_00000019
  ILSVRC2012_test_00000022
  ILSVRC2012_test_00000030
  ILSVRC2012_test_00000072
  ILSVRC2012_test_00000082
  ILSVRC2012_test_00000108
  ILSVRC2012_test_00000130
  ILSVRC2012_test_00000172
  ```

</br>

9.在`sed`命令中包含斜杆的替换。一般情况下，我们会用`/`作为`sed`指令的分隔符进行字符串的查找替换。例如：

```python
cat test.txt | sed 's/.jpg/.png/' > test1.txt
```

可以将`.jpg`字符串替换为`.png`。但当要查找或替换的字符串包含`/`本身时，该方式无法使用。此时可以利用`sed`命令会将紧跟在`s`后面的字符作为分隔符的特性，将分隔符改为`#`或其他字符。例如：

```python
cat test.txt | sed 's#^#data/#' > test2.txt
```

可以在每一行的开头添加`data/`字符串。
- 在每一行开头添加`'`单引号字符，`ls | sed "s/^/'/"`
- 在每一行末尾添加`',`字符，`ls | sed "s/$/',/"`。
完整使用`ls | sed "s/^/'/" | sed "s/$/',/"`效果如下：
```
'ILSVRC2012_test_00000003.png',
'ILSVRC2012_test_00000023.png',
'ILSVRC2012_test_00000025.png',
'ILSVRC2012_test_00000026.png',
'ILSVRC2012_test_00000034.png',
'ILSVRC2012_test_00000038.png',
'ILSVRC2012_test_00000064.png',
'ILSVRC2012_test_00000086.png',
'ILSVRC2012_test_00000105.png',
'ILSVRC2012_test_00000128.png',
```

> 参考资料：
>
> 1. [Sed替换 内容带反斜杠（/）](https://blog.csdn.net/weixin_39031707/article/details/104065184)

</br>

10.有时候可能会许村将两个文件中的数据合并到一个文件，并且数据分处第一列和第二列，此时可以通过`paste`命令实现。例如，文件`train.txt`和`train1.txt`分别包含以下内容：

```python
$ cat train.txt
1.jpg
2.jpg
3.jpg
4.jpg
$ cat train1.txt
1.jpg
2.jpg
3.jpg
4.jpg
```

通过`paste train.txt train1.txt`命令可以获得以下输出：

```python
1.jpg   1.jpg
2.jpg   2.jpg
3.jpg   3.jpg
4.jpg   4.jpg
```

> 参考资料：
>
> 1. [Linux下paste命令，按列合并文件](https://blog.csdn.net/jiao_zhoucy/article/details/20693179)

</br>

11.可以通过以下脚本批量移动文件：

```bash
for dir in DUT-OMRON DUTS_Test ECSSD HKU-IS PASCAL-S SOD THUR15K
do
    echo "processing $dir data"
    if [ -d $dir ]; then
        echo
    else
        mkdir $dir
    fi

    image=$dir/image
    echo "image is $image"
    if [ -d $image ]; then
        echo
    else
        mkdir $image
    fi
    
    mask=$dir/mask
    echo "mask is $mask"
    if [ -d $mask ]; then
        echo
    else
        mkdir $mask
    fi
    
    mv ./img/$dir/* $image
    mv ./gt/$dir/* $mask
done
```

先通过`[-d $dir]`确定是否存在`$dir`文件夹，如果没有则创建，然后将`img`和`gt`文件夹下的图片移动到对应的文件夹。

> 参考资料：
>
> 1. 《Linux命令行和shell脚本编程》第12章

</br>

12.在Linux系统中可以很方便地对两个文本文件求取交集、并集和差集

- 交集：`sort a.txt b.txt | uniq -d`
- 并集：`sort a.txt b.txt | uniq`
- 差集：
  - a.txt - b.txt：`sort a.txt b.txt b.txt | uniq -u`
  - b.txt - a.txt：`sort b.txt a.txt a.txt | uniq -u`

> 参考资料：
>
> 1. [Linux两个文件求交集、并集、差集](https://www.cnblogs.com/molong1208/p/5358509.html)

</br>

13.结合[11]和[12]可以方便地求取image和mask文件夹中图片的交集并输出到对应的txt文件：

```bash
for dir in DUT-OMRON DUTS_Test ECSSD HKU-IS PASCAL-S SOD THUR15K
do
    echo "processing $dir data"
    image=$dir/image
    mask=$dir/mask
    ls $image | sed 's/.jpg//' | sed 's/.png//' >$dir/image.txt
    ls $mask | sed 's/.png//' | sed 's/.bmp//' >$dir/mask.txt

    sort $dir/image.txt $dir/mask.txt | uniq -d >$dir/test.txt
    cat $dir/test.txt | wc
done
```

</br>

14.`sh` vs `bash`：注意，二者并不等价。bash是sh的超集。

- 通过`sh test7.sh`运行脚本，会出现如下报错：

  ![image-20220716172241725](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220716172241725.png)

- 通过`bash test7.sh`运行脚本，可以正常运行：

  ![image-20220716172342306](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220716172342306.png)

> 参考资料：
>
> 1. [Difference between sh and bash](https://www.geeksforgeeks.org/difference-between-sh-and-bash/)
> 2. [Shell script fails: Syntax error: "(" unexpected](https://unix.stackexchange.com/questions/45781/shell-script-fails-syntax-error-unexpected)

</br>

15.在bash脚本中将命令行的输出复制给一个变量：

```bash
#!/bin/bash
output=$(cat batchDir.sh | wc)
echo "${output}"
# 或者
echo $output
```

其输出如下：

![image-20220716172942136](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220716172942136.png)

> 参考资料：
>
> 1. [How do I set a variable to the output of a command in Bash?](https://stackoverflow.com/questions/4651437/how-do-i-set-a-variable-to-the-output-of-a-command-in-bash)

</br>

16.在Anaconda中创建环境时`-n/--name`不能和`-p/--prefix`同时使用：

- `-n, --name`：环境名，如`conda create --name py35 python=3.5`。
- `-p, --prefix`：环境位置的完整路径。如`conda create --prefix /users/.../yourenvname python=2.7`。

> 参考资料：
>
> 1. [conda create](https://docs.conda.io/projects/conda/en/latest/commands/create.html#conda-create)
> 2. [how to specify new environment location for conda create](https://stackoverflow.com/questions/37926940/how-to-specify-new-environment-location-for-conda-create)

</br>

17.删除conda环境：`conda env remove -n ENV_NAME`。

</br>

18.在编写bash脚本时，指定`IFS`变量为多个分隔符：`IFS=', | \\'`。如下图所示：

```python
#!/bin/bash
echo "reading values from a file"
file="list"
# change seperator
# save IFS before changing it
IFS_OLD=$IFS
# set multiple seperators?
IFS='; | :'
for state in $(cat $file)
do
    echo "Visit $state"
done
IFS=${IFS_OLD}

```

> 参考资料：
>
> 1. [Split string using 2 different delimiters in Bash](https://stackoverflow.com/questions/25163486/split-string-using-2-different-delimiters-in-bash)

</br>

19.在Ubuntu终端常使用Ctrl + L组合快捷键清空终端屏幕，但向上滚动时屏幕内容依然存在。此时可以通过

- `printf '\ec'`
- `reset`

二者来完全清空屏幕，但`reset`为彻底清除，执行速度较慢。故可以通过在`~/.bashrc`中添加如下别名：

```bash
alias cls='printf "\ec"'
```

来使用`cls`来清空终端屏幕。

> 参考资料：
>
> 1. [Linux终端彻底清空屏幕](https://blog.csdn.net/pngynghay/article/details/23176757)

</br>

20.在使用`ls`命令时，有时我们仅需要列出目录，然后将目录导入txt文件。下面是3种不同的方法：

1. `ls -d */`：其输出如下：

   ```bash
   DUT-OMRON/  DUTS_Test/  ECSSD/  HKU-IS/  PASCAL-S/  SOD/  THUR15K/
   ```

   可以通过`ls -d */ | sed 's#/##' > dir.txt`将数据集名称导出到`dir.txt`

2. `ls -F | grep "/$"`。`-F`会在输出的不同文件类型后面加上后缀，文件后会加上`*`，管道后会加上`|`，目录后会加上`/`。

3. `ls -l | grep "^d"`。使用`grep`匹配输出每行开头的`d`字符。可以通过`awk`命令列出目录名本身：

   ```bash
   ls -l | grep "^d" | awk '{print $8}'
   ```

> 参考资料：
>
> 1. [Linux Shell 只列出目录的方法](https://blog.csdn.net/DLUTBruceZhang/article/details/9244897)

</br>

21.在Ubuntu系统中，常常需要查看当前运行的进程，并根据需要筛选。下面介绍`ps`命令的选项：

- `ps -A`：列出所有的进程

- `ps -w`：显示加宽可以显示更多的信息

- `ps -au`：显示较为详细的信息

- `ps -aux`：显示所有包含其他使用者的进程。下面是对该命令输出信息的介绍，其输出格式为

  ```bash
  USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
  ```

  - `USER`：进程拥有者
  - `PID`：process id
  - `%CPU`：CPU占用率
  - `VSZ`：虚拟内存大小
  - `RSS`：内存大小
  - `TTY`：minor device number of tty
  - `STAT`：进程状态
    - D：无法终端的休眠状态（通常为IO的进程）
    - R：正在执行中
    - S：静止状态
    - T：暂停执行
    - Z：不存在但暂时无法消除
    - W：没有足够的内存分页可分配
    - <：高优先序的进程
    - N：低优先序的进程
    - L：有内存分页分配且Lock在内存内

  - START：进程开始时间
  - TIME：执行的时间
  - COMMAND：执行的指令

较为常用的命令为`ps -u FT`，其可以显示用户`FT`正在运行的进程，并可以通过`grep`对进程进行筛选。

此外，`top`命令可以显示正在运行的进程并实时更新。

> 参考资料：
>
> 1. [Linux ps 命令](https://www.runoob.com/linux/linux-comm-ps.html)
> 2. [ubuntu查看所有正在运行的进程](https://blog.csdn.net/yaoqiuxiang/article/details/9449179)

</br>

22.`kill`命令常用于通过进程号PID终止特定进程。可以用`kill -l`命令列出所有可用的信号，最常用的信号是：

- `1 (HUP)`：重新加载进程
- `9 (KILL)`：杀死一个进程
- `15 (TERM)`：正常停止一个进程

示例：

`kill -9 12345`：彻底杀死进程号为12345的进程。

> 参考资料：
>
> 1. [Linux kill命令](https://www.runoob.com/linux/linux-comm-kill.html)

</br>

23.服务器远程登录配置：

> 参考资料：
>
> 1. [腾讯云 Permission denied (publickey,gssapi-keyex,gssapi-with-mic)](https://www.chengxiaobai.cn/record/tencent-cloud-denied-permission-publickey-gssapikeyex-gssapiwithmic.html)
> 2. [Linux 配置SSH免密登录 “ssh-keygen”的基本用法](https://cloud.tencent.com/developer/article/1720991)
> 3. [设置 SSH 通过密钥登录](https://www.runoob.com/w3cnote/set-ssh-login-key.html)

</br>

24.使用`chmod`变更文件或目录权限。可以使用`chmod -R 777 ./folder`对目录以及目录一下的文件递归执行更改权限操作。

> 参考资料：
>
> 1. [chmod](https://wangchujiang.com/linux-command/c/chmod.html)

</br>

25.查看Linux系统内核版本和系统架构：
- `hostnamectl`：查看内核版本和系统架构
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220913164139.png)
- `uname -a`：查看内核版本和系统架构
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20220913164340.png)
> 参考资料：
> 1. [查看Linux系统架构类型的5条常用命令](https://www.sysgeek.cn/find-out-linux-system-32-or-64-bit/)
> 2. [查看Linux内核版本](https://www.cnblogs.com/linuxprobe/p/11664104.html)

</br>
26.查看Linux空间使用情况：
- `df -lh`：查看分区使用情况
- `du -sh /home/FT`：查看当前用户使用的存储空间大小
> 参考资料：
> 1. [Linux笔记』查看磁盘空间大小和所有用户各自占用空间](https://blog.csdn.net/abc13526222160/article/details/84962310#:~:text=Linux%E4%B8%8B%E6%9F%A5%E7%9C%8B%E6%96%87%E4%BB%B6%E5%8D%A0,%E5%91%BD%E4%BB%A4%E6%9C%89%E4%B8%80%E4%BA%9B%E5%8C%BA%E5%88%AB%E7%9A%84%E3%80%82)
> 2. [Linux 查看磁盘空间](https://www.runoob.com/w3cnote/linux-view-disk-space.html)

</br>
27.Ubuntu添加和删除用户
> 参考资料：
> 1. [如何在Ubuntu添加和删除用户](https://www.myfreax.com/how-to-add-and-delete-users-on-ubuntu-18-04/#:~:text=%E9%80%9A%E8%BF%87GUI%E6%B7%BB%E5%8A%A0%E6%96%B0%E7%94%A8%E6%88%B7&text=%E6%89%93%E5%BC%80Ubuntu%E8%AE%BE%E7%BD%AE%EF%BC%8C%E6%89%BE%E5%88%B0%20%E7%94%A8%E6%88%B7,%E5%91%98%E7%94%A8%E6%88%B7%E5%B9%B6%E8%BE%93%E5%85%A5%E4%BF%A1%E6%81%AF%E3%80%82)

</br>
28.配置环境变量
- `export PATH=/usr/local/bin:$PATH`：将环境变量放在环境变量检索目录最开始，即优先查找该变量
- `export PATH=$PATH:/usr/local/bin`：将环境变量放在环境变量检索目录最后，即最后查找该变量
> 1. [Multiple CUDA versions on machine nvcc -V confusion](https://stackoverflow.com/questions/40517083/multiple-cuda-versions-on-machine-nvcc-v-confusion)
> 2. [How to correctly add a path to PATH?](https://unix.stackexchange.com/questions/26047/how-to-correctly-add-a-path-to-path)

</br>

29.复制指定目录下的全部文件到另一个目录中：
```bash
cp -r dir1 dir2
```
> 参考资料：
> 1. [linux复制指定目录下的全部文件到另一个目录中，linux cp 文件夹](https://www.cnblogs.com/zdz8207/p/linux-cp-dir.html)

</br>
30.rename批量修改文件名：

> 参考资料：
> 1. [每天学习一个命令: rename 批量修改文件名 | Verne in GitHub](https://einverne.github.io/post/2018/01/rename-files-batch.html)













