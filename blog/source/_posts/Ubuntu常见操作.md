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











