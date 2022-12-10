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
`ps ef | grep python | awk '{print $2} | xargs -r kill -9'`
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