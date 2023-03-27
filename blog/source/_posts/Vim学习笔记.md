---
title: Vim学习笔记
date: 2022-11-20 13:05:32
categories:
- 工具
tags:
- Vim
---
本文记录一下在学习Vim过程中的经验和总结：
<!--more-->

### Vim删除查找匹配的行
例如，要全局替换掉包含`file:`的行：
```vim
:g/file:/d
```
> 参考资料：
> 1. [vim 删除匹配行_中国风2012的博客-CSDN博客_vim 删除匹配行](https://blog.csdn.net/hanshileiai/article/details/50528505)

### Vim复制、粘贴然后替换
以下面代码为例：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191509.png)
我们的目的是将`pass`替换为`if __name__ == "__main__"`。步骤如下：
1.在visual模式下选择`if __name__ == "__main__"`，然后在normal模式下按下`y`复制。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191750.png)
2.在visual模式下选择`pass`，按下`y`即可进行替换：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191812.png)
便可得到下述结果：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230327191848.png)
> 参考资料：
> 1. [Copy, delete, then paste in Vim - Super User](https://superuser.com/questions/371160/copy-delete-then-paste-in-vim)