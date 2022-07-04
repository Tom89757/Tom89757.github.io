---
title: python中知识点01
date: 2022-07-05 01:25:06
categories:
- 深度学习
tags:
- 笔记
- python
---

本文记录一下python中常用的知识点：

<!--more-->

1.在Python shell中清屏。在Windows和Linux中的实现原理均为调用os.system提供的清屏函数。

- Windows：

  ```python
  import os
  clear = lambda: os.system('cls')
  ```

  然后调用`clear()`即可。也可以直接运行`os.system('cls')`。

- Linux：

  ```python
  import os
  clear = lambda: os.system('clear')
  ```

  然后调用`clear()`即可。也可以直接运行`os.system('clear')`。

为了避免每次启动Python shell后都需要重新定义`clear`函数。可以设置如下脚本：

```python
import os
if os.name == 'nt':
	def cls():
		_ = os.system('cls')
else:
	def clear():
		_ = os.system('clear')
```

并使得上述脚本在启动Python shell时自动运行。

> 参考资料：
>
> 1. [How to clear Python shell?](https://www.tutorialspoint.com/how-to-clear-python-shell#)

</br>

2.设置在Python shell启动时自动执行某个脚本。下面记录在Ubuntu系统中的设置，Windows系统同理：

- 创建脚本并写入想要执行的命令，例如创建`startup.py`并写入：

  ```python
  import os
  if os.name == 'nt':
  	def cls():
  		_ = os.system('cls')
  else:
  	def clear():
  		_ = os.system('clear')
  ```

  该脚本在打开Python shell中创建了函数`clear()`，可以通过执行`clear()`实现清屏。

- 在当前Ubuntu系统中使得环境变量`PYTHONSTARTUP`指向上述脚本对应的路径，假设上述脚本路径为`/home/FT/scripts/startup.py`。则可以通过在bash中执行

  `export PYTHONSTARTUP=/home/FT/scripts/startup.py`

  创建临时环境变量，也可以在`~/.bashrc`中添加该命令使其永久化。

- 此时在bash中打开python shell（包括Anaconda中的python shell），可直接通过`clear()`实现清屏。

> 参考资料：
>
> 1. [PYTHONSTARTUP](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONSTARTUP)

</br>
