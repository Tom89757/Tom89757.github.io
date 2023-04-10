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

2.设置在Python shell启动时自动执行某个脚本。

下面记录在Ubuntu系统中的设置：

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

下面记录在Windows系统中的设置：

- 创建脚本并写入想要执行的命令，例如上述的`startup.py`。

- 添加用户环境变量`PYTHONSTARTUP`：

  ![image-20220707103000076](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220707103000076.png)

- 打开cmd，运行`refreshenv`（需安装Chocolatey）更新环境变量。
- 运行`python`，会发现`startup.py`中的命令已生效。

> 参考资料：
>
> 1. [PYTHONSTARTUP](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONSTARTUP)
> 1. [win10 python3.5 自动补全设置](https://www.cnblogs.com/zkwarrior/p/9374302.html)

</br>

3.Python中的metaclass。

**类也是对象**
Python关于类是什么这个问题有古怪的答案，它借鉴了来自Smalltalk语言的设计。

在大多数语言中，类只是描述如何生成一个对象的代码段。在Python中差不多也是这样的：

```python
>>> class ObjectCreator(object):
...       pass
...

>>> my_object = ObjectCreator()
>>> print(my_object)
<__main__.ObjectCreator object at 0x8974f2c>

```

但是类在Python中意味着更多。类本身也是对象。

只要你使用关键字`class`，Python就会执行它并创建一个对象。以下的指令：

```python
>>> class ObjectCreator(object):
...       pass
...
```

在内存中创建了一个名为`ObjectCreator`的对象。

这个对象（即这个类）它自身具有可以创建对象（即实例）的能力，这就是为什么它是一个类。

但是，它仍然是一个对象，因此：

- 你可以将它赋值给一个变量
- 你可以打印它
- 你可以复制它
- 你可以给它添加属性
- 你可以把它作为函数参数传递

例如：

```python
>>> print(ObjectCreator) # you can print a class because it's an object
<class '__main__.ObjectCreator'>
>>> def echo(o):
...       print(o)
...
>>> echo(ObjectCreator) # you can pass a class as a parameter
<class '__main__.ObjectCreator'>
>>> print(hasattr(ObjectCreator, 'new_attribute'))
False
>>> ObjectCreator.new_attribute = 'foo' # you can add attributes to a class
>>> print(hasattr(ObjectCreator, 'new_attribute'))
True
>>> print(ObjectCreator.new_attribute)
foo
>>> ObjectCreatorMirror = ObjectCreator # you can assign a class to a variable
>>> print(ObjectCreatorMirror.new_attribute)
foo
>>> print(ObjectCreatorMirror())
<__main__.ObjectCreator object at 0x8997b4c>
```

**动态地创建类**

因为类是对象，所以你可以像任何对象那样自由地创建他们。

首先，你可以在一个函数中使用`class`创建一个类：

```python
>>> def choose_class(name):
...     if name == 'foo':
...         class Foo(object):
...             pass
...         return Foo # return the class, not an instance
...     else:
...         class Bar(object):
...             pass
...         return Bar
...
>>> MyClass = choose_class('foo')
>>> print(MyClass) # the function returns a class, not an instance
<class '__main__.Foo'>
>>> print(MyClass()) # you can create an object from this class
<__main__.Foo object at 0x89c6d4c>
```

但是它不够动态，因为你仍然需要自己写整个类。

因为类是对象，它们必须被某个东西生成。

当你使用`class`关键字的时候，Python自动地创建了这个对象。但是和在Python中的大多数东西一样，它给了你手动来做这件事的方法。

记得函数`type`吗？这个好的古老的函数让你知道一个对象是什么类型。

```python
>>> print(type(1))
<type 'int'>
>>> print(type("1"))
<type 'str'>
>>> print(type(ObjectCreator))
<type 'type'>
>>> print(type(ObjectCreator()))
<class '__main__.ObjectCreator'>
```

其实，`type`还有一个完全不同的能力，它也能够自由地创建类。`type`可以将一个类的描述作为参数，然后返回一个类。

（注：虽然相同的函数根据传给它参数的不同有两种完全不同的用途看起来很蠢。但是`type`的这种情况是由于Python向后兼容所导致的问题）

`type`以下面的方式起作用：

```python
type(name, bases, attrs)
```

其中：

- `name`：类名
- `bases`：父类的元组（用于继承，可以为空）
- `attr`：字典，包含属性名和属性值

例如：

```python
>>> class MyShinyClass(object):
...       pass
```

可以通过以下的方式手动创建：

```python
>>> MyShinyClass = type('MyShinyClass', (), {}) # returns a class object
>>> print(MyShinyClass)
<class '__main__.MyShinyClass'>
>>> print(MyShinyClass()) # create an instance with the class
<__main__.MyShinyClass object at 0x8997cec>
```

可以注意到我们同时使用`MyShinyClass`作为类名并且作为变量来做类引用。它们可以不同，但是没有必要使问题复杂化。

`type`接受一个字典来定义类的属性。所以：

```python
>>> class Foo(object):
...       bar = True
```

也能被翻译为：

```python
>>> Foo = type('Foo', (), {'bar':True})
```

并且作为一个普通类使用：

```python
>>> print(Foo)
<class '__main__.Foo'>
>>> print(Foo.bar)
True
>>> f = Foo()
>>> print(f)
<__main__.Foo object at 0x8a9b84c>
>>> print(f.bar)
True
```

当然，你也可以从`Foo`继承它，所以：

```python
>>>   class FooChild(Foo):
...         pass
```

将会是：

```python
>>> FooChild = type('FooChild', (Foo,), {})
>>> print(FooChild)
<class '__main__.FooChild'>
>>> print(FooChild.bar) # bar is inherited from Foo
True
```

最后，你想向你的类中添加方法。只需定义一个具有合适标识的函数并且将其作为一个属性给它赋值。

```python
>>> def echo_bar(self):
...       print(self.bar)
...
>>> FooChild = type('FooChild', (Foo,), {'echo_bar': echo_bar})
>>> hasattr(Foo, 'echo_bar')
False
>>> hasattr(FooChild, 'echo_bar')
True
>>> my_foo = FooChild()
>>> my_foo.echo_bar()
True
```

> 参考资料：
>
> 1. [What are metaclasses in Python?](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python)

</br>

4.目前`pip`已经不提供`pip search <package name>`的服务，运行后会出现以下报错：

![image-20220717103944297](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220717103944297.png)

此时，可以通过在 [duckduckgo](https://duckduckgo.com/) 搜索 `!pip <package name>`来获得相关的package信息。

> 参考资料：
>
> 1. [How do I search for an available Python package using pip?](https://stackoverflow.com/questions/17373473/how-do-i-search-for-an-available-python-package-using-pip)
> 2. [PyPI XMLRPC search API has been disabled due to flood of requests. `pip search` may be deprecated.](https://www.reddit.com/r/Python/comments/kfxibk/pypi_xmlrpc_search_api_has_been_disabled_due_to/)
> 3. [Remove the pip search command #5216](https://github.com/pypa/pip/issues/5216)

</br>

5.`pip`安装指定版本package：`pip install Package_name==version`。例如：

```python
pip install mmdet==2.12.0
```

> 参考资料：
>
> 1. [pip install specific version](https://www.google.com/search?q=pip+install+specific+version&oq=pip+install+specivi&aqs=chrome.1.69i57j0i512l9.4361j0j7&sourceid=chrome&ie=UTF-8)

</br>

6.Python中两个有用的函数：

- `dir()`：
- `help()`：

> 参考资料：
>
> 1. [Difference between dir() and help()](http://net-informations.com/python/iq/help.htm)

</br>

7.根据字符串常量动态创建以字符串命名的变量：

> 参考资料：
>
> 1. [Python进阶：如何将字符串常量转化为变量？](https://segmentfault.com/a/1190000018534188)

</br>

8.当通过`socket`创建本地服务器时：

```python
import socket

HOST, PORT = '', 8888

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print('Serving HTTP on port %s ...' % PORT)
while True:
    client_connection, client_address = listen_socket.accept()
    request = client_connection.recv(1024)
    print(request.decode("utf-8"))

    http_response = """\
HTTP/1.1 200 OK

Hello, World!
"""
    client_connection.sendall(http_response.encode("utf-8"))
    client_connection.close()
```

运行该脚本可能发生以下情况：

![image-20220727134014236](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727134014236.png)

其原因为本地端口`8888`被其他的应用占用（本例中其被Charles占用），解决方法为使用其他的端口。

> 参考资料：
>
> 1. [Python实现简单的web服务器](https://zhuanlan.zhihu.com/p/35318041)
> 2. [socket.error: [Errno 10013\] An attempt was made to access a socket in a way forbidden by its access permissions](https://stackoverflow.com/questions/2778840/socket-error-errno-10013-an-attempt-was-made-to-access-a-socket-in-a-way-forb)

</br>

8.如下，构建本地代理服务器来请求`https://www.baidu.com`：

```python
# 代理
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

proxy_handler = ProxyHandler({
    'http': 'http://127.0.0.1:9743',
    'https': 'https://127.0.0.1:10000'
})
url = "https://www.baidu.com"
# request = Request(url)
opener = build_opener(proxy_handler)
try:
    response = opener.open(url)
    # response = opener.open(request)
    print(response.read().decode('utf-8'))
except URLError as e:
    print(e.reason)
```

运行后会出现如下情况：

![image-20220727140934265](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727140934265.png)

此时注释掉构建的`https`代理服务器：

```python
# 'https': 'https://127.0.0.1:10000'
```

运行正常：

![image-20220727141031507](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220727141031507.png)

故推测是构建的`https`型代理服务器出现问题，有待探究。

> 参考资料：
>
> 1. [urllib，request 设置代理](https://www.cnblogs.com/huangguifeng/p/7635512.html)
> 2. [Errno 10061 : No connection could be made because the target machine actively refused it ( client - server )](https://stackoverflow.com/questions/12993276/errno-10061-no-connection-could-be-made-because-the-target-machine-actively-re)

</br>
9.下面两段代码的区别：
```python
def max(a, b):
	if a>b:
		return a
	return b
```
```python
def max(a, b):
	if a>b:
		return a
	else:
		return b
```
> 参考资料：
> 1. [It is more efficient to use if-return-return or if-else-return?](https://stackoverflow.com/questions/9191388/it-is-more-efficient-to-use-if-return-return-or-if-else-return)

</br>
10.Python中星号*(asterisk)的各种用法总结
> 参考资料：
> 1. [All you need to know about Asterisks in Python](https://bas.codes/posts/python-asterisks)

</br>
11.python语法：判断字符串中是否包含某子字符串
```python
test_str = 'helloworld'

if 'world' in test_str:
	print('yes')
else:
	print('no')
```
> 参考资料：
> 1. [Python判断字符串是否包含特定子串的7种方法](https://cloud.tencent.com/developer/article/1699719)

</br>
11.格式化字符串：
```python
TAG = 'scwssod'
id = 30
filename = 'mytest_%s_model-%d.log'%(TAG, id)
```

</br>
12.查看函数的类汇编代码：
可以通过python的`dis`库查看函数的类汇编形式的代码，如下图所示：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016154442.png)
同样，可以通过其`ast`库查看代码会被转换为怎样的语法树（python=3.10.6，python3.8输出或打印的语法树没有换行）：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016161921.png)

![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016161739.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016161809.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221016161809.png)
> 参考资料：
> 1. 《代码之髓》3.2节
> 2. [`ast`](https://docs.python.org/3.10/library/ast.html#module-ast "ast: Abstract Syntax Tree classes and manipulation.") — Abstract Syntax Trees

</br>
13.Python按键 (key) 或值 (value) 对字典进行排序
- 按键排序：
```python
# 声明字典
key_value ={}     

# 初始化
key_value[2] = 56       
key_value[1] = 2 
key_value[5] = 12 
key_value[4] = 24
key_value[6] = 18      
key_value[3] = 323 

print ("按键(key)排序:")   

# sorted(key_value) 返回重新排序的列表
# 字典按键排序
for i in sorted (key_value) : 
	print ((i, key_value[i]), end =" ") 
```
- 按值排序：
```python
# 声明字典
key_value ={}     

# 初始化
key_value[2] = 56       
key_value[1] = 2 
key_value[5] = 12 
key_value[4] = 24
key_value[6] = 18      
key_value[3] = 323 


print ("按值(value)排序:")   
print(sorted(key_value.items(), key = lambda kv:(kv[1], kv[0])))    
```
> 参考资料：
> 1. [Python 按键(key)或值(value)对字典进行排序 | 菜鸟教程](https://www.runoob.com/python3/python-sort-dictionaries-by-key-or-value.html)
> 2. [python字典按照key,value进行排序的几种方法_51CTO博客_python 对字典按照value进行排序](https://blog.51cto.com/hzf16321/2721549)
> 3. [python笔记17-字典如何按value排序 - 上海-悠悠 - 博客园](https://www.cnblogs.com/yoyoketang/p/9147052.html)

</br>
14.Python中创建包和导入包的操作
> 参考资料：
> 1. [Python创建包，导入包（入门必读）](http://c.biancheng.net/view/4669.html)
> 2. [Distutils/Tutorial - Python Wiki](https://wiki.python.org/moin/Distutils/Tutorial?highlight=%28setup.py%29)

</br>
15.Python中清空txt文件
```python
with open("test.txt", "a") as file:
	file.truncate(0)
```
> 参考资料：
> 1. [python如何清空txt文件 - 问答 - 亿速云](https://www.yisu.com/ask/6997.html)

</br>
16.pip、setuptools、python环境变量、python命令行参数等
- [pip指南 · January Star](http://chenjiee815.github.io/pipzhi-nan.html)
- [setuptools指南：未完待续 · January Star](http://chenjiee815.github.io/setuptoolszhi-nan-wei-wan-dai-xu.html)
- [Python环境变量 · January Star](http://chenjiee815.github.io/pythonhuan-jing-bian-liang.html)
- [Python命令行参数 · January Star](http://chenjiee815.github.io/pythonming-ling-xing-can-shu.html)
>参考资料：
>1. [All Posts · January Star](http://chenjiee815.github.io/archives.html)

</br>
17.Python中的正则表达式匹配：

> 参考资料：
> 1. [用python正则表达式提取字符串_猪笨是念来过倒的博客-CSDN博客_python正则表达式提取字符串](https://blog.csdn.net/liao392781/article/details/80181088)
> 2. [正则表达式 第三篇：分组和捕获 - 悦光阴 - 博客园](https://www.cnblogs.com/ljhdo/p/10678281.html)

</br>
18.Python中的`next()`函数：
- 描述：`next()`函数返回iterator的下一个项目，主要和生成迭代器的`iter()`一起使用
- 语法：`next(iterable[,defaulat]`
- 参数说明：iterable为可迭代对象，default可选，用于设置在没有下一个元素时返回该默认值，如果不设置又没有下一个元素，会触发`StopIteration`异常
- 返回值：返回下一个item。
- 实例：对于一个pytorch中的`DataLoader`对象，可以如下使用：
```python
from torch.utils.data import Dataset, DataLoader
class myDataset(Dataset):
	...
dataset = myDataset(...)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
iter_loader = iter(loader) # 生成迭代器
next_input, next_target, _ , _ = next(iter_loader) # 访问迭代器下一个item
```
> 参考资料：
> 1. [Python next() 函数 | 菜鸟教程](https://www.runoob.com/python/python-func-next.html)
> 2. [python - What does next() and iter() do in PyTorch's DataLoader() - Stack Overflow](https://stackoverflow.com/questions/62549990/what-does-next-and-iter-do-in-pytorchs-dataloader)
> 3. [torch.utils.data.DataLoader "next" function? - PyTorch Forums](https://discuss.pytorch.org/t/torch-utils-data-dataloader-next-function/87270/2)

</br>
19.python中的`enumerate`的用法：
```python
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']  
>>> list(enumerate(seasons))  
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]  
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始  
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

>>> seq = ['one', 'two', 'three']  
>>> for i, element in enumerate(seq):  
...     print i, element  
...  
0 one  
1 two  
2 three
```
> 参考资料：
> 1. [Python enumerate() 函数 | 菜鸟教程](https://www.runoob.com/python/python-func-enumerate.html)

</br>
20.安装`pip install albumentations`时报错：
```bash
Could not install packages due to an OSError: [WinError 5]
```
解决方案：添加`--user`选项
```bash
pip3 install --upgrade albumentations --user
```
> 参考资料：
> 1. [tensorflow - Could not install packages due to an EnvironmentError: [WinError 5] Access is denied: - Stack Overflow](https://stackoverflow.com/questions/51912999/could-not-install-packages-due-to-an-environmenterror-winerror-5-access-is-de)

</br>
21.`pip install package`时出现如下Warning:
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230217212259.png)
原因：对应包损坏
解决方案：在对应路径下`d:\ml\anaconda3\envs\testenv\lib\site-packages`找到名字前缀为`~ip`的文件夹并删除。
> 参考资料：
> 1. [Found Solution to: WARNING: Ignoring invalid distribution -{packageName} ({pathToIssue}) : Python](https://www.reddit.com/r/Python/comments/x70edr/found_solution_to_warning_ignoring_invalid/)

</br>
22.pip更新：
```bash
python -m pip install --upgrade pip
```
> 参考资料：
> 1. [python - There was an error checking the latest version of pip - Stack Overflow](https://stackoverflow.com/questions/72439001/there-was-an-error-checking-the-latest-version-of-pip)

</br>
23.python中`import`相关的路径问题：

> 参考资料：
> 1. [Relative imports in Python 3 - Stack Overflow](https://stackoverflow.com/questions/16981921/relative-imports-in-python-3)
> 2. [python - __init__.py can't find local modules - Stack Overflow](https://stackoverflow.com/questions/34753206/init-py-cant-find-local-modules)
> 3. [【python】关于import你需要知道的一切！一个视频足够了_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1K24y1k7XA/?spm_id_from=333.999.0.0&vd_source=71b57f2bb132ac1f88ed255cad4a06a6)

</br>
24.导出pip list到`requirements.txt`文件在另一个环境中安装：
```bash
## 导出
pip freeze >requirements.txt
## 安装
pip install -r requirements.txt
```
> 参考资料：
> 1. [pip requirements导出当前项目所用的包list列表_苦咖啡's运维之路的技术博客_51CTO博客](https://blog.51cto.com/alsww/1893100)

</br>
15.在Python中只声明变量而不赋值：
```python 
result = None
for i in range(10):
	if i == 0:
		result = i 
	else:
		result += i
```
> 参考资料：
> 1. [在Python中是否可以只声明变量而不赋值？ - 问答 - 腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/ask/sof/28703)

</br>
16.Python中`dict` vs `collections.OrderedDict`：
```python
# A Python program to demonstrate working of OrderedDict  
from collections import OrderedDict

print("This is a Dict:n")  
d = {}  
d['a'] = 1  
d['b'] = 2  
d['c'] = 3  
d['d'] = 4

for key, value in d.items():  
print(key, value)

print("nThis is an Ordered Dict:n")  
od = OrderedDict()  
od['a'] = 1  
od['b'] = 2  
od['c'] = 3  
od['d'] = 4

for key, value in od.items():  
print(key, value)  
Output:

This is a Dict:  
('a', 1)  
('c', 3)  
('b', 2)  
('d', 4)

This is an Ordered Dict:  
('a', 1)  
('b', 2)  
('c', 3)  
('d', 4)
```
> 参考资料：
> 1. [OrderedDict in Python](https://prutor.ai/ordereddict-in-python/#:~:text=The%20only%20difference%20between%20dict,inserted%20is%20remembered%20by%20OrderedDict.)
> 2. [python - Difference between dictionary and OrderedDict - Stack Overflow](https://stackoverflow.com/questions/34305003/difference-between-dictionary-and-ordereddict)

</br>
17.遍历argparse的`parse_args()`：
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=int, default=55)
parser.add_argument('--b', type=int, default=66)
parser.add_argument('--c', type=int, default=77)

args = parser.parse_args()
print(args)
# vars() 函数返回对象object的属性和属性值的字典对象。
for arg in vars(args):
    print(arg, ':', getattr(args, arg))  # getattr() 函数是获取args中arg的属性值
```
> 参考资料：
> 1. [python遍历argparse的parse_args()_python 遍历args_集电极的博客-CSDN博客](https://blog.csdn.net/qq_38463737/article/details/121103702)

</br>
18.python代码可以正常运行，但是debug时出现`No Module named 'mmseg'`。
原因：在代码文件开头添加了上层路径
```python
# test.py
import sys
sys.path.insert(0, "..")
import mmseg
```
在代码直接运行时，运行`python test.py`，可以从上层路径直接导入`mmseg`；但是在进行debug时，运行`pythn tools/test.py`，路径相对关系发生变化。无法正常导入`mmseg`。
解决方案：
```python
# test.py
import sys 
sys.path.insert(0, "..")
sys.path.insert(0, ".") # 添加本地路径
import mmseg
```
> 参考资料：
> 1. [https://youtrack.jetbrains.com/issue/PY-43911/run-my-python-code-works-but-debugging-has-problems-ModuleNotFoundError-No-module-named-MaryPackage](https://youtrack.jetbrains.com/issue/PY-43911/run-my-python-code-works-but-debugging-has-problems-ModuleNotFoundError-No-module-named-MaryPackage)

</br>
19.在对python文件进行调试时，有将下面的字典类型变量复制、保存到txt文件并转为更易读的json格式的需求：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230304125946.png)
步骤：
1. 右键上述cfg局部变量，并copy value，保存到txt文件`cfg.txt`。
2. 从txt文件中读取，或者直接复制给变量赋值
3. 使用`json`库将其保存至`.json`文件：
```python
import json

x = txt_content # 复制赋值或从txt读取
with open("cfg.json", 'w') as outfile:
	json.dump(x, outfile, indent=2)
```
> 参考资料：
> 1. [python - How to Format dict string outputs nicely - Stack Overflow](https://stackoverflow.com/questions/3733554/how-to-format-dict-string-outputs-nicely)
> 2. [python - How do I write JSON data to a file? - Stack Overflow](https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file)

</br>
20.在python中，有时有对tuple或list中的元素值进行逐个比较（不是引用）的需求，此时需要引入`operator`模块，并调用`eq`函数：
```python
>> import operator
>> operator.eq('hello', 'name')
False
>> operator.eq('hello', 'hello')
True
```
`operator`中还存在其它对象比较函数：
```python
operator.lt(a, b)
operator.le(a, b)
operator.eq(a, b)
operator.ne(a, b)
operator.ge(a, b)
operator.gt(a, b)
operator.__lt__(a, b)
operator.__le__(a, b)
operator.__eq__(a, b)
operator.__ne__(a, b)
operator.__ge__(a, b)
operator.__gt__(a, b)
```
> 参考资料：
> 1. [Python Tuple(元组) cmp()方法 | 菜鸟教程](https://www.runoob.com/python/att-tuple-cmp.html)

</br>
21.在Python中获取当前运行文件所在路径：
```python
print(os.path.expanduser(os.path.abspath(__file__)))
```
PS：直接在python interpreter中运行该命令会报错如下错误
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230407165649.png)

</br>

22.Python中的`*`号。
```python
>>> x, y = (1, 2, 3)
ValueError: too many values to unpack (expected 2)

>>> x, *y = 1, 2, 3
>>> x
1 
>>> y 
[2, 3]

>>> def foo(x, *args):
>>>     print(x)
>>>     print(args)

>>>foo(1, 2, 3, 4)
1
[2, 3, 4]

>>> names = ("Jack", "Johnson", "Senior")
>>> fist_name, *surnames =  names
>>> print(surnames)
["Johnson", "Senior"]

```
> 参考资料：
> 1. [What does for x, *y in list mean in python - Stack Overflow](https://stackoverflow.com/questions/57814195/what-does-for-x-y-in-list-mean-in-python)

</br>
23.Python中tuple实现和list相似的`append`操作：
```python
a = []
a.append(1)
>> a
[1]

b = ()
b += (1,)
>> b
(1,)

c = (1,)
>> c
(1,)
c += (2,)
>> c
(1,2,)
```
>参考资料：
>1. [Site Unreachable](https://www.tutorialspoint.com/How-to-append-elements-in-Python-tuple)

















