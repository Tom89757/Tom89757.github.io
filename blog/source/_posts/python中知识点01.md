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

