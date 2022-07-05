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



> 参考资料：
>
> 1. [What are metaclasses in Python?](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python)





















