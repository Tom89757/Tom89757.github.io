---
title: 为什么Python中列表不能用作字典的key
date: 2022-06-30 09:51:51
categories:
- 笔记
tags:
- python
---

本文是对 [Why Lists Can't Be Dictionary Keys](https://wiki.python.org/moin/DictionaryKeys) 一文的翻译。

<!--more-->

**Valid Python dictionary keys**

对python字典的键(key)的唯一要求是key是hashable。可变类型像列表，字典和集合不能满足要求，将会导致错误`TypeError: unhashable type: 'list'`。

**Why Lists Can't Be Dictionary Keys**

在python中字典也称为mappings，因为字典将key对象映射或关联到value对象。正因为此，python mappings必须满足，对给定的一个key对象，能够决定哪个value对象与之关联。

一种简单的实现方法是存储一个(key, value) pairs的列表，然后每次根据key请求值时对列表进行线性搜索。但是，这种实现方法在有大量(key, value) pairs时非常低效——从复杂性上看，算法复杂度为$O(n)$，$n$为元素个数。

python字典的实现通过要求key对象提供一个"hash" function将查找元素的复杂度降到了$O(1)$。这样一个hash function读取key对象的信息并通过它生成一个整数，称为hash值。hash值被用来确定对应的(key, value) pair应该放入哪个"bucket"。这个查找函数的伪代码看起来像下面这样：

```python
def lookup(d, key):
    '''dictionary lookup is done in three steps:
       1. A hash value of the key is computed using a hash function.

       2. The hash value addresses a location in d.data which is
          supposed to be an array of "buckets" or "collision lists"
          which contain the (key,value) pairs.

       3. The collision list addressed by the hash value is searched
          sequentially until a pair is found with pair[0] == key. The
          return value of the lookup is then pair[1].
    '''
    h = hash(key)                  # step 1
    cl = d.data[h]                 # step 2
    for pair in cl:                # step 3
        if key == pair[0]:
            return pair[1]
    else:
        raise KeyError, "Key %s not found." % key
```

这样的一个查找算法要想正确工作，hash function必须提供保证：当两个key生成不同的hash值时，这两个key不等价。即：

```python
for all i1, i2, if hash(i1) != hash(i2), then i1 != i2
```

否则，一个key对象的hash值可能使我们在错误的bucket中查找，因此永远找不到关联的value。

这样的一个查找算法要想高效工作，大多数的bucket应该只有少量的元素（最好是一个）。考虑使用下面的hash function会发生什么：

```python
def hash(obj):
    return 1
```

注意该函数满足一个hash function的需求——每当两个key有不同的hash值时，它们代表不同的对象。但是这是一个很糟糕的hash function，因为它意味着所有的(key, value) pairs将被放入一个列表中，所以每次查找都会查找整个列表。因此一个最理想的hash function具有的属性是，如果两个key生成相同的hash value，那么两个key对象是等价的，即：

```python
for all i1, i2, if hash(i1) == hash(i2), then i1 == i2
```

能够近似具有该属性的hash function会将(key, value) pairs平均的分配在各个bucket中，使查找时间减少。

**Types Usable as Dictionary Keys**

以上的讨论应该可以解释为什么Python要求：

*要能被用作字典的key，一个车对象必须支持hash function(如通过__hash__)，相等比较(如通过__eq__或__cmp__)，并且必须满足上述的正确性条件*

**Lists as Dictionary Keys**

简单来说，列表不能作为字典key是因为列表不能提供一个有效的__hash__方法，当然，一个很显然的问题是，为什么列表不提供。

考虑能够为列表提供哪些hash function。

如果列表通过id实现hash，根据hash function的定义这当然是有效的——有不同hash值的列表将有不同的id。但是列表是容器，并且大多数在列表上的操作也把它们当作容器处理。所以通过列表id实现hash可能会产生以下不期望的行为：

- 查找具有相同内容的不同列表可能会得到不同的结果，尽管比较具有相同内容的列表时会认为它们等价。
- 照字面意义在字典查找中使用列表将是pointless——这会导致`KeyError`。

如果列表通过内容实现hash(和元组一样)，也将是一个有效的hash function——具有不同hash值的列表有不同的内容。会再一次出现问题，但问题不在hash function的定义上。考虑当一个列表被用作一个字典的key，当这个列表被更改时会发生什么？如果这个更改改变了列表的hash值（因为它改变了列表内容），那么列表将在字典错误的"bucket"中。这会导致以下不期望的错误：

```python
>>> l = [1, 2]
>>> d = {}
>>> d[l] = 42
>>> l.append(3)
>>> d[l]
Traceback (most recent call last):
  File "<interactive input>", line 1, in ?
KeyError: [1, 2, 3]
>>> d[[1, 2]]
Traceback (most recent call last):
  File "<interactive input>", line 1, in ?
KeyError: [1, 2]
```

因为字典不知道key对象被修改，这样的errors只会在进行key查找时出现，而不是在key对象被更改时发现，这会导致这样的错误非常难以调试。

已经发现这两种hash列表的方法是都会有不期望的副作用，Python采取以下的特性也就很明显：

**内置的列表类型不应该作为字典key来使用**

注意到因为元组是不可变的，它们不会遭遇和列表相同的问题——它们可以通过内容进行hash而不需要担心内容修改。因此，在Python中，它们提供了一个有效的__hash__方法，因此也能作为字典key。

**User Defined Types as Dictionary Keys**

那有没有关于用户自定义的key类型的例子呢？

默认，所有用户自定义的类型当具有`hash(object)`方法（默认为`id(hash)`）和`cmp(object1, object2)`（默认为`cmp(id(object1), id(object2)`）时，可以作为字典key。以上对列表的讨论考虑了相同的要求，发现列表并不满足。为什么用户自定义类型不一样呢？

1. 在那些对象必须被放入一个mapping的情况中，object identity通常比object content更为重要。
2. 在那些对象内容真的很重要的情况下，默认设置可以通过重写__hash__和__cmp__或者__eq__来重定义。

注意当对象和一个值关联时，简单地将值作为对象的属性之一是更好的实践方式。

**Tutorials on Python's dictionaries**

下面是一些解释字典的常见用法和细节的教程：

1. [The official manual on data structures](https://docs.python.org/3/tutorial/datastructures.html)
2. [Python Dictionary : How To Create And Use them, With Examples](https://wiki.python.org/moin/DictionaryKeys)

> 参考资料：
>
> 1. [Why Lists Can't Be Dictionary Keys](https://wiki.python.org/moin/DictionaryKeys)









