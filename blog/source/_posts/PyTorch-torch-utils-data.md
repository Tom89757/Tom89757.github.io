---
title: PyTorch torch.utils.data
date: 2022-07-20 21:55:35
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录一下PyTorch中最核心的组成部分之一——`torch.utils.data`。

<!--more-->

### torch.utils.data

该 package 的核心类为 `torch.utils.data.DataLoader`，表示在一个数据集上的迭代，其支持：

* map-style 和 iterable-style 的数据集
* 定制化数据加载顺序
* 自动 batching
* 单线程和多线程的数据加载
* 自动内存 pinning (固定)

这些选项通过以下的 `DataLoader` 对象的构造器配置，其有signature：

    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, *, prefetch_factor=2,
               persistent_workers=False)

#### Dataset Types

`DataLoader`构造器最重要的参数为`dataset`，它指定了从中加载数据的数据集对象。PyTorch支持两种不同类型的数据集：

* map-style datasets
* iterable-style datasets

#### Map-Style datasets

一个map-style的数据集需要实现`__getitem__()`和`__len__()`这两个protocols，表示从indices/keys (可能非整型) 到 data samples的映射。

> protocols: 管理数据传输和接收的形式和步骤，如HTTP protocol。

例如，一个数据集，当能够通过`dataset[idx]`访问时，可以从磁盘上的文件夹中读取第`idx`张image和它对应的label。详见 [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

#### Iterable-style datasets

一个iterable-style的数据集是`IterableDataset`子类的一个实例，该子类需要实现`__iter__()` protocol，并且表示在data samples上的一个迭代。这种类型的数据集尤其适合这种情况，当随机读取代价很大甚至不可能，或者batch size依赖于所获取的数据。

例如，一个数据集，当调用`iter(dataset)`时，可以返回来自数据库、远程服务器甚至实时生成的logs的数据读取流。详见 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

PS：当使用`IterableDataset`进行multi-process data loading时，相同的数据对象在每个worker process上重复，因此必须对副本进行不同的配置以避免重复数据，可以看 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)文档了解如何实现。

#### Data Loading Order and `Sampler`

对于 iterable-style 数据集，数据加载顺序完全由用户定义的迭代器控制。这允许更容易的chunk-reading和动态的batch size的实现（如，通过每次生成一个 batched sample）

本节的剩余部分关心map-style数据集的情况。`torch.utils.data.Sampler`类被用来指定在数据加载中使用的indices/keys的序列。它们代表在数据集indices上的迭代器对象，例如，在SGD (stochastic gradient decent) 的公共实例中，一个Sampler可以任意排列indices的列表并且每次生成一个indice，或者对于mini-batch SGD生成少量indices。

一个sequential或者shuffled的sampler将会自动根据传递给`Dataloader`的`shuffle`参数构造。可选地，用户可能使用`sampler`参数来指定一个custom Sampler object，每次生成要取的下一个index/key。

一个可以一次生成一个batch indices列表的custom Sampler可以作为`batch_sampler`参数传递。automatic batching可以通过`batch_size`和`drop_last`参数来开启。详见下节获取细节。

PS：`sampler`和`batch_sampler`都不兼容iterable-style数据集，因为它们没有key/index的概念。

#### Loading Batched and Non-Batched Data

`DataLoader`支持自动地将通过`batch_size`、`drop_last`、`batch_sampler`和`collate_fn`(有默认函数)参数的每个取到的data samples整理到batches中。

**Automatic batching(default)**

最通用的情况，对应取得 a minibatch of data并将它们整理进batched samplers，例如整理一维Tensors为batch的维度。

当`batch_size`(默认为1)不为None时，data loader生成batched samples而不是individual samples，`batch_size`和`drop_last`被用来指定data loader如何获取batches of dataset keys。对于map-style数据集，用户可以选择指定`batch_sampler`，其将一次生成一个list of keys。

PS：`batch_size`和`drop_last`是用来从`sampler`中构建一个`batch_sampler`的关键。对于map-style数据集，`sampler`要么由用户提供，要么基于`shuffle`参数构建。对于iterable-style数据集，没有`sampler`或`batch_sampler`的概念

在通过sampler的indices取得 a list of samples后，作为`collate_fn`参数传递的函数被用来将list of samples整理为batches。在这种情况下，从map-style数据集中的加载数据可以大致等价于：

    for indices in batch_sampler:
        yield collate_fn([dataset[i] for i in indices])

从iterable-style数据集中加载数据可以大致等价于：

    dataset_iter = iter(dataset)
    for indices in batch_sampler:
        yield collate_fn([next(dataset_iter) for _ in indices])

一个custom `collate_fn` 可以被用来 customize collation，例如，填充序列数据到batch的最大长度。

#### Disable automatic batching

在某些情况下，用户可能想在数据集中手动管理batching，或者只是简单地加载individual samples。例如，可能直接加载batched data代价更小（例如从数据库中进行块访问，或者读取连续的内存块），或者batch size是数据依赖的，或者程序被设计在individual sample上运行。在这些情况下，不使用automatic batching（使用`collate_fn`整理samples）可能更好，此时可以让数据加载器直接返回dataset对象的每个成员。

当`batch_size`和`batch_sampler`都为None时（默认`batch_sampler`为None，就禁止了automatic batching。每个从dataset中获取的sampler被作为`collate_fn`参数传递的函数处理。

当禁止automatic batching，默认的`collate_fn`简单的转换Numpy arrays为Pytorch Tensors，并且保持everything else untouched。

在这种情况下，从一个map-style数据集中加载数据可以大致等价于：

    for index in sampler:
        yield collate_fn(dataset[index])

从一个iterable-style数据集中加载数据可以大致等价于：

    for data in iter(dataset):
        yield collate_fn(data)

#### Working with `collate_fn`

当启用或禁用automatic batching时，`collate_fn`的使用略有不同。

当禁用batching时，`collate_fn`被单个的data sample调用，输出从data loader iterator中生成。这种情况下，默认的`default_fn`简单地转换Numpy arrays为Pytorch tensors。

当启用batching时，`collate_fn`每次被a list of data samples调用，需要将生成的input samples整理为a batch。本节的剩余部分描述默认的`collate_fn` ([`default_collate()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate)) 的行为。

例如，如果每个sample包含一个3-channel的数据和一个整型的class label，也就是说，dataset的每个元素返回一个tuple (`image, class_index`)，默认的`collate_fn`会整理这样的list of tuples到a single tuple of a batched image tensor和a batched class label Tensor。尤其是，默认的`collate_fn`有如下的属性：

* 总是将batch dimension作为新的dimension
  
* 自动地转换NumPy arrays和Python numerical values为PyTorch Tensors
  
* 保留数据结构，例如如果每个sample为一个dictionary，它输出一个有相同set of keys的dict，但是将batched Tensors作为值（或者lists，如果值不能转换为Tensors）。对list、tuple、namedtuple都是如此。
  
  用户可能使用定制化的`collate_fn`来实现custom batching，例如，沿着一个维度整理而不是第一个，填充变长的序列，或者对custom data types添加support。
  

如果你遇到DataLoader的输出的维度或类型和期望的不同，你应该检查你的`collate_fn`。

#### Single- and Multi-process Data Loading

DataLoader默认使用single-process数据加载。

在一个Python process内部，[Global Interpreter Lock (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) 避免在threads的完全并行的Python代码。为了避免block数据加载时的computation code，Pytorch通过将`num_workers`设置为正值来进行multi-process的数据处理。

**Single-process data loading (default)**

在这个模式，data fetching和DataLoader初始化在相同的process中进行。因此，数据加载可能会block computing。但是，这个模式可能在资源在processes (如，shared memory, file descriptors) 之间共享数据被限制时使用会更好，或者整个数据集很小可以完全在内存中整个加载。此外，single-process加载通过在进行error trace时更具有可读性，因此对调试很有用。

**Multi-process data loading**

设置参数`num_workers`为正数可以用指定数量的loader worker processes来multi-process地加载数据。

> Warning：在数次迭代之后，loader worker processes将消耗和parent process相同量的CPU memory。略

在这个模式，每次DataLoader的一个迭代器被创建时（如当你调用enumerate(dataloader)），`num_workers`数量的worker processes也被创建。此时，`dataset`、`collate_fn`和`worker_init_fn`被传递给每个worker，worker利用这些参数进行初始化并且获取数据。这意味着数据集的访问连同它的内部IO，transforms (包括`collate_fn`) 在worker process中运行。

[`torch.utils.data.get_worker_info()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info "torch.utils.data.get_worker_info") 返回在一个worker process中的多种有用的信息（包括worker id，dataset副本，初始化seed等），并且在main process中返回None。用户可能在dataset中使用这个函数和`worker_init_fn`来单独配置每个dataset副本，并且判断代码是否运行在一个worker process中。例如，这可能在sharding the dataset时尤其有用

> sharding: 将数据集存储在不同的服务器上

对map-style的数据集，main process使用`sampler`生成indices然后将indices发送给workers。所以任何shuffle随机化在main process中进行，然后再通过indices进行引导数据加载。

对于iterable-style数据集，因为每个worker process得到一个数据集对象的副本，直接进行multi-process加载经常会导致数据重复。使用`torch.utils.data.get_worker_info()`和`worker_init_fn`，用户可以独立配置每个副本。

一旦迭代终止或者迭代器被进行垃圾回收，workers就会终止。

PS：通常不建议在multi-process加载中返回CUDA tensors，因为许多微妙的原因，详见[CUDA in multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note)。作为替代，建议使用 [automatic memory pinning](https://pytorch.org/docs/stable/data.html#memory-pinning)(也就是设置`pin_memory=True`)，这可以在CUDA-enabled GPUs上进行很快的数据传输。

**Platform-specific behaviors**

因为workers依赖于Python [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.10)")，worker的启动行为在Windows和Unix平台上有所不同。略

PS：建议将主要的script代码放在`if __name__=='__main__'`中；建议确保`collate_fn`、`worker_init_fn`和`dataset`代码在最外层被定义，也就是`__main__`的外面。

**Randomness in multi-process data loading**

默认，每个worker将它的PyTorch seed设置为`base_seed`+`worker_id`，`base_seed`是main process通过它的RNG或者一个指定的`generator`生成。但是，来自其他libraries的seeds可能在初始化workers时重复，导致每个worker返回相同的随机数字。

在`worker_init_fn`中，你可以通过[`torch.utils.data.get_worker_info().seed`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info "torch.utils.data.get_worker_info") 和 [`torch.initial_seed()`](https://pytorch.org/docs/stable/generated/torch.initial_seed.html#torch.initial_seed "torch.initial_seed")访问每个worker的PyTorch seed set，并且使用它来在数据加载之前seed其他的libraries。

#### Memory Pinning

从主机到GPU的数据的copies会快得多，当它们从 pinned (page-locked) memory 中创建时。详见 [Use pinned memory buffers](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning) 如何更通用地使用 pinned memory。

对于数据加载来说，传递`pin_memory=True`给`Dataloader`将自动的把获取到的数据放在pinned memory，因此会使得对CUDA-enabled GPUs有更快的数据传输。

默认的memory pinning logic 只会识别Tensors和包含Tensors的maps/iterables。默认，如何pinning logic看到一个custom type (如果你有一个`collate_fn`返回一个custom batch type)，或者你的batch的每个元素为一个custom type，pinning logic不会认出它们，并将返回batch（或元素）而不pin the memory。为了对custom batch或者custom data type进行memory pinning，需要在custom type中定义一个`pin_memory()`方法。如下所示：

    class SimpleCustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.inp = torch.stack(transposed_data[0], 0)
            self.tgt = torch.stack(transposed_data[1], 0)
    
        # custom memory pinning method on custom type
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self
    
    def collate_wrapper(batch):
        return SimpleCustomBatch(batch)
    
    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                        pin_memory=True)
    
    for batch_ndx, sample in enumerate(loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())

完整声明形式为：

    CLASS torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False, pin_memory_device='')

DataLoader，联合一个dataset和一个sampler，提供在给定数据集上的一个迭代。

DataLoader支持map-style和iterable-style的数据集的sing-或multi-process加载，定制化的加载顺序和可选的automatci batching (collation) 和memory pinning。

参数：

* `dataset(Dataset)`：数据集，从中加载数据
  
* `batch_size(int, optional)`：对每个batch有多少个样本被加载（默认为1）
  
* `shuffle(bool, optional)`：设置为`True`时在每个epoch数据都会reshuffle（默认为False）
  
* `sampler(Sampler or Iterable, optional)`：定义从dataset中获取samples的策略。可以是任何有`__len__`实现的Iterable。如果指定sampler，`shuffle` must not be specified。
  
* `batch_sampler(Sampler or Iterable, optional)`：类似sampler，但是一次返回a batch of indices。和`batch_size`、`shuffle`、`sampler`和`drop_last`相互排斥。？
  
* `num_worker(int, optional)`：用于data loading的subprocesses的数量。0表示数据将会在main process中加载（默认为0）
  
* `collate_fn(callable, optional)`：合并a list of samples以形成 a mini-batch of Tensor(s)。当从一个map-style数据集中进行batched loading时会用到。
  
* `pin_memory(bool, optional)`：如果为`True`，data loader在返回Tensor之前会复制Tensors到device/CUDA的pinned memory。如果你的数据元素为custom type，或者你的`collate_fn`返回的batch为custom type，看下面的示例。
  
* `drop_last(bool, optional)`：设置为`True`时会drop最后的不完整的batch，如果dataset size不能被batch size整除的话。如果为`False`，数据集的尺寸不能被batch size整除，那么最后的batch将会更小（默认为False）
  
* `timeout(numeric, optional)`：如果为正，表示从workers收集a batch的timeout值。应该总是非负（默认为0）
  
* `worker_init_fn(callable, optional)`：如果不为`None`，将使用worker id ([0, num_workers-1]范围内的整数) 作为输入在每个worker subprocess中被调用，在seeding之后，data loading之前（默认为None）
  
* `generator(torch.Generator, optional)`：如果不为`None`，RandomSampler将使用RNG生成随机indexes，并且为workers生成`base_seed`（默认为None）
  
* `prefetch_factor(int, optional, keyword-only arg)`：被每个worker提前加载的batches的数量。`2`表示在所有wrokers上将有总共2*num_workers的batches被提前获得（默认为2）
  
* `persistent_workers(bool, optional)`：如果为`True`，data loader在一个dataset被处理完一次后不会关闭worker processes，这允许保持workers Dataset instances存活（默认为False）
  
* `pin_memory_device(str, optional)`：如果pin_memory设置为True，data loader在返回Tensors之前会将他们复制到device pinned memory。
  

#### torch.utils.data.Dataset

一个表示Dataset的抽象类。

所有表示从keys到data samples的映射的数据集都应该是它的子类。所有的子类应该重写`__getitme__()`，该方法支持对一个给定的key获取对应的data sample。子类也能选择性地重写`__len__()`，该方法返回数据集的尺寸，该尺寸与Sampler的实现和DataLoader的默认选项有关。

PS：DataLoader默认构建一个生成integral indices的index sampler。为了使它可以作用于具有non-integral indices/keys的map-style数据集，必须提供一个custom sampler。

#### torch.utils.data.default_collate(batch)

一个函数，将a batch of data作为输入，将batch内的元素放入一个具有outer dimenstion (batch size)的tensor。其输出类型可能是一个`torch.Tensor`，一个`torch.Tensor`的Sequence，一个`torch.Tensor`的Collection，或者不变，其依赖于输入类型。当在DataLoader中定义batch_size或者batch_sampler时该函数可以作为collation的默认函数。下面是通常的input type (基于batch内部的元素类型) 和它映射为的output type：

参数：

*    `batch`：等待整理的single batch

调用实例：

    # Example with a batch of `int`s:
    default_collate([0, 1, 2, 3])
    # Example with a batch of `str`s:
    default_collate(['a', 'b', 'c'])
    # Example with `Map` inside the batch:
    default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
    # Example with `NamedTuple` inside the batch:
    Point = namedtuple('Point', ['x', 'y'])
    default_collate([Point(0, 0), Point(1, 1)])
    # Example with `Tuple` inside the batch:
    default_collate([(0, 1), (2, 3)])
    # Example with `List` inside the batch:
    default_collate([[0, 1], [2, 3]])

#### torch.utils.data.Sampler(data_source)

所有Samplers的基类。

每个Sampler子类必须提供`__iter__()`方法，以此提供在dataset元素的indices上的迭代，和一个`__len__()`返回迭代器的长度。

PS：`__len__()`并不是DataLoader严格要求的，但是在有任何涉及到DataLoader的长度计算时最好提供。

> 参考资料：
>
> 1. [TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html)