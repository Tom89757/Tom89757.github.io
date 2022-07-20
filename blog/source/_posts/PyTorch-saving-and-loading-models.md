---
title: PyTorch saving and loading models
date: 2022-07-20 21:52:59
categories:
- 深度学习
tags:
- Pytorch
- 文档
---

本文记录PyTorch中的核心操作之一——saving and loading models。

<!--more-->

### saving and loading models

本文档提供了对Pytorch models进行存储和加载的不同使用场景的解决方案。当谈到存储和加载模型，有三个核心函数很相似：

* `torch.save`：存储一个serialized object到磁盘，该函数使用Python的`pickle` utility来序列化（serialization）。Models/tensors和各种类型对象的字典都可以使用该函数存储
* `torch.load`：使用`pickle`的unpickling能力来反序列化pickled对象文件到内存中。该函数也可以设置用来加载数据的设备（如gpu），见 [Saving & Loading Model Across Devices](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)。
* `torch.nn.Module.load_state_dict`：使用反序列话的state_dict加载模型的参数字典，详细信息见 [What is a state_dict?](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)。

**什么是`state_dict`?**

在Pytorch中， 一个`torch.nn.Module`模型的可学习的参数（如权重和偏差）被包含在模型的参数中（可以通过`model.parameters()`获取。一个state_dict就是一个简单的Python字典对象，其将每个layer映射到它的参数tensor。注意只有具有可学习参数的layers（如卷积层，线性层等）和具有registered buffers（batchnorm's running_mean）的layers在模型的state_dict中有入口。Optimizer对象（`torch.optim`）也有一个state_dict，它包含关于优化器的状态信息和使用的超参数。

因为state_dict是Python字典，所以它们可以很容易地存储、更新、更变和恢复，这使得Pytorch的模型和优化器得以模块化。

**Example**

下面看一下 [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) 教程中一个简单的分类器的state_dict：

    # Define model
    class TheModelClass(nn.Module):
        def __init__(self):
            super(TheModelClass, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Initialize model
    model = TheModelClass()
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

其输出为：

    Model's state_dict:
    conv1.weight     torch.Size([6, 3, 5, 5])
    conv1.bias   torch.Size([6])
    conv2.weight     torch.Size([16, 6, 5, 5])
    conv2.bias   torch.Size([16])
    fc1.weight   torch.Size([120, 400])
    fc1.bias     torch.Size([120])
    fc2.weight   torch.Size([84, 120])
    fc2.bias     torch.Size([84])
    fc3.weight   torch.Size([10, 84])
    fc3.bias     torch.Size([10])
    
    Optimizer's state_dict:
    state    {}
    param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]

**存储和加载模型用于推断**

存储/加载`state_dict`（建议）

Save：`torch.save(model.state_dict(), PATH)`

Load：

    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()

PS：PyTorch1.6版本将`torch.save`的存储格式转换为了一个新的基于zipfile的文件格式。`torch.load`仍然保持加载老的pth/pt格式文件的能力。如果想要使用`torch.save`存储老的文件格式pth/pt，可以使用参数 `_use_new_zipfile_serialization=False`。

当加载一个模型用于推断时，只有必要存储训练模型的可学习的参数。使用`torch.save()`存储模型的state_dict将对以后恢复模型给出最大的灵活性，这也是推荐它存储模型的原因。

一个PyTorch的惯例是使用pt/pth扩展名来存储模型。

记住在进行推断之前你必须调用`model.eval()`来设置dropout和batch normalization层来评估模型，不做这一步将导致生成不一致的推断结果。

PS：注意`load_state_dict()`函数将一个字典对象而不是一个存储对象的路径作为参数，这意味着在将state_dict传给该函数之前必须对其反序列化，例如，不能加载模型通过`model.load_state_dict(PATH)`。

PS：如果逆想要保存性能最好的模型（根据获得的验证损失），不要忘记`best_model_state=model.state_dict()`返回的是对state的引用而不是它的copy。你必须序列化`best_model_state`或者使用 `best_model_state = deepcopy(model.state_dict())` 否则你的`best_model_state`将会随着后续的训练迭代继续更新。结果，最终的模型state可能是一个过拟合模型的state。

**存储和加载模型**

Save：`torch.save(model, PATH)`

Load：

    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()

上述的存储/加载过程使用最直观的语法，涉及最少的代码。以这种方式存储模型将使用Python的pickle模块存储整个模型。该方法的缺点在于序列化的数据和特定的类以及当模型存储时的目录结构绑定。其原因在于pickle不存储模型类本身，而是存储一个包含该类的文件的路径，这个类会在加载时用到。因为这个原因，你的代码在其他的项目或者在重构后中使用可能会以多种形式中断。

一个PyTorch的惯例是使用pt/pth扩展名来存储模型。

记住在进行推断之前你必须调用`model.eval()`来设置dropout和batch normalization层来评估模型，不做这一步将导致生成不一致的推断结果。

> 参考资料：
>
> 1. [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)