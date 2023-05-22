---
title: React学习笔记01
date: 2023-05-22 19:30:32
categories:
- 前端 
- 笔记
tags:
- React 
---
本文记录一下学习React过程中的笔记：
<!--more-->
参考资料：
1. [Intro to React - Google 簡報](https://docs.google.com/presentation/d/1Ep-rzXbNfd9HhAWoRgFchV_xGzPMQ5wm3lQIMpA6FQY/edit#slide=id.g10b8b83d570_1_614)
### 定义
React是一个构建user interfaces的JavaScript library。React Apps是 "components of components"。一个React App可以看作一棵Component Tree：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522194007.png)

因此，React可以更具体地定义为：
- React是一个Framework，让你可以将webside分割为多个reusable components
- 每个component是一个像是你自己定义的"custom HTML tag"。
- 你的React app可以分割为a "tree" of components。


### Comment Component
以Comment为例，讲述如何generalize出一个Comment Component。

#### Props
React components使用props来和彼此communicate。每个parent component可以通过给它的child components传递props来传递信息。Props可以是任意的JavaScript value，包括objects, arrays, 和functions。详见参考资料。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522200459.png)
现在我们得到了一个reusable comment component，我们大功告成了吗？并没有，因为props是immutable，我们不能对comments点击Like。因为我们不能使用props来存储Like信息。
> 参考资料：
> 1. [Passing Props to a Component – React](https://react.dev/learn/passing-props-to-a-component)


#### State
Components需要记忆信息：当前input value, 当前image, shopping cart等。在React中，这种component-specific memory称为state。
State是由一个component保有的private information。详见参考资料
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522201654.png)

> 参考资料：
> 1. [State: A Component's Memory – React](https://react.dev/learn/state-a-components-memory)

快速回顾一下，父组件Post和子组件Comment有如下关系：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522201843.png)
以twitter页面为例：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522205614.png)


