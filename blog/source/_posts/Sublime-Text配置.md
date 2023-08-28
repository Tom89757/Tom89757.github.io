---
title: Sublime Text配置
date: 2023-08-28 13:05:49
categories:
- 环境配置
tags:
- Sublime Text 
---
本文记录一下在 Sublime Text编辑器中一些常用的配置方法。
<!--more-->

### 打开Vim编辑模式
1. 选择`Preferences->Settings`菜单
2. 编辑右侧settings文件中的`ignored_packages`配置，即添加
 ```
ignored_packages" : []
```
3. 默认模式为插入模式，可以添加以下配置进行修改：
```json
"vintage_start_in_command_mode": true
```
> 参考资料：
> 1. [Vintage Mode](https://www.sublimetext.com/docs/vintage.html)
> 2. [Sublime text is your vim editor – Wenbin Fei](https://wenbinfei.github.io/sublime-vim/)

### 自动保存

> 参考资料：
> 1. [How to enable Autosave feature in Sublime Text editor](https://salitha94.blogspot.com/2017/11/how-to-enable-autosave-feature-in-sublime-text.html)