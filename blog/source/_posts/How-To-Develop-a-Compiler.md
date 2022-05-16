---
title: How to Develop a Compiler
date: 2021-07-15 12:18:39
categories:
- 笔记
tags: 
- 编译器
---

本文主要记录一下在阅读《自制编译器》并实现一个简化版的C语言编译器过程中所踩过的坑。

<!--more-->

## 环境配置

###  运行系统

由于本人只有一台配置落后的笔记本，书中要求的运行系统又是Linux，故采取了windows安装WSL的方案，具体见 [在windows上安装WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) 。

- windows版本：windows10 19043.1110
- WSL版本：Ubuntu 20.04.2 LTS

### 安装软件和依赖

书中并未要求安装gcc，由于后续可能需要进行所实现的编译器Cb和gcc的比对，决定安装gcc。安装流程为（见 [安装gcc](https://stackoverflow.com/questions/62215963/how-to-install-gcc-and-gdb-for-wslwindows-subsytem-for-linux)）：

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt autoremove -y
sudo apt-get install gcc -y
```

- gcc版本：gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

书中要求安装JRE 1.5以上版本和Java编程器。安装流程为：

```bash
sudo apt install openjdk-11-jdk
```

- Java版本：openjdk 11.0.11

### 安装书中实现的编译器Cb



