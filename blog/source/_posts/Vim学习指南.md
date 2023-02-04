---
title: Vim学习指南
date: 2023-02-03 16:43:38
categories:
- 环境配置
tags:
- Vim
---

本文记录一下从IDE转为Vim的一些资料和教程：
<!--more-->

1. [Neovim配置实战](https://juejin.cn/book/7051157342770954277/section/7051399376945545252)：掘金小册，30块
2. [NeoVim 基本配置 - 某春雨的后花园](https://ichunyu.github.io/neovim/)
3. [我的现代化 NeoVim 配置介绍/教程 - 知乎](https://zhuanlan.zhihu.com/p/467428462)
4. [我的现代化Neovim配置 - 知乎](https://zhuanlan.zhihu.com/p/382092667)
5. [入门指南 | SpaceVim](https://spacevim.org/cn/quick-start-guide/#windows)
6. [学习 Neovim 全 lua 配置 - 知乎](https://zhuanlan.zhihu.com/p/571617696)：Neovim配置实战的旧版，免费

### 碰到的一些问题的解决方法
1. [No C compiler found · Issue #274 · LunarVim/Neovim-from-scratch · GitHub](https://github.com/LunarVim/Neovim-from-scratch/issues/274)

### neovim配置步骤

#### 借鉴配置
可供借鉴的配置有：
1. [GitHub - nshen/learn-neovim-lua: Neovim 配置实战：从 0 到 1 打造自己的 IDE](https://github.com/nshen/learn-neovim-lua)
2. [GitHub - leslie255/nvim-config: A pretty epic NeoVim setup](https://github.com/leslie255/nvim-config)
3. [GitHub - ayamir/nvimdots: A well configured and structured Neovim.](https://github.com/ayamir/nvimdots)
此处以第3个配置为例，介绍配置流程。（因为第3个配置最为复杂，且使用较为丰富的插件，和更小众、更新型的插件管理工具lazy.nvim。
此处，先把`dotfiles`仓库中的对应文件夹映射到`~/.config/nvim`：
```bash
ln -s /mnt/d/Desktop/dotfiles/nvim2 ~/.config/nvim
```
#### 安装lazy.vim插件管理器
1. 直接在终端运行`nvim init.lua`，让neovim自动安装，出现如下报错，即无法从github clone lazy.vim进行安装。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230204222425.png)
原因：git出现问题，需要进行配置。
解决方案：向github账户添加ssh验证。
```bash
git config --global user.name "你的github账户名"
git config --global user.email "你的github账户默认的邮箱地址"
ssh-keygen -t rsa -b 4096 -C "你的github账户默认的邮箱地址"
cat ~/.ssh/id_rsa.pub # 添加到git ssh
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts 
```
2. 进行上述配置后，再次运行`nvim init.lua`，会发现如下错误，即`efm-langserver`无法安装：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230204223330.png)
原因：`efm-langserver`使用go语言编写，其使用go的包管理工具进行安装。
解决方案：安装go语言。
```bash
# 下载对应版本go压缩包后解压到指定目录
sudo tar -C /usr/local -xzf go1.20.linux-amd64.tar.gz
# 将对应目录添加至PATH环境变量，写入~/.bashrc使其永久生效
export PATH=$PATH:/usr/local/go/bin
```
3. 再次运行`nvim init.lua`，发现仍然无法安装。
原因：go仓库源在国内无法稳定访问
解决方案：配置国内镜像源，在`.bashrc`文件中写入：
```bash
# 启用 Go Modules 功能
go env -w GO111MODULE=on

# 配置 GOPROXY 环境变量，以下三选一

# 1. 七牛 CDN
go env -w  GOPROXY=https://goproxy.cn,direct
```
4. 再次运行`nvim init.lua`，发现仍然无法安装。报如下错误：
```bash
Can only use path@version syntax with 'go get'?
```
原因：go1.16以下版本无法通过`go install path@version`安装。
解决方案：卸载掉`apt-get`默认安装的`go1.13`版本，手动安装`go1.20`并添加到指定路径（也可以手动安装efm和gopls）：
```bash
export PATH=$PATH:/usr/local/go/bin
# 启用 Go Modules 功能
go env -w GO111MODULE=on
# 2. 阿里云
go env -w GOPROXY=https://mirrors.aliyun.com/goproxy/,direct
```
> 参考资料：
> 1. [Can only use path@version syntax with 'go get'? · Issue #220 · mattn/efm-langserver · GitHub](https://github.com/mattn/efm-langserver/issues/220)
> 2. [efm-langserver: Error while opening a new .cc file · Issue #98 · ayamir/nvimdots · GitHub](https://github.com/ayamir/nvimdots/issues/98)


### neovim配置java环境

> 参考资料：
> 1. [Java in Neovim | Chris@Machine](https://www.chiarulli.me/Neovim/24-neovim-and-java/)
> 2. [Using Neovim as a Java IDE | Kevin Sookocheff](https://sookocheff.com/post/vim/neovim-java-ide/)
> 3. [Neovim for Beginners — Java. Use Neovim for Java application… | by alpha2phi | Medium](https://alpha2phi.medium.com/neovim-for-beginners-java-6a86cf1a91a5)
> 4. [GitHub - lxyoucan/nvim-as-java-ide: 从零开始搭建Neovim Java IDE开发环境](https://github.com/lxyoucan/nvim-as-java-ide)