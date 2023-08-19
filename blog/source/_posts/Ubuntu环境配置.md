---
title: Ubuntu环境配置
date: 2023-06-09 16:12:41
categories:
- 环境配置
tags:
- Ubuntu 
---
本文记录一下Ubuntu系统的环境配置：
<!--more-->

### tldr
以简洁的方式说明各个常用命令如`tar`的用法。
安装：
- `npm install -g tldr`：node.js中安装
- `pip3 install tldr`：linux中安装
使用：
- `tldr git`：即可查看git的用法，第一次使用时需要建立索引
- 给git bash配置颜色。参考资料2/3/4。
```bash
mklink C:\Users\A\.oh-my-zsh\custom\plugins\tldr\_tldr D:\Desktop\dotfiles\git\tldr-node-client\bin\completion\zsh\_tldr
mklink C:\Users\A\.tldrrc D:\Desktop\dotfiles\git\.tldrrc
```
PS：根据参考资料2和3，在gitbash中`tldr`的输出没有颜色，此时可以在`.zshrc`中添加`export FORCE_COLOR=2`。
> 参考资料：
> 1. [GitHub - tldr-pages/tldr: 📚 Collaborative cheatsheets for console commands](https://github.com/tldr-pages/tldr)
> 2. [How to get colors? · Issue #1262 · tldr-pages/tldr · GitHub](https://github.com/tldr-pages/tldr/issues/1262)
> 3. [tldr doesnt pick color config from .tldrrc file · Issue #276 · tldr-pages/tldr-node-client · GitHub](https://github.com/tldr-pages/tldr-node-client/issues/276)
> 4. [GitHub - tldr-pages/tldr-node-client: Node.js command-line client for tldr pages](https://github.com/tldr-pages/tldr-node-client)

### tree 
在不使用sudo的情况下安装tree命令：
```bash
mkdir ~/deb
cd ~/deb
apt download tree
dpkg-deb -xv ./*deb ./
export PATH="/home/FT/deb/usr/bin:$PATH" # 加入路径
```
使用：
```bash
tree -I '*png|*pyc|*.jpg'
```
> 参考资料：
> 1. [apt - How to install tree on Ubuntu without sudo right? - Ask Ubuntu](https://askubuntu.com/questions/1322467/how-to-install-tree-on-ubuntu-without-sudo-right)
> 2. [How do we specify multiple ignore patterns for `tree` command? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/47805/how-do-we-specify-multiple-ignore-patterns-for-tree-command)


### trash-cli
使用`trash`替代`rm`，更加安全，类似回收站

> 参考资料：
> 1. [trash-cli · PyPI](https://pypi.org/project/trash-cli/)


### bat
和cat功能类似，具有语法高亮
- `sudo apt install bat`：在Ubuntu上安装
- `choco install bat`：在windows10上安装
> 参考资料：
> 1. [GitHub - sharkdp/bat: A cat(1) clone with wings.](https://github.com/sharkdp/bat/#installation)


### ripgrep
使用regex模式递归地在当前目录下搜索的工具。
> 参考资料：
> 1. [GitHub - BurntSushi/ripgrep: ripgrep recursively searches directories for a regex pattern while respecting your gitignore](https://github.com/BurntSushi/ripgrep#installation)

### 综合类
1. [What helps people get comfortable on the command line?](https://jvns.ca/blog/2023/08/08/what-helps-people-get-comfortable-on-the-command-line-/)：推荐了很多使你习惯使用命令行的工具

### 从deb文件安装软件
GitHub Copilot: 在Linux中，可以使用`dpkg`命令从`.deb`文件安装软件包。具体来说，可以按照以下步骤进行操作：
1. 下载`.deb`文件。可以从软件包的官方网站或其他可信来源下载`.deb`文件。（下载对应架构的版本，amd64/arm64）
2. 打开终端，并使用`cd`命令进入`.deb`文件所在的目录。
3. 使用`dpkg`命令安装软件包。例如，要安装名为`example.deb`的软件包，可以使用以下命令：
   ```
   sudo dpkg -i example.deb
   ```
   其中，`-i`选项表示安装软件包，`example.deb`为要安装的软件包文件名。需要使用`sudo`命令以管理员权限运行`dpkg`命令。
4. 如果安装过程中出现依赖项错误，可以使用以下命令来修复依赖项：
   ```
   sudo apt-get install -f
   ```
   该命令会自动安装缺失的依赖项，并尝试修复其他依赖项错误。
5. 安装完成后，可以使用`dpkg`命令来查询已安装的软件包。例如，要查询名为`example`的软件包是否已安装，可以使用以下命令：
   ```
   dpkg -s example
   ```
   如果软件包已安装，该命令会显示软件包的详细信息，包括版本号、安装时间等。
需要注意的是，使用`dpkg`命令安装软件包时，可能会出现依赖项错误或其他问题。如果无法解决问题，可以尝试使用其他软件包管理工具，例如`apt`或`yum`，来安装软件包。
总之，要在Linux中从`.deb`文件安装软件包，可以使用`dpkg`命令，并使用`sudo`命令以管理员权限运行该命令。如果出现依赖项错误，可以使用`apt-get`命令来修复依赖项。