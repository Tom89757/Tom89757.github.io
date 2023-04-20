---
title: ohmyzsh使用指南
date: 2023-04-19 17:36:05
tags:
categories:
- 环境配置
tags:
- tmux 
---

本文介绍一下ohmyzsh的学习和使用过程：
<!--more-->
参考资料6和7为主要配置流程：


> 参考资料：
> 1. [GitHub - ohmyzsh/ohmyzsh: 🙃 A delightful community-driven (with 2,100+ contributors) framework for managing your zsh configuration)
> 2. [Installing ZSH · ohmyzsh/ohmyzsh Wiki · GitHub](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH)
> 3. [windows subsystem for linux - [process exited with code 1], can't open WSL, zsh - Stack Overflow](https://stackoverflow.com/questions/67261530/process-exited-with-code-1-cant-open-wsl-zsh)
> 4. [优雅简洁的zim美化你的zsh终端，媲美甚至超越 ohmyzsh - 勒勒乐了 - 博客园](https://www.cnblogs.com/matytan/p/16684665.html)
> 5. [优雅简洁的zim美化你的zsh终端，媲美甚至超越 ohmyzsh_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Bg411m7ND)
> 6. [终端神器ohmyzsh_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1C7411V7M8)
> 7. [Linux/Mac OS下安装并配置oh my zsh | SunPages](https://www.sunhanwu.top/archives/ohmyzsh)
> 8. [GitHub - unixorn/awesome-zsh-plugins: A collection of ZSH frameworks, plugins, themes and tutorials.](https://github.com/unixorn/awesome-zsh-plugins)
> 9. [oh-my-zsh让终端好用到飞起~ - 掘金](https://juejin.cn/post/6844903939121348616)

### 将.bashrc中的alias迁移到.zshrc

> 参考资料：
> 1. [macos - Easiest way to migrate aliases from bash to zsh - Ask Different](https://apple.stackexchange.com/questions/371867/easiest-way-to-migrate-aliases-from-bash-to-zsh)


### zsh在/mnt/...目录中的git仓库中prompt很慢

> 参考资料：
> 1. [Slow prompt command (oh-my-zsh) · Issue #1 · hsab/WSL-config · GitHub](https://github.com/hsab/WSL-config/issues/1)


### 代理配置
配置完成后：
- `proxy set`：开启代理
- `proxy unset`：关闭代理
- `proxy test`：查看代理状态
> 参考资料：
> 1. [WSL2配置代理 - Leaos - 博客园](https://www.cnblogs.com/tuilk/p/16287472.html)


### 为指定用户安装zsh (不需要root权限)
下述安装过程主要参考资料6和资料7：
1. 从源码安装`zsh`的依赖包`ncurses`：
```bash
wget ftp://ftp.gnu.org/gnu/ncurses/ncurses-6.1.tar.gz tar xf ncurses-6.1.tar.gz cd ncurses-6.1 ./configure --prefix=$HOME/local CXXFLAGS="-fPIC" CFLAGS="-fPIC" make -j && make install
```
2. 构建并安装`zsh`（将下述命令写入脚本`zsh.sh`并运行`sh zsh.sh`j：
```bash
ZSH_SRC_NAME=$HOME/packages/zsh.tar.xz ZSH_PACK_DIR=$HOME/packages/zsh ZSH_LINK="https://sourceforge.net/projects/zsh/files/latest/download" if [[ ! -d "$ZSH_PACK_DIR" ]]; then echo "Creating zsh directory under packages directory" mkdir -p "$ZSH_PACK_DIR" fi if [[ ! -f $ZSH_SRC_NAME ]]; then curl -Lo "$ZSH_SRC_NAME" "$ZSH_LINK" fi tar xJvf "$ZSH_SRC_NAME" -C "$ZSH_PACK_DIR" --strip-components 1 cd "$ZSH_PACK_DIR" ./configure --prefix="$HOME/local" \ CPPFLAGS="-I$HOME/local/include" \ LDFLAGS="-L$HOME/local/lib" make -j && make install
```
3. 设置默认shell为`zsh`。创建`~/.bash_profile`文件并写入：
```bash
export PATH=$HOME/local/bin:$PATH export SHELL=`which zsh` [ -f "$SHELL" ] && exec "$SHELL" -l
```
4. 运行`source ~/.bash_profile`启动`zsh`。选择`q`选项暂不进行配置，此时位于zsh shell。
5. 直接采用资料6中的配置（可以将`ohmyzsh.sh`下载到本地运行`sh ohmyzsh.sh`
```bash
sh -c "$(curl -fsSL https://www.sunhanwu.top/upload/2022/12/ohmyzsh.sh)"
```
6. 安装第三方插件：
```bash
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```
7. 在`.zshrc`中配置所需要的插件（功能介绍见资料6）：
```bash
bash plugins=( git extract fzf z zsh-autosuggestions zsh-syntax-highlighting wd sudo )
```
8. 将原本`.bashrc`中的`alias`和`export`配置写入`~/.bash_aliases`和`~/.bash_path`文件。然后在`.zshrc`文件中写入生效：
```bash
if [ -f ~/.bash_aliases ]; then
   . ~/.bash_aliases
fi

if [ -f ~/.bash_path ]; then
   . ~/.bash_path
fi

```
运行`source ~/.zshrc`即可生效。
> 参考资料：
> 1. [zsh/INSTALL at master · zsh-users/zsh · GitHub](https://github.com/zsh-users/zsh/blob/master/INSTALL)
> 2. [software installation - Installing zsh from source file without root user access - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/673669/installing-zsh-from-source-file-without-root-user-access)
> 3. [Z-Shell Frequently-Asked Questions](https://zsh.sourceforge.io/FAQ/zshfaq01.html#l7)
> 4. [linux - Install zsh without root access? - Stack Overflow](https://stackoverflow.com/questions/15293406/install-zsh-without-root-access)
> 5. [Building Zsh from Source and Configuring It on CentOS - jdhao's digital space](https://jdhao.github.io/2018/10/13/centos_zsh_install_use/)
> 6. [Linux/Mac OS下安装并配置oh my zsh | SunPages](https://www.sunhanwu.top/archives/ohmyzsh)












