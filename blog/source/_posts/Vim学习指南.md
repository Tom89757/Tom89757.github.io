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
7. [Learn Neovim The Practical Way. All articles on how to configure and… | by alpha2phi | Medium](https://alpha2phi.medium.com/learn-neovim-the-practical-way-8818fcf4830f)：medium上关于neovim配置的系列文章
8. [GitHub - LunarVim/Neovim-from-scratch: 📚 A Neovim config designed from scratch to be understandable](https://github.com/LunarVim/Neovim-from-scratch)：Lunar官方配置教程
9. [Neovim IDE from Scratch](https://www.youtube.com/watch?v=ctH-a-1eUME&list=PLhoH5vyxr6Qq41NFL4GvhFp-WLd5xzIzZ&ab_channel=chris%40machine)：Neovim配置视频教程
10. [LunarVim | LunarVim](https://www.lunarvim.org/)：neovim的变种，便于配置
11. [Home | SpaceVim](https://spacevim.org/)：neovim的变种，便于配置
12. [Getting Started | AstroNvim](https://astronvim.com/)：neovim的变种，便于配置
13. [Interactive Vim tutorial](https://www.openvim.com/)：在线vim交互式教程。
14. [GitHub - fengstats/vscode-settings: my vscode settings and theme config]：[VSCode | 日常工作流分享_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1i24y1L7uG)中分享的配置
15. [GitHub - LintaoAmons/CoolStuffes: 我的分享放这里了，大家随便拿去用啊，记得给个星星就行啦～](https://github.com/LintaoAmons/CoolStuffes)：b站up主的配置
16. [NvChad](https://github.com/NvChad/NvChad)：NvChad，另一个neovim的变种。

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

### 配置主题
以上述安装的配置为例。
1. 在对应配置文件`lua/ui/plugins.lua`中添加主题：
```lua
ui["folke/tokyonight.nvim"] = {
	lazy = false,
	name = "tokyonight", 
	config = conf.tokyonight,
}
```
2. 在对应配置文件中`lazy-lock.json`中添加主题仓库信息：
```json
"tokyonight": { "branch": "main", "commit": "affb21a81e6d7de073378eb86d02864c594104d9" },
```
3. 在`lua/core/settings.lua`中将`colorsheme`设置为`tokyonight`：
```lua
settings["colorscheme"] = "tokyonight"
```


### neovim配置java环境
1. 下载解压jdt-language-server：
```bash
#创建workspace目录，后面会用到
mkdir -p ~/.local/share/nvim/lsp/jdt-language-server/workspace/folder
cd ~/.local/share/nvim/lsp/jdt-language-server
# 下载jdt-language-server-xxxxx.tar.gz
wget https://download.eclipse.org/jdtls/milestones/1.9.0/jdt-language-server-1.9.0-202203031534.tar.gz
# 解压
tar -zxvf jdt-language-server-1.9.0-202203031534.tar.gz
```
2. 创建`/lua/modules/lang/usr/`文件夹，并添加`config.lua`和`plugins.lua`两个文件，文件内容如下：
config.lua：
```lua
local config = {}

function config.java()
	local opts = {
		cmd = {
			"java", 
			"-Declipse.application=org.eclipse.jdt.ls.core.id1",
			"-Dosgi.bundles.defaultStartLevel=4",
			"-Declipse.product=org.eclipse.jdt.ls.core.product",
			"-Dlog.protocol=true",
			"-Dlog.level=ALL",
			"-Xms1g",
			"--add-modules=ALL-SYSTEM",
			"--add-opens",
			"java.base/java.util=ALL-UNNAMED",
			"--add-opens",
			"java.base/java.lang=ALL-UNNAMED",
			"-jar",
			"/home/fg/.local/share/nvim/lsp/jdt-language-server/plugins/org.eclipse.equinox.launcher_1.6.400.v20210924-0641.jar",
			"-configuration",
			"/home/fg/.local/share/nvim/lsp/jdt-language-server/config_linux",
			"-data",
			"/home/fg/.local/share/nvim/lsp/jdt-language-server/workspace/folder",
		},
		root_dir = require("jdtls.setup").find_root({ ".git", "mvnw", "gradlew" }),
		settings = {
			java = {},
		},
		init_options = {
			bundles = {},
		},
	}
	require("jdtls").start_or_attach(opts)
end

return config
```
plugins.lua：
```lua
local custom = {}
local conf = require("modules.lang.user.config")

custom["mfussenegger/nvim-jdtls"] = {
	lazy = true,
	ft = "java",
	config = conf.java,
}

return custom
```
3. 在`lazy-lock.json`添加仓库信息：
```json
"nvim-jdtls": { "branch": "master", "commit": "1f640d14d17f20cfc63c1acc26a10f9466e66a75" },
```
4. 在`/lua/modules/editor/config.lua`中添加`java`的语法高亮：
```lua
require("nvim-treesitter.configs").setup({
		ensure_installed = {
		"java",
		},
		})
```
5. 在`/lua/modules/editor/config.lua`中添加java的debug配置：
```lua
-- local dap = require('dap')
dap.adapters.java = function(callback)
		callback({
			type = 'server';
			host = '127.0.0.1';
			port = port;
		})
	end

-- local dap = require('dap')
dap.configurations.java = {
  {
	classPaths = {},

	-- If using multi-module projects, remove otherwise.
	projectName = "yourProjectName",

	javaExec = "/usr/bin/java",
	mainClass = "your.package.name.MainClassName",
	modulePaths = {},
	name = "Launch YourClassName",
	request = "launch",
	type = "java"
  },
}	
```
> 参考资料：
> 1. [GitHub - lxyoucan/nvim-as-java-ide: 从零开始搭建Neovim Java IDE开发环境](https://github.com/lxyoucan/nvim-as-java-ide)
> 2. [Usage · ayamir/nvimdots Wiki · GitHub](https://github.com/ayamir/nvimdots/wiki/Usage)
> 3. [Java · mfussenegger/nvim-dap Wiki · GitHub](https://github.com/mfussenegger/nvim-dap/wiki/Java)
> 4. [GitHub - mfussenegger/nvim-jdtls: Extensions for the built-in LSP support in Neovim for eclipse.jdt.ls](https://github.com/mfussenegger/nvim-jdtls)

</br>
6.Vim模式在中文输入法下的问题。
> 参考资料：
> 1. [Vim模式在中文输入法下的问题 - 建议反馈 - Obsidian 中文论坛](https://forum-zh.obsidian.md/t/topic/11234/2)

</br>
7.对Obsidian的`.obsidian.vimrc`文件配置`Ctrl`键（对Obsidian而言需要安装Vimrc Support插件）
```.vimrc
"set some Ctrl- shortcuts"
nmap <C-[> <C-d> #映射为outdent，不能生效
imap <C-[> <C-d> #映射伪outdent，不能生效
```
PS：
> 参考资料：
> 1. [key bindings - How to disable Ctrl key? - Vi and Vim Stack Exchange](https://vi.stackexchange.com/questions/4060/how-to-disable-ctrl-key)
> 2. [key bindings -  Vim Stack Exchange](https://vi.stackexchange.com/questions/3225/disable-esc-but-keep-c/3570#3570)
> 3. [GitHub - esm7/obsidian-vimrc-support: A plugin for the Obsidian.md note-taking software](https://github.com/esm7/obsidian-vimrc-support)
> 4. [Can't map the Tab key w/o Mod Keys · Issue #48 · esm7/obsidian-vimrc-support · GitHub](https://github.com/esm7/obsidian-vimrc-support/issues/48) 
> 5. [vim tips and tricks: indenting](https://www.cs.swarthmore.edu/oldhelp/vim/indenting.html#:~:text=To%20indent%20the%20current%20line,by%20sw%20(repeat%20with%20.%20)&text=then%20try%20hitting%20the%20F5,or%20just%20%3Aset%20paste%20).)

</br>
8.设置vim中的yy可以copy to clipboard：
- Obsidian配置，在`.obsidian.vimrc`文件中添加：
```.vimrc
"Yank to system clipboard"
set clipboard=unnamed #重启后生效
```
- VSCode配置，在settings中搜索`vim clip`点击生效：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/1.png)
> 参考资料：
>1. [VS code VIM extension copy and paste - Stack Overflow](https://stackoverflow.com/questions/58306002/vs-code-vim-extension-copy-and-paste)
>2. [How to copy to clipboard in Vim? - Stack Overflow](https://stackoverflow.com/questions/3961859/how-to-copy-to-clipboard-in-vim)
>3. [GitHub - esm7/obsidian-vimrc-support: A plugin for the Obsidian.md note-taking software](https://github.com/esm7/obsidian-vimrc-support)

</br>
9.Vim中各种map模式nmap/imap等详解：
> 参考资料：
> 1. [What is the difference between the remap, noremap, nnoremap and vnoremap mapping commands in Vim? - Stack Overflow](https://stackoverflow.com/questions/3776117/what-is-the-difference-between-the-remap-noremap-nnoremap-and-vnoremap-mapping)
> 2. [key bindings - What is the difference between unmap and mapping to  nop? - Vi and Vim Stack Exchange](https://vi.stackexchange.com/questions/16392/what-is-the-difference-between-unmap-and-mapping-to-nop?noredirect=1)

</br>
10.`Ctrl`/`Shift`/方向键/Fn键的映射：

> 参考资料：
> 1. [vim - Map Shift + F3 in .vimrc - Super User](https://superuser.com/questions/508655/map-shift-f3-in-vimrc)
> 2. [vim: how to specify arrow keys - Stack Overflow](https://stackoverflow.com/questions/7542381/vim-how-to-specify-arrow-keys)

</br>
11.`.vimrc`配置文件中`<CR>`的含义：
> 参考资料：
> 1. [What is the meaning of a <CR> at the end of some vim mappings? - Stack Overflow](https://stackoverflow.com/questions/22142755/what-is-the-meaning-of-a-cr-at-the-end-of-some-vim-mappings)

</br>

</br>
12.Ubuntu和wsl中进行基础的`.vimrc`配置：

> 参考资料：
> 1. [A basic .vimrc file that will serve as a good template on which to build. · GitHub](https://gist.github.com/simonista/8703722)
> 2. [ubuntu下vim的配置vimrc-掘金](https://juejin.cn/s/ubuntu%E4%B8%8Bvim%E7%9A%84%E9%85%8D%E7%BD%AEvimrc)
> 3. [vim - Error Trailing Characters in Ubuntu - Stack Overflow](https://stackoverflow.com/questions/9206797/error-trailing-characters-in-ubuntu)
> 4. [Vim cursor can't be changed · Issue #4335 · microsoft/terminal · GitHub](https://github.com/microsoft/terminal/issues/4335)
> 5. [Changing Vim cursor in Windows Terminal : vim](https://www.reddit.com/r/vim/comments/uvizcu/changing_vim_cursor_in_windows_terminal/)