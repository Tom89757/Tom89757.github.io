---
title: nodejs使用指南
date: 2022-05-14 23:32:40
categories:
- 开发工具
tags:
- npm
- nodejs
---



本文记录一下使用npm管理包时的常见操作：

<!--more-->

1.`npm root -g`：查看`npm`安装全局packages时的安装地址。例如在Windows10系统中运行其结果如下：

![image-20220514233532921](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220514233532921.png)

> 参考资料：[Where does npm install packages?](https://stackoverflow.com/questions/5926672/where-does-npm-install-packages)

</br>

2.`npm install package_name`：局部安装package。安装的package在当前目录的node_modules文件夹中。简写形式为`npm i package_name`。

</br>

3.`npm install -g package_name`：全局安装package。安装的package运行`npm root -g`后的展示路径中。简写形式为`npm i -g package_name`。

</br>

4.`npm uninstall package_name `：局部卸载package。要卸载的package在当前目录的node_modules文件夹中。简写形式为`npm un package_name` 。

</br>

5.`npm uninstall -g package_name `：全局卸载package。要卸载的package在运行`npm root -g`后的展示路径中中。简写形式为`npm un -g package_name` 。

</br>

6.`npm i package_name` vs `npm i package_name --save` vs `npm i package_name --save-dev`：

PS：本问题在为hexo博客安装`hexo`和`hexo-cli`包时出现，解决方案为使用`--save-dev`选项安装。

- `npm i package_name --save`：当需要为自己的`app/modules`安装依赖时，使用该命令可以在安装相应package之后将其添加到当前路径中`package.json`文件的`"dependencies"`子对象中。如下图所示：

  ![image-20220515003241850](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515003241850.png)

- `npm i package_name`：当前版本中，该命令与`npm i package_name --save`等价，同样会将安装的package添加到当前路径中`package.json`文件的`"dependencies"`子对象中。即`--save`为`npm i`命令的默认选项。

- `npm i package_name --save-dev`：当为开发安装依赖包时，需要使用此命令。该命令会将 the third-party package添加到当前路径中`package.json`文件的`"devDependencies"`子对象中。如下图所示：

  ![image-20220515004019352](https://hexo-1302648630.cos.ap-beijing.myqcloud.com/2022-05-15/image-20220515004019352.png)

> 参考资料：
>
> 1. [What is the difference between --save and --save-dev](https://stackoverflow.com/questions/22891211/what-is-the-difference-between-save-and-save-dev)
> 2. [npm-install](https://docs.npmjs.com/cli/v8/commands/npm-install)

</br>
7.Linux安装nodejs：
```bash
VERSION=v10.15.0
DISTRO=linux-x64
sudo mkdir -p /usr/local/lib/nodejs
sudo tar -xJvf node-$VERSION-$DISTRO.tar.xz -C /usr/local/lib/nodejs 
# Nodejs
VERSION=v10.15.0
DISTRO=linux-x64
export PATH=/usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin:$PATH
```
> 参考资料：
> 1. [Installation · nodejs/help Wiki · GitHub](https://github.com/nodejs/help/wiki/Installation)

</br>
8.查看当前项目所安装的依赖：
Github Copilot：您可以使用 `npm list` 命令来查看当前 npm 项目中各个包的版本。在项目根目录下运行该命令，将输出当前项目中所有已安装的包及其版本号。如果您只想查看生产环境依赖项的版本，可以使用 `npm list --prod` 命令。如果您只想查看开发环境依赖项的版本，可以使用 `npm list --dev` 命令。

</br>
9.指定npm某个包的版本：
```package.json
"react": "16.14.0" #某个版本
"react": "latest" #最新版本
```

</br>
10.查看某个npm包的所有版本：
Github Copilot：您可以使用 npm view 命令来查看某个包的所有版本。在终端中运行以下命令，将 package-name 替换为您要查看的包的名称：
``` bash
npm view <package-name> versions
```
这将输出该包的所有版本号。如果您只想查看稳定版本，可以使用 npm view package-name versions --json | grep -v "-" 命令。
例如 `npm view react version`

</br>
查看当前npm项目中各个包的版本
Github Copilot：您可以使用 `npm list` 命令来查看当前 npm 项目中各个包的版本。在项目根目录下运行该命令，将输出当前项目中所有已安装的包及其版本号。如果您只想查看生产环境依赖项的版本，可以使用 `npm list --prod` 命令。如果您只想查看开发环境依赖项的版本，可以使用 `npm list --dev` 命令。