---
title: 客户端Web开发工具
date: 2023-05-22 21:44:29
categories:
- 前端 
- 资料 
tags:
- Web开发
---
本文记录一下MDN官方给出的客户端Web开发工具：
<!--more-->
参考资料：
1. [理解客户端 web 开发工具 - 学习 Web 开发 | MDN](https://developer.mozilla.org/zh-CN/docs/Learn/Tools_and_testing/Understanding_client-side_tools)

> 尝试使用这里列举的工具的先决条件是学习HTML/CSS/JavaScript的核心基础知识。

本指南分为五个部分：
- 客户端工具概览
- 命令行速成教程
- 包管理基础
- 一个完整的工具链
- 发布你的应用（英文）

## 客户端工具概览
客户端工具可以分为三个阶段：
- 安全网络：在代码开发期间有用的工具
- 转换：以某种方式转换代码的工具，例如将一种中间语言转换为浏览器可以理解的JavaScript语言
- 开发后阶段：编写完代码后有用的工具，如测试和部署工具

### 安全网络
包括使你的开发过程更容易生成稳定可靠的代码；也包括帮助你避免错误或自动纠正错误，而不必每次都从头开始构建代码。
#### Linter
Linter检查你的代码并告诉你存在任何错误，它们是什么类型的错误，以及出现在哪些代码上。（通常，不仅可以报告错误，还可以报告任何违反团队正在使用的样式指南的行为）
1. [eslint](https://eslint.org/)：业界标准的JavaScript linter，一些公司 [shared their eslint configs](https://www.npmjs.com/search?q=keywords:eslintconfig)
2. [csslint](http://csslint.net/)：用于CSS
3. [webhint](https://webhint.io/)：可以作为[Node.js command-line tool](https://webhint.io/docs/user-guide/) 和 [VS Code extension](https://marketplace.visualstudio.com/items?itemName=webhint.vscode-webhint)使用。

#### 源代码控制
Git

#### 代码格式化
其根据样式规则，确保代码被正确格式化。理想情况下自动修复它们发现的错误。[Prettier](https://prettier.io/)是其中的翘楚，会在命令行速成课中介绍。

#### 打包工具
这些工具让你的代码准备投入到生产环境中，例如通过tree-shaking确保只有code libraries中的部分被放入最后的产品代码中；或者去除所有产品代码中的空格使得其在被上传到服务器之前足够小。
1. [Parcel](https://parceljs.org/)：不仅可以完成上述任务，还能打包像是HTML/CSS/images文件等为bundles方便部署；也可以自动帮你添加依赖以便你随时使用它们。
2. [Webpack](https://webpack.js.org/)：另一个流行的打包工具，做相似的事情。

### 转换
web应用程序生命周期的这个阶段通常运行你编写 "modern code" (例如最新的CSS或者JavaScript特性，这些特性可能还没有得到浏览器的支持)，或者完全使用另一种语言编写，例如 [TypeScript](https://www.typescriptlang.org/)。转换工具将帮你生成与浏览器兼容的代码，以用于生产环境。
web开发被认为有三种语言HTML/CSS/JavaScript。这些语言都有转换工具。转换主要提供了两个好处：
1. 可以使用最新的语言特性编写代码，然后将其转换为可以在日常设备上使用的代码。
	- [Babel](https://babeljs.io/)：一个JavaScript编译器，开发人员可以编写和发布 [plugins for Babel](https://babeljs.io/docs/en/plugins)。
	- [PostCSS](https://postcss.org/)：类似Babel，支持最先进的CSS特性。如何没有对应的CSS特性，PostCSS会安装一个JavaScript polyfill来模拟想要的CSS效果
2. 选择使用一种完全不同的语言编写代码，并转换为web兼容的语言：
	- [Sass/SCSS](https://sass-lang.com/)：CSS扩展语言，允许使用诸多特性
	- [TypeScript](https://www.typescriptlang.org/)：JavaScript的超集，提供了一些额外特性。TypeScript编译器生成产品代码时将TypeScript转换为JavaScript。
	- 框架例如 [React](https://reactjs.org/)、[Ember](https://emberjs.com/) 和 [Vue](https://vuejs.org/)：提供了很多功能，允许你使用它们构建在普通JavaScript上的自定义语法来使用它们。后台框架的JavaScript努力解释它们的定制语法，并呈现为最终的web应用程序。

### 开发后阶段
开发后阶段工具确保软件可以访问web并继续运行，包括部署流程、测试框架、审计工具等。一旦配置完毕，基本自动运行，只有出现错误时才弹出窗口。
#### 测试工具
测试工具自动对你的代码进行测试，在确保下一步操作之前（例如推送到github repo前）代码是准确的。
- 编写测试的框架包括：[Jest](https://jestjs.io/), [Mocha](https://mochajs.org/), and [Jasmine](https://jasmine.github.io/)。
- 自动测试运行和通知系统包括：[Travis CI](https://travis-ci.org/), [Jenkins](https://www.jenkins.io/), [Circle CI](https://circleci.com/), and [others](https://en.wikipedia.org/wiki/List_of_build_automation_software#Continuous_integration)
#### 部署工具
部署系统允许你发布网站，可用于静态或动态站点，通常和测试系统一起工作。例如，一个典型的工具链会等待你推送系统到远程仓库，运行测试看是否changes是可行的，如果测试通过自动部署你的app到产品站点。
[Netlify](https://www.netlify.com/)是最流行的部署工具，其它的包括 [Vercel](https://vercel.com/) and [GitHub Pages](https://pages.github.com/)。

#### 其它
-  [Code Climate](https://codeclimate.com/)：收集代码质量指标
- [webhint browser extension](https://webhint.io/docs/user-guide/extensions/extension-browser/)：进行跨浏览器兼容性和其它checks的性能运行时分析
-  [GitHub bots](https://probot.github.io/)：提供更强大的Github功能
- [Updown](https://updown.io/)：提供app运行时监控。
...

### 如何选择并寻求特殊工具的帮助
见 [客户端工具概述 - 学习 Web 开发 | MDN](https://developer.mozilla.org/zh-CN/docs/Learn/Tools_and_testing/Understanding_client-side_tools/Overview)

## 命令行速成课
本节课直接从添加工具开始
### 添加工具
安装node.js

### 下载Prettier
```bash
npm install --global prettier //全局安装
prettier //检查是否安装成功
prettier --check index.js //检查index.js是否符合格式
prettier --write index.js //修正index.js中的格式错误
```
使用Prettier有很多实现自动化的方法。
- 在将代码提交到git repo前：使用[Husky](https://github.com/typicode/husky)
- 在代码编辑器中保存时：无论是[VS Code](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode), [Atom](https://atom.io/packages/prettier-atom), 或[Sublime Text](https://packagecontrol.io/packages/JsPrettier)。推荐。
- 作为持续集成检查（continuous integration checks) 的一部分：[GitHub Actions](https://github.com/features/actions)使用

### 其它可以玩的工具

## 软件包管理基础
### 项目中的依赖项
依赖指由他人编写的第三方软件，理想情况下可以为你解决单一的问题。项目依赖可以是整个JavaScript框架例如React或者Vue，也可以是很小的日期库，也可以是一个命令行工具。在web上发布前，需要一些现代工具将代码和依赖项构建为bundle。
> bundle通常用于指定一个单独的文件，包含了软件的所有JavaScript，通常被尽可能压缩以减少下载和浏览器中访问所需的时间。

像npm这样软件包管理器不仅可以干净地添加和删除依赖项，还有其他的优点。

### 什么是软件包管理器
软件包管理器是一个可以管理你的项目依赖项的系统。
- 软件包管理器让你可以安装新的依赖（包），管理包在文件系统上的存储位置，并让你可以发布自己的包。
理论上你可以手动下载和存储项目依赖项，包管理器是非必须的。但此时需要你手动进行如下操作：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522230859.png)

此外：
- 包管理器可以处理重复的依赖项（这对于前端开发非常重要和常见）。
- 包管理器提供的局部安装可以提供代码可移植性，锁定依赖项版本。
PS：不同的包管理器还有yarn和pnpm。

### 软件包仓库
仓库是实现软件包发布和安装的关键部分。npm仓库位于  [npmjs.com](https://www.npmjs.com/)。npm官方软件包仓库不是唯一的选择，你可以选择Microsoft或Github提供的代理服务。

### 使用软件包生态系统
通过实例介绍如何使用软件包管理器和仓库安装命令行使用程序。
详见见[软件包管理基础 - 学习 Web 开发 | MDN](https://developer.mozilla.org/zh-CN/docs/Learn/Tools_and_testing/Understanding_client-side_tools/Package_management)。


### 为生产环境构建我们的代码
上述代码还没有准备好用于生产环境。大多数构建工具都有”开发模式“和”生产模式“，重要的区别在于，在最终网站中不需要在开发中使用的需要多功能，因此这些功能将在生产环境中被剥离，例如”模块热替换“，”实时重新加载“和”未压缩和注释的源代码“。
详见文档
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522233229.png)


### 减少应用的文件大小
我们可以要求软件检查我们对代码的使用，在构件中仅仅包含实际使用的函数（称为Tree Shaking）。
主要由三个主要的打包工具：
- [Rollup](https://rollupjs.org/guide/en/)：将Tree Shaking和代码拆分作为核心特性。
- Webpack需要一些配置（配置可能不止一些，十分复杂）
- Parcel：在Parcel 2之前，需要一个特殊的标志`--experimental-scope-hoisting`来进行Tree Shaking构建。
下面继续使用parcel。
```bash
npx parcel build index.html --experimental-scope-hoisting
```
此时发现生成的js文件要小得多。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522233237.png)

### 包管理器客户端的简要指南
- pnpm旨在提供和npm完全相同的参数选项，但使用和npm不同的方法下载和存储包，以减少总磁盘空间的占用。
- yarn相比npm更快。
PS：不需要使用npm包管理器从npm仓库安装包， pnpm和yarm可以使用和npm相同的`package.json`格式，并安装来自npm和其它软件包仓库的任何软件包。
下面回顾使用包管理器进行的常见操作：
#### 初始化一个项目
```bash
npm init
yarn init
```
#### 安装依赖
```bash
npm install date-dns 
yarn add date-fns 
```
#### 更新依赖
```bash
npm update 
yarn upgrade
```
版本号的确定通过 [semver](https://semver.org/)系统实现，一个测试semver值的绝佳方法是使用 [semver 计算器](https://semver.npmjs.com/)。版本可以表示为`Major.Minor.Patch`，例如`2.0.1`。

#### 漏洞检查
```bash
npm audit
yarn audit
```
会检查项目的所有依赖树，并通过漏洞数据库检查正在使用的特定版本。漏洞的详细信息可见[Snyk 项目](https://snyk.io/)

#### 检查一个依赖
```bash
npm ls date-fns
yarn why date-fns
```
将显示已安装该依赖项的版本，以及它如何被包含在你的项目中。

#### 创建自己的命令
软件包管理器还支持创建自己的命令并从命令行执行它们。
详见文档。

## 介绍完整的工具链
### 介绍我们的学习案例
本文创建的工具链将用于构建和部署一个迷你网站，取自[NASA 开放 API](https://api.nasa.gov/)，在线版本为 [Counting potential earth HAZARDS…](https://near-misses.netlify.app/)

### 工具链中使用的工具
本文中将使用以下工具和功能
- [JSX](https://reactjs.org/docs/introducing-jsx.html)：和React相关的语法扩展，允许在JavaScript中定义component structure。
- 最新的JavaScript内置特性（在撰写本文时），例如`import`。
- 有用的开发工具，例如用于格式化的Prettier和用于代码规范检查的ESLint。
- PostCSS提供CSS嵌套功能
- Parcel用于构建和压缩代码，并自动编写配置文件的内容
- GitHub用于管理源代码
- Netlify用于自动化部署过程。

### 工具链及其固有的复杂性
最小的工具链是根本没有工具链。可以根据需要增删工具链中的环节。

### 一些先决条件
注册GitHub和Netlify

### 工具的三个阶段
- 安全的网络：也可以称为开发环境
- 编译与构建：允许在开发过程中使用编程语言的最新特性或其它语言（如JSX或TypeScript），然后转译为浏览器可以允许的代码
- 开发后阶段：确保软件持续运行。包括测试和部署

### 创建开发环境
包括：
- 软件包安装工具：node.js和npm
- 代码修订控制：git
- 代码格式化工具：prettier
- 代码检查工具：ESLint。建议同时本地和全局安装此工具。对于共享的项目可以让拷贝自己版本的人可以遵循应用于项目的规则；全局则可以随时检查任何文件。

### 配置初始项目
详见文档。
#### 获取项目代码文件
详见文档
#### 安装我们的工具
详见文档
#### 配置我们的工具
分别配置`.prettierrc.json`和`.eslintrc.json`文件。详见文档

### 构建和转换工具
我们将使用Parcel，Parcel将处理安装任何所需的转换工具和配置，在大多数情况下不需要我们干预。详见文档

#### 使用现代特性
我们使用 [CSS 嵌套](https://drafts.csswg.org/css-nesting/)，而不是像 [Sass](https://sass-lang.com/) 这样的工具。Parcel使用 [PostCSS](https://postcss.org/)在嵌套CSS和本地支持的CSS之间转换。因此本项目需要包含一个PostCSS插件，这里使用[postcss-preset-env](https://preset-env.cssdb.org/)。详见文档

#### 构建和运行
```bash
npx parcel src/index.html
```
详见文档

## Deploying our app (en-US)
最后一部分，采用前面文章的example toolchain然后部署我们的sample app。我们将代码推送到github，使用Netfily进行部署，甚至展示怎样添加一个简单的测试。

### Post Development
在项目的该阶段有大量潜在问题需要解决，创建一个toolchain减少手工干预是很重要的。这里是一些需要为该项目考虑的一些事情：
- 生成一个production build：确保文件最小化，chunked, 应用了tree-shaking，并且版本是"cache busted"。
- 运行测试：确保失败的测试可以组织部署
- 实际将更新的代码部署到一个live URL。
> cache busting是一种破坏浏览器caching机制的策略，强制浏览器下载一个你的代码的新的copy。Parcel（和许多其它工具）会对每个新的build生成独一无二的文件名。这个独特的文件名会打断浏览器的cache，因此确保浏览器会在每次更新代码时下载最新的代码。

本项目将使用Netlify，Netlify提供hosting，即提供一个URL来在线查看我们的项目，从而可以和他人分享。
尽管Netlify提供 [drag and drop deployment service](https://app.netlify.com/drop)，我们打算在每次推送到一个Github repo时在Netlify触发一个新的部署。
我们可以提交我们的代码然后推送到Github，更新的代码会自动触发整个build routine，我们唯一需要做的就是"push"。

### The build process
运行`npx parcel build src/index.html`，Parcel可以构建产品，而不是只是运行代码用于开发和测试。
新创建的production code位于`dist`目录，包含所有的运行网站的文件，可以上传到服务器。
但是手动构建代码不是我们的目标，我们想做的事让构建自动进行并且`dist`目录中的结果部署在我们的网站上。
我们code所在的地方，GitHub和Netlify需要彼此沟通，这样每次更新Github Code仓库时，Netlify会自动挑选changes，运行build任务，最终发布一个更新。
我们将添加build命令到`package.json`作为npm script，这步补充必须的，但是是配置的最佳实践。在所有的项目中，我们都可以依赖`npm run build`来做完整的build步骤，而不需要为每个项目记住特定的构建参数。
1. 打开`package.json`文件，找到`scripts` property。
2. 添加`build`命令：
```json
"scripts": {
	//...
    "build": "parcel build src/index.html"
},
```
3. 现在可以通过在根目录运行`npm run build`完成production build步骤。

### Committing changes to GitHub
本节将让你在一个git仓库中存储代码，但它并不是一个git教程。更详细的git教程见 [Git and GitHub](https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/GitHub)。
```bash
git status
git add ./
git commit -m 'committing initial code'
git remote add github https://github.com/Tom89757/super-website.git
git push github main 
```
详见文档

### Using Netlify for deployment
详见文档

### Testing
我们将展示如何添加一个initial test到项目中，以及如何使用该测试防止或允许项目部署进行。
有许多测试类型：
- End-to-end testing。
- Integration testing。
- Unit testing。
以及许多其它类型。测试可以作用于JavaScript，rendered DOM，用户交互，CSS甚至page外观。
本项目将创建一个简单的test来检查第三方的NASA data feed以确保它为正确的格式。如果测试失败，将会阻止项目存活。本测试不使用测试框架，但是有许多测试框架以供选择 [framework options](https://www.npmjs.com/search?q=keywords%3Atesting)。测试本身不是那么重要，重要的是如何处理测试成功和失败的情况。
Netlify只询问build命令，所以需要使测试称为build的一部分。如果测试失败，build也失败，Netlify不会进行项目部署。
测试添加详见文档。


## Bug汇总
运行`npx parcel src/index.html`后出现`plugin is not a function`构建错误：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230523225301.png)
解决方案：
```bash
npm install --save-dev postcss-preset-env@6.7.0
```
> 参考资料：
> 1. [javascript - index.css:undefined:undefined: plugin is not a function - Stack Overflow](https://stackoverflow.com/questions/70884769/index-cssundefinedundefined-plugin-is-not-a-function)