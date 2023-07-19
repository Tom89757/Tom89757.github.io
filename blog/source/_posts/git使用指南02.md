---
title: git使用指南02
date: 2023-06-15 23:44:08
categories:
- 开发工具
tags:
- git
---

本文记录一下使用git时的常见操作。
<!--more-->

1.git更新版本并添加到Windows Terminal：

> 参考资料：
> 1. [How to Upgrade git to the Latest Version on Windows](https://linuxhint.com/upgrade-git-latest-version-windows/)
> 2. [Adding Git-Bash to the new Windows Terminal - Stack Overflow](https://stackoverflow.com/questions/56839307/adding-git-bash-to-the-new-windows-terminal)

</br>

2.git terminal在VSCode中行为很奇怪：
更新git版本。
> 参考资料：
> 1. [git bash terminal is acting wierd in vscode · Issue #184719 · microsoft/vscode · GitHub](https://github.com/microsoft/vscode/issues/184719)
> 2. [visual studio code - git bash terminal is acting wierd in vscode - Stack Overflow](https://stackoverflow.com/questions/76479076/git-bash-terminal-is-acting-wierd-in-vscode)


</br>
3.git bash终端中文显示乱码：
解决方案1（无效）：
- `git config --global core.quotepath false`
- 在设置里将`text`改为`zh-CN`和`UTF-8`。
解决方案2：
- `chcp.com 65001`：`65001`对应UTF-8，可以写在`.zshrc`文件中。
> 参考资料：
> 1. [Git Bash终端中文输出显示乱码解决方案 - lybingyu - 博客园](https://www.cnblogs.com/sdlz/p/13023342.html)
> 2. [windows - Unicode (utf-8) with git-bash - Stack Overflow](https://stackoverflow.com/questions/10651975/unicode-utf-8-with-git-bash)

</br>
4.git中创建pull request
> 参考资料：
> 1. [How to Code Good Part 2 - Google 幻灯片](https://docs.google.com/presentation/d/15WY-h0lr-t3jDO0eJTaeBq5zbnwfjPxPr1aauI1xhF4/edit#slide=id.g1f8707ec125_0_4)

</br>
5.git仓库fork后创建pull request
Github Copilot：如果您想将自己的更改贡献给原始仓库，可以通过 Pull Request 的方式向原始仓库提交更改请求。下面是一些基本步骤：
1. 在 GitHub 上 Fork 原始仓库，将其复制到自己的账户中。
2. 在自己的账户中 Clone Fork 后的仓库到本地。
3. 在本地仓库中创建一个新的分支，并在该分支中进行代码更改。
4. 将更改推送到自己的 GitHub 仓库中。
5. 在自己的 GitHub 仓库中创建一个 Pull Request，请求将更改合并到原始仓库中。
6. 在 Pull Request 中描述您所做的更改、为什么要做这些更改以及如何测试这些更改。
7. 等待原始仓库的维护者审核并合并您的更改。
请注意，不同的项目可能有不同的贡献流程和规则。在参与开源项目时，建议仔细阅读项目的贡献指南和代码贡献流程，以了解项目的贡献规则和流程。

</br>
6.本仓库pull request和fork后pull request的区别：
GitHub Copilot: Fork 和 Pull Request 都是 GitHub 中用于协作开发的重要机制。
Fork 是指将一个开源项目的代码库复制到自己的 GitHub 账户中，以便进行修改和贡献。通过 Fork，开发人员可以在自己的分支中进行代码更改，并将这些更改推送到自己的 GitHub 仓库中。这样，他们就可以在自己的仓库中管理代码，并随时将更改推送到原始项目中。
Pull Request 是指开发人员在自己的分支中进行代码更改后，向原始项目的维护者请求审核和合并这些更改的机制。通过 Pull Request，开发人员可以描述他们所做的更改、为什么要做这些更改以及如何测试这些更改。项目维护者可以查看 Pull Request，并在必要时提出问题或建议修改。一旦 Pull Request 被审核通过，项目维护者可以将更改合并到项目的主分支中。
总的来说，Fork 和 Pull Request 都是用于协作开发的机制，但它们的使用场景和目的有所不同。Fork 通常用于在自己的分支中进行代码更改和管理，而 Pull Request 通常用于向原始项目贡献代码并请求审核和合并。

</br>
7.`git pull origin main`报错`fetch-pack: unexpected disconnect while reading sideband packet`

> 参考资料：
> 1. [git - error: RPC failed; curl transfer closed with outstanding read data remaining - Stack Overflow](https://stackoverflow.com/questions/38618885/error-rpc-failed-curl-transfer-closed-with-outstanding-read-data-remaining)
> 2. [git - Github - unexpected disconnect while reading sideband packet - Stack Overflow](https://stackoverflow.com/questions/66366582/github-unexpected-disconnect-while-reading-sideband-packet)

</br>
8.在local git repository（本地git仓库）中同时在两个分支上工作
```git
gb backend # 切换到backend分支
git worktee add frontend # 在当前worktree添加frontend分支
```
此时目录结构如下：

> 参考资料：
> 1. [github - i want to work on different branches in git in my local repository which have different branches from origin set as upstream - Stack Overflow](https://stackoverflow.com/questions/59579681/i-want-to-work-on-different-branches-in-git-in-my-local-repository-which-have-di)
> 2. [Multiple Branches in Git ⋆ Mark McDonnell](https://www.integralist.co.uk/posts/multiple-branches-in-git/#:~:text=Git%20offers%20a%20feature%20referred,directories%20where%20they%20are%20stored.)
> 3. [Git's Best And Most Unknown Feature - YouTube](https://www.youtube.com/watch?v=2uEqYw-N8uE)

</br>
9.`git rebase`操作

> 参考资料：
> 1. [The Git Rebase Handbook – A Definitive Guide to Rebasing](https://www.freecodecamp.org/news/git-rebase-handbook/)