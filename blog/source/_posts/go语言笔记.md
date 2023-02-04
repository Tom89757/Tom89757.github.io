---
title: go语言笔记
date: 2022-05-15 23:15:43
categories:
- 笔记
tags: 
- golang
---

本文记录一下在学习 go 语言过程中遇到的问题。

<!--more-->

1.下述代码中`_ "github.com/go-sql-driver/mysql"`中的`_`的作用。

```go
import (
	"database/sql"
	"fmt"

	_ "github.com/go-sql-driver/mysql"
)
```

> 参考资料：
>
> 1. [What does an underscore in front of an import statement mean?](https://stackoverflow.com/questions/21220077/what-does-an-underscore-in-front-of-an-import-statement-mean)
> 2. [Import declarations](https://go.dev/ref/spec#Import_declarations)


</br>
2.镜像源配置：
在`.bashrc`文件中写入：
```bash
# 启用 Go Modules 功能
go env -w GO111MODULE=on

# 配置 GOPROXY 环境变量，以下三选一

# 1. 七牛 CDN
go env -w  GOPROXY=https://goproxy.cn,direct

# 2. 阿里云
go env -w GOPROXY=https://mirrors.aliyun.com/goproxy/,direct

# 3. 官方
go env -w  GOPROXY=https://goproxy.io,direct
```
> 参考资料：
> 1. [Go 国内加速：Go 国内加速镜像 | Go 技术论坛](https://learnku.com/go/wikis/38122)

</br>
3.Linux安装go
> 参考资料：
> 1. [Go 语言环境安装 | 菜鸟教程](https://www.runoob.com/go/go-environment.html)