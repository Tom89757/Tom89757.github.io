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
