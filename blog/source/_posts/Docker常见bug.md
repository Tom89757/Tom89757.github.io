---
title: Docker常见bug
date: 2023-08-09 20:59:19
categories:
- 开发工具
tags:
- Docker
---
本文记录一下Docker的常见bug：
<!--more-->

### 报错 `Builder lifecycle 'creator' failed with status code 51`

> 参考资料：
> 1. [Cloud Native Buildpacks/Paketo with Java/Spring Boot: How to configure different JDK download uri (e.g. no access to github.com) - Stack Overflow](https://stackoverflow.com/questions/65212231/cloud-native-buildpacks-paketo-with-java-spring-boot-how-to-configure-different)
> 2. [落叶飘](https://www.lmm.show/24/)：有对应gitee仓库
> 3. [https://raw.githubusercontent.com/paketo-buildpacks/bellsoft-liberica/main/buildpack.toml](https://raw.githubusercontent.com/paketo-buildpacks/bellsoft-liberica/main/buildpack.toml)
> 4. [How To Configure Paketo Buildpacks - Paketo Buildpacks](https://paketo.io/docs/howto/configuration/)