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
原因：在构建image的过程中，使用代理下载github上的依赖项。这可能导致错误。详见资料5
解决方案：

> 参考资料：
> 1. [Cloud Native Buildpacks/Paketo with Java/Spring Boot: How to configure different JDK download uri (e.g. no access to github.com) - Stack Overflow](https://stackoverflow.com/questions/65212231/cloud-native-buildpacks-paketo-with-java-spring-boot-how-to-configure-different)
> 2. [落叶飘](https://www.lmm.show/24/)：有对应gitee仓库
> 3. [https://raw.githubusercontent.com/paketo-buildpacks/bellsoft-liberica/main/buildpack.toml](https://raw.githubusercontent.com/paketo-buildpacks/bellsoft-liberica/main/buildpack.toml)
> 4. [How To Configure Paketo Buildpacks - Paketo Buildpacks](https://paketo.io/docs/howto/configuration/)
> 5. [github bellsoft-jre17.0.5+8-linux-amd64.tar.gz x509: certificate signed by unknown authority · Issue #353 · paketo-buildpacks/bellsoft-liberica · GitHub](https://github.com/paketo-buildpacks/bellsoft-liberica/issues/353)
> 6. [docker - How to set dependency-mapping binding in gradle bootBuildImage (Spring-boot 2.7.1, native) - Stack Overflow](https://stackoverflow.com/questions/74399883/how-to-set-dependency-mapping-binding-in-gradle-bootbuildimage-spring-boot-2-7)
> 7. [Image building with newer Paketo base-platform-api-0.3 fails · Issue #23009 · spring-projects/spring-boot · GitHub](https://github.com/spring-projects/spring-boot/issues/23009)


### Docker一直处于"Docker is staring..."界面
以管理员身份打开powershell，运行：
```powershell
Enable-WindowsOptionalFeature -Online -FeatureName $("Microsoft-Hyper-V", "Containers") -All
```
重启电脑。
> 参考资料：
> 1. [Docker forever in "Docker is starting.." at Windows task - Stack Overflow](https://stackoverflow.com/questions/43041331/docker-forever-in-docker-is-starting-at-windows-task)
> 2. [Solve Problem Docker stucks and still in starting mode - YouTube](https://www.youtube.com/watch?v=JRluDLoiHXM&t=5s)
> 3. [c# - How to enable the Windows 10 "Containers" feature? - Stack Overflow](https://stackoverflow.com/questions/36590514/how-to-enable-the-windows-10-containers-feature)


### 运行`docker build`后出现no matching manifest for windows/amd64 in the manifest list entries
解决方案：右键docker图标Switching to Linux containers
> 参考资料：
> 1. [Docker: "no matching manifest for windows/amd64 in the manifest list entries" - Stack Overflow](https://stackoverflow.com/questions/48066994/docker-no-matching-manifest-for-windows-amd64-in-the-manifest-list-entries)