---
title: IDEA常见bug
date: 2023-08-13 22:42:27
categories:
- 环境配置
tags:
- IDEA
---

本文记录一下在IDEA中一些常见的bug：
<!--more-->

### [IntelliJ: Error:java: error: release version 5 not supported](https://stackoverflow.com/questions/59601077/intellij-errorjava-error-release-version-5-not-supported)
解决方案：
重新选择jdk版本或maven
> 参考资料：
> 1. [IntelliJ: Error:java: error: release version 5 not supported - Stack Overflow](https://stackoverflow.com/questions/59601077/intellij-errorjava-error-release-version-5-not-supported)

###  [Error: Could not find or load main class in intelliJ IDE](https://stackoverflow.com/questions/10654120/error-could-not-find-or-load-main-class-in-intellij-ide)
原因：在更改项目后，IDEA build应用时未覆盖同名文件，导致更改未生效
解决方案：见参考资料
> 参考资料：
> 1. [java - Error: Could not find or load main class in intelliJ IDE - Stack Overflow](https://stackoverflow.com/questions/10654120/error-could-not-find-or-load-main-class-in-intellij-ide)

### [Plugin 'org.springframework.boot:spring-boot-maven-plugin:' not found](https://stackoverflow.com/questions/64639836/plugin-org-springframework-bootspring-boot-maven-plugin-not-found)
在`pom.xml`中添加：
```xml
<version>${project.parent.version}</version>
```
> 参考资料：
> 1. [java - Plugin 'org.springframework.boot:spring-boot-maven-plugin:' not found - Stack Overflow](https://stackoverflow.com/questions/64639836/plugin-org-springframework-bootspring-boot-maven-plugin-not-found) 