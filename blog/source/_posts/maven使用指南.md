---
title: maven使用指南
date: 2023-08-13 19:34:52
categories:
- 开发工具
tags:
- maven 
---

本文记录一下maven的使用指南：
<!--more-->

### ./mvnw spring-boot:run运行失败
报错`Caused by: org.apache.maven.plugin.MojoExecutionException: Process terminated with exit code: 1`
在Application.class中添加异常处理：
由
```java
SpringApplication.run(CatalogServiceApplication.class, args);
```
改为：
```java
try {
	SpringApplication.run(CatalogServiceApplication.class, args);
} catch (Exception e) {
	e.printStackTrace();
}
```
> 参考资料：
> 1. [tomcat - Process finished with exit code 1 Spring Boot Intellij - Stack Overflow](https://stackoverflow.com/questions/46428611/process-finished-with-exit-code-1-spring-boot-intellij)

### ./mvnw spring-boot:build-image运行失败
报错`Caused by: org.apache.maven.plugin.PluginExecutionException: Execution default-cli of goal org.springframework.boot:spring-boot-maven-plugin:3.1.2:build-image failed: Docker API call to 'localhost/v1.24/images/docker.io/paketobuildpacks/builder:base/json' failed with status code 404 "Not Found"`

> 参考资料：
> 1. [Use a local builder image when the image does not exist in the remote repository · Issue #25831 · spring-projects/spring-boot · GitHub](https://github.com/spring-projects/spring-boot/issues/25831)
> 2. [gradle - Use Spring Native with custom Docker registry - Stack Overflow](https://stackoverflow.com/questions/72095321/use-spring-native-with-custom-docker-registry)