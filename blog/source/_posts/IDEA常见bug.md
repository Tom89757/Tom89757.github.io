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


### Could not autowire. No beans of 'BookRepository' type found

> 参考资料：
> 1. [java - intellij incorrectly saying no beans of type found for autowired repository - Stack Overflow](https://stackoverflow.com/questions/26889970/intellij-incorrectly-saying-no-beans-of-type-found-for-autowired-repository)
> 2. [https://youtrack.jetbrains.com/issue/IDEA-281454/Could-not-autowire.-No-beans-of-XXX-type-found-IntelliJ-2021.2.3-Spring-2.5.6](https://youtrack.jetbrains.com/issue/IDEA-281454/Could-not-autowire.-No-beans-of-XXX-type-found-IntelliJ-2021.2.3-Spring-2.5.6)

### bad SQL grammar [INSERT INTO order (book_isbn, quantity, status, created_date, last_modified_date, version)

GitHub Copilot: 这个错误可能是因为 `order` 是 SQL 中的保留关键字，不能用作表名。您可以尝试将表名改为其他名称，例如 `orders`，并更新相应的 SQL 语句。
如果您使用的是 Spring Data JPA，可以在实体类上使用 `@Table` 注解来指定表名，例如：
```java
@Entity
@Table(name = "orders")
public class Order {
    // ...
}
```
如果您使用的是 Spring Data R2DBC，可以在 `application.properties` 文件中配置表名，例如：
```
spring.r2dbc.url=r2dbc:postgresql://localhost:5432/mydatabase
spring.r2dbc.username=myusername
spring.r2dbc.password=mypassword
spring.r2dbc.initialization-mode=always
spring.r2dbc.sql-client.init-location=classpath:/schema.sql
spring.r2dbc.sql-client.init-mode=embedded
spring.r2dbc.sql-client.name=orders
```
在上面的配置中，`spring.r2dbc.sql-client.name` 属性指定了表名为 `orders`。您需要将其替换为您实际使用的表名。

### bad SQL grammar [INSERT INTO orders (book_isbn, quantity, status, created_date, last_modified_date, version) 
[42P01] relation "orders" does not exist
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230825155514.png)

### 报错Cannot resolve symbol "'StreamBridge'"
原因：Spring Cloud和Spring boot版本不匹配
解决方案：更改build.gradle使得二者版本匹配
> 参考资料：
> 1. [StreamBridge bean not found in Spring context - Stack Overflow](https://stackoverflow.com/questions/75367370/streambridge-bean-not-found-in-spring-context)