---
title: hexo博客图床配置
date: 2022-05-08 20:24:35
categories:
- 环境配置
tags:
- hexo博客
---

本文记录一下如何将本地`.md`文件中图片上传到 github 仓库或腾讯云对象存储中，并在本地`.md`文件中将图片地址替换为对应的 github 或腾讯云存储链接。

<!--more-->

### 上传至 github 仓库

该方法的思路是在 github 中新建一个仓库（必须为public仓库，否则无法访问）作为图床，并通过 PicGo 软件将本地图片上传至对应的 github 仓库。具体步骤如下：

#### 准备工作

- Typora 0.11.18 (beta)：[下载地址](https://typora.en.uptodown.com/windows/versions)
- PicGo 2.3.0：[下载地址](https://github.com/Molunerfinn/PicGo/releases)
- nodejs v16.13.0

#### 软件配置

- 打开 PicGo 在图床配置里选择 GitHub 图床进行配置，并设为默认图床

  ![image-20220516220239293](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516220239293.png?token=AKWAGW46DCB4MTDZJIFCFR3CQJMZU)

- 打开 Typora 偏好设置并进行图像设置：

  ![image-20220516220428369](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516220428369.png?token=AKWAGW6CJAWAMQRBNJQZ4RLCQJNAO)

- 编辑 markdown 文档会发现插入文档的图片会自动上传至 github 仓库并将图片地址更新为对应 github 仓库地址。也可右键图像手动上传：

  ![image-20220516220638096](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220516220638096.png?token=AKWAGW6R34UTLRBW7F5KINLCQJNIS)
  

由于使用github图床，有时上传时会发生如下的上传错误：

```python
------Error Stack Begin------
RequestError: Error: read ECONNRESET
at new RequestError (D:\Program Files\PicGo\resources\app.asar\node_modules\request-promise-core\lib\errors.js:14:15)
at Request.plumbing.callback (D:\Program Files\PicGo\resources\app.asar\node_modules\request-promise-core\lib\plumbing.js:87:29)
at Request.RP$callback [as _callback] (D:\Program Files\PicGo\resources\app.asar\node_modules\request-promise-core\lib\plumbing.js:46:31)
at self.callback (D:\Program Files\PicGo\resources\app.asar\node_modules\request\request.js:185:22)
at Request.emit (events.js:200:13)
at Request.onRequestError (D:\Program Files\PicGo\resources\app.asar\node_modules\request\request.js:881:8)
at ClientRequest.emit (events.js:200:13)
at TLSSocket.socketErrorListener (_http_client.js:402:9)
at TLSSocket.emit (events.js:200:13)
at emitErrorNT (internal/streams/destroy.js:91:8)
-------Error Stack End-------
```

其原因为无法请求github服务器。在使用V2rayN的情况下，直接在PicGO中配置如下代理和镜像地址：

![image-20220804212205450](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220804212205450.png)

会导致如下报错：

```python
------Error Stack Begin------
RequestError: Error: tunneling socket could not be established, cause=socket hang up
    at new RequestError (D:\Develop\PicGo\resources\app.asar\node_modules\request-promise-core\lib\errors.js:14:15)
    at Request.plumbing.callback (D:\Develop\PicGo\resources\app.asar\node_modules\request-promise-core\lib\plumbing.js:87:29)
    at Request.RP$callback [as _callback] (D:\Develop\PicGo\resources\app.asar\node_modules\request-promise-core\lib\plumbing.js:46:31)
    at self.callback (D:\Develop\PicGo\resources\app.asar\node_modules\request\request.js:185:22)
    at Request.emit (node:events:394:28)
    at Request.onRequestError (D:\Develop\PicGo\resources\app.asar\node_modules\request\request.js:877:8)
    at ClientRequest.emit (node:events:394:28)
    at ClientRequest.onError (D:\Develop\PicGo\resources\app.asar\node_modules\tunnel-agent\index.js:179:21)
    at Object.onceWrapper (node:events:514:26)
    at ClientRequest.emit (node:events:394:28)
-------Error Stack End------- 
2022-08-04 21:02:55 [PicGo ERROR] 
```

其原因在于PicGO只支持http协议的代理，而V2rayN只支持socks5协议代理。故应将代理软件替换为支持http协议的软件，有两种选择ShadowsocksR和Clash，我选择了Clash。更改代理软件并设置相应端口后会发现上传成功。

> 参考资料：
> 1. [为Typora配置图床,实现图片自动上传](https://www.jianshu.com/p/4740993c5843)
> 2. [PicGo+GitHub图床配置&常见错误 - Eighty Percent](http://b.aksy.space/study-notes/514.html)

### 上传至腾讯云对象存储

