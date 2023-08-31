---
title: Kubernetes使用指南
date: 2023-08-15 10:43:51
categories:
- 开发工具
tags:
- Kubernetes 
---
本文记录一下Kubernetes的使用指南：
<!--more-->

### 报错 Unable to resolve the current Docker CLI context "default"
根据参考资料2运行`docker context use default`。
> 参考资料：
> 1. [All docker cli commands fail: "unable to resolve docker endpoint" · Issue #3790 · docker/for-mac · GitHub](https://github.com/docker/for-mac/issues/3790)
> 2. [Minikube v1.30.1 unable to resolve current docker CLI context after upgrade on Ubuntu 22.04 LTS · Issue #16788 · kubernetes/minikube · GitHub](https://github.com/kubernetes/minikube/issues/16788)


### 启用Ingress addons插件时报错 Exiting due to MK_ENABLE: run callbacks: running callbacks
```minikube
minikube delete --all
choco install minikube //更新minikube
minikube start --cpus 2 --memory 4g --driver docker --profile polar
minikube addons enable ingress --profile polar
kubectl get all -n ingress-nginx
```
> 参考资料：
> 1. [kubernetes - Problem with minikube and nginx ingress when reinstalled minikube - Stack Overflow](https://stackoverflow.com/questions/67194692/problem-with-minikube-and-nginx-ingress-when-reinstalled-minikube)
> 2. [minikube start | minikube](https://minikube.sigs.k8s.io/docs/start/)