---
title: 虚拟机上CentOS7系统网络配置
date: 2021-12-11 19:28:15
categories:
- 环境配置
tags: 
- 虚拟机
- CentOS7
- 网络
---

本文记录一下在Virtual Box上安装的CentOS7系统上配置网络的过程，以便后续查阅。

<!--more-->


### 查看当前网络配置

在安装好CentOS7系统后，打开终端运行：
```
ping www.baidu.com
ping 192.168.*.*    //主机地址
```
会发现ping不通。

同样，通过 `ifconfig` 查看会发现还未给虚拟机分配Ipv4地址。

在进一步设置之前，需要在虚拟机设置里将网络设置为：Bridged Adapter + WiFi

### 为虚拟机分配IP地址

在此，我们通过 `dhclient` 来为虚拟机分配一个静态的IP地址（需要切换到root模式）

通过 `ifconfig` 可以查看给虚拟机分配的IP地址，并记录下来。

然后，通过
```
vim /etc/sysconfig/network-scripts/ifcfg-enp0s3
```
对网络配置文件进行编辑。需要修改的内容如下：
```
BOOTPROTO=static
ONBOOT=yes
IPADDR=192.168.123.*    //前面分配的IP地址
NETMASK=255.255.255.0
GATEWAY=192.168.123.1
DNS1=119.29.29.29
```
在将上述文件保存后，通过：
```
systemctl restart network.service
```
重启网络服务使上述配置生效。



### 检查配置是否生效
此时，通过
```
ping www.baidu.com
ping 192.168.*.*    //主机地址
```
会发现可以ping通。

