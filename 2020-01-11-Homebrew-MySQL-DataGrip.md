---
title: brew+MySQL+DataGrip
date: 2020-01-11 21:31:50
- Tips
tags:
- Ubuntu
- Mac
- Linuxbrew
- Homebrew
- MySQL
- DataGrip
---


## brew安装MySQL

brew安装软件之前可以先执行`brew search XXX`查看brew仓库是否存在此软件，在安装MySQL之前我们先搜索一下

```shell
ubuntu@VM-0-9-ubuntu ~ brew search mysql
==> Formulae
automysqlbackup               mysql-client@5.7              mysql-search-replace
mysql                         mysql-connector-c++           mysql@5.6
mysql++                       mysql-connector-c++@1.1       mysql@5.7
mysql-client                  mysql-sandbox                 mysqltuner
```

可以发现是没问题的，所以执行`brew install mysql`，这里不指定版本号即默认安装最新版。在此过程中会自动安装MySQL的依赖库，默认情况下这些依赖库是只能被brew安装的软件使用的，如果你需要从其他位置使用brew提供的依赖库需要手动export这些库的路径到`.zshrc`中（不export也是可以的）






