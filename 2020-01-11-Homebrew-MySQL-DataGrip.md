---
title: brew+MySQL+DataGrip
date: 2020-01-11 21:31:50
category:
- Tips
tags:
- Ubuntu
- Mac
- Linuxbrew
- Homebrew
- MySQL
- DataGrip
- HeidiSQL
---

参考：

> [DataGrip](https://www.jetbrains.com/datagrip/)

## 1. brew安装MySQL

brew安装软件之前可以先执行`brew search XXX`查看brew仓库是否存在此软件，在安装MySQL之前我们先搜索一下

```shell
ubuntu@VM-0-9-ubuntu ~ brew search mysql
==> Formulae
automysqlbackup               mysql-client@5.7              mysql-search-replace
mysql                         mysql-connector-c++           mysql@5.6
mysql++                       mysql-connector-c++@1.1       mysql@5.7
mysql-client                  mysql-sandbox                 mysqltuner
```

可以发现是没问题的，所以执行`brew install mysql`（如果brew安装很慢，则使用`sudo apt install mysql-server`安装），这里不指定版本号即默认安装最新版。在此过程中会自动安装MySQL的依赖库，默认情况下这些依赖库是只能被brew安装的软件使用的，如果你需要从其他位置使用brew提供的依赖库需要手动export这些库的路径到`.zshrc`中（不export也是可以的），安装完成后显示以下内容

```shell
==> mysql
We've installed your MySQL database without a root password. To secure it run:
    mysql_secure_installation

MySQL is configured to only allow connections from localhost by default

To connect run:
    mysql -uroot

Warning: mysql provides a launchd plist which can only be used on macOS!
You can manually execute the service instead with:
  mysql.server start
```

在Ubuntu上只能通过`mysql.server start`启动，如果在macOS上可以通过`brew services start mysql`启动，**在使用MySQL之前先启动MySQL服务**，再按照上面的提示完成一些设置`mysql_secure_installation`，具体会涉及到几个选项，需要设置MySQL root用户的密码，以及允许从远端连接MySQL（仅供测试Ubuntu服务器，本机macOS可以禁止）等等。

```shell
ubuntu@VM-0-9-ubuntu ~ mysql.server start
Starting MySQL
.. * 
 ubuntu@VM-0-9-ubuntu ~ mysql_secure_installation

Securing the MySQL server deployment.

Connecting to MySQL using a blank password.

VALIDATE PASSWORD COMPONENT can be used to test passwords
and improve security. It checks the strength of password
and allows the users to set only those passwords which are
secure enough. Would you like to setup VALIDATE PASSWORD component?

Press y|Y for Yes, any other key for No: n
Please set the password for root here.

New password: 

Re-enter new password: 
By default, a MySQL installation has an anonymous user,
allowing anyone to log into MySQL without having to have
a user account created for them. This is intended only for
testing, and to make the installation go a bit smoother.
You should remove them before moving into a production
environment.

Remove anonymous users? (Press y|Y for Yes, any other key for No) : y
Success.


Normally, root should only be allowed to connect from
'localhost'. This ensures that someone cannot guess at
the root password from the network.

Disallow root login remotely? (Press y|Y for Yes, any other key for No) : n

 ... skipping.
By default, MySQL comes with a database named 'test' that
anyone can access. This is also intended only for testing,
and should be removed before moving into a production
environment.


Remove test database and access to it? (Press y|Y for Yes, any other key for No) : y
 - Dropping test database...
Success.

 - Removing privileges on test database...
Success.

Reloading the privilege tables will ensure that all changes
made so far will take effect immediately.

Reload privilege tables now? (Press y|Y for Yes, any other key for No) : y
Success.

All done! 
```

使用`mysql -uroot -pXXXX`进入MySQL，`XXXX`为上面你设置的root用户密码，然后执行`show databases;`，就可以查看当前存在的数据库了

```shell
ubuntu@VM-0-9-ubuntu ~ mysql -uroot -pXXXX
mysql: [Warning] Using a password on the command line interface can be insecure.
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 8
Server version: 8.0.18 Homebrew

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
4 rows in set (0.01 sec)
```

## 2. 服务器安全组配置

如果需要远程连接服务器的MySQL数据库，那么需要将服务器的安全组配置一下，允许MySQL服务的端口被访问，MySQL默认端口号是3306

安全组分为两类：出站就是你访问外网；入站就是外网访问你。

如果我们需要访问服务器的3306端口，则入站规则配置如下：

| 来源      | 端口协议 | 策略 |
|-----------|----------|------|
| 0.0.0.0/0 | TCP:3306     | 允许 |

来源为`0.0.0.0/0`，表示允许所有IP访问服务器，也可以设置为指定的某些IP；端口协议指定允许被访问的端口，MySQL是`TCP:3306`，SSH是`TCP:22`， HTTP或HTTPS是`TCP:80`以及`TCP:443`等等。

## 3. MySQL创建新用户

默认的root用户权限非常大，需要谨慎使用，所以可以创建一个新的权限没有那么大的用户。需要先启动MySQL服务`mysql.server start`，然后命令行执行`mysql -uroot -pXXXX`进入MySQL，这里使用的是root用户，通过在root用户下创建新用户，`mysqlname`即我们指定的新用户名，`password`为该用户的密码

```shell
mysql> CREATE USER 'mysqlname'@'%' IDENTIFIED BY 'password';
Query OK, 0 rows affected (0.02 sec)
```

**然后需要进行授权，否则新用户无法进行其他操作**，`GRANT privileges ON databasename.tablename TO 'username'@'host'`，`privileges`代表一些比如`SELECT, INSERT`权限，你也可以设置为`ALL`，`databasename.tablename`为允许用户操作的数据库和表，谨慎使用`ALL`和通配符`*.*`

```shell
mysql> GRANT ALL ON *.* TO 'mysqlname'@'%';
Query OK, 0 rows affected (0.01 sec)
```


然后使用新用户登录MySQL执行`mysql -umysqlname -ppassword`，可以查看新用户可以操作的数据库，和root相同，实际并不推荐

```shell
 ubuntu@VM-0-9-ubuntu ~ mysql -umysqlname -ppassword
mysql: [Warning] Using a password on the command line interface can be insecure.
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 11
Server version: 8.0.18 Homebrew

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
1 row in set (0.01 sec)
```

然后尝试创建新的数据库，并添加一些表以及字段等等

```shell
# 创建数据库test
mysql> create database test;
Query OK, 1 row affected (0.01 sec)

# 使用test数据库
mysql> use test;
Database changed

# 创建user表，添加id、name字段
mysql> create table user (
    -> id int primary key auto_increment,
    -> name varchar(32)
    -> );
Query OK, 0 rows affected (0.04 sec)
```

添加数据并查看

```shell
mysql> insert into user (name) values ('Sam');
Query OK, 1 row affected (0.01 sec)

mysql> select * from user;
+----+------+
| id | name |
+----+------+
|  1 | Sam  |
+----+------+
1 row in set (0.00 sec)
```

## 4. 使用数据库软件连接服务器MySQL

命令行操作数据库有点不方便，可以使用一些数据库软件连接数据库并进行操作，因为之前已经设置了安全组，所以可以访问服务器上的数据库（如果数据库在本机，原理基本相同，基本上只需要修改host为localhost或者127.0.0.1）。

比较好用的数据库软件是DataGrip，功能很强大，支持的数据库类型也比较多，当然其他开源数据库软件也都很好用，比如Windows上的HeidiSQL。

### 4.1 DataGrip连接



### 4.2 HeidiSQL连接

HeidiSQL连接Ubuntu服务器上的MySQL，配置如下：

| 网络类型                       | 主机名/IP     | 用户名/密码           |    端口              |       
|------------------------------|-------------|------------------------|---------------------|
| MariaDB or MySQL(SSH tunnel) | 127.0.0.1   | 服务器上创建的新用户/密码  | 服务器MySQL服务端口号  |

{% asset_img hei1.png %}

SSH隧道配置如下：

| plink.exe位置 | SSH主机+端口    | 用户名/密码     |
|---------------|-----------------|-----------------|
| 自行下载路径  | 服务器公网IP+22 | 服务器用户/密码 |

{% asset_img hei2.png %}

连接成功后的结果如下，可以看到我们之前添加的数据库以及表数据

{% asset_img hei3.png %}