---
title: Mac+Anaconda+PyCharm/VSCode环境搭建
date: 2018-10-13 10:44:00
categories:
- tips
tags:
- anaconda
- pycharm
- python
- vscode
mathjax: true
---

# 令人愉悦的Python开发环境搭建

## 1. 系统选择

* macOS
* Windows
* Linux
  
写代码我选macOS，黑苹果或者apple都可以。区别：

* 黑苹果性能可控，配置自定义，可以比apple官网上强不少而且便宜，缺点是驱动配置很扎心，系统更新很难受，适合搞机党。
* apple官方产品造型好，系统环境稳定，接近linux的开发环境，缺点是配置固定，GPU辣鸡，贵死。

## 2. Python环境

macOS自带的应该是python2.7版本，理论上可以直接跑代码：

```python
python filename.py
```

但是初始环境的第三方库不多，需要自行安装，但是由于涉及到本机系统环境，我不推荐在本机环境装库，为什么呢，等你系统崩溃就凉了，所以可以使用一个好东西--虚拟环境。

在介绍虚拟环境之前我们需要知道python版本兼容问题：

python目前的版本有两种2.x和3.x，为什么有两个版本呢，而且为什么要强调有这么两个版本呢，那是因为这两个版本**不兼容**，哈哈哈。3.x中对2.x的部分内容进行了修改优化，比如说可以在2.x的python版本下运行的代码，到3.x中**不一定**能运行，而且有的第三方库只支持3.x，比如Pytorch。所以开发时需要提前确定我们需要哪个版本的Python环境。

这时候就有必要了解虚拟环境了，这是个好东西啊！

<!-- more -->

## 3. 虚拟环境+Anaconda

虚拟环境类似虚拟机，但是不是虚拟机，它只是提供一个开发环境，硬件还是本机的硬件，不过你可在虚拟环境里面管理虚拟环境中的第三方库，虚拟环境中的库与本机的库是分离开的，互不影响，也就是说，比如你本机有Python2.7，但是虚拟环境中可以装python3.6，在虚拟环境中跑代码使用的就是python3.6，是不是就解决了python版本问题。

[Anaconda](https://www.anaconda.com/)是一个可以创建并管理虚拟环境的工具，通过anaconda创建的虚拟环境默认就包括了很多机器学习需要的第三方库，所以我推荐使用anaconda，当然你也可以使用其他的工具（virtualenv...）

### 1. 下载
[Anaconda官网下载地址macOS](https://www.anaconda.com/download/#macos)

为什么这里也有两个版本呢，那是因为Anaconda安装时可以覆盖本机的python环境软连接，所以如果macOS本机为2.7，下载安装Anaconda3，最后你在命令行中运行python的时候启动的是python3而不是python2。下载安装包有两种，一个是带图形界面的Anaconda，一个是只有命令行的Anaconda，新手选带GUI的。

<!-- 
![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda01.png) -->

{% asset_img conda01.png Anaconda %}


### 2. 安装
得到Anaconda2-5.2.0-MacOSX-x86_64的安装包，然后就是正常的安装过程，你会得到

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda02.png) -->

{% asset_img conda02.png Anaconda %}


通过这个软件可以管理虚拟环境

{% asset_img conda03.png Anaconda %}

{% asset_img conda04.png Anaconda %}

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda03.png) -->

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda04.png) -->

### 3. 验证
验证Anaconda环境，在命令行输入
   
```
conda
```
输出下面内容，说明安装成功

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda05.png) -->
{% asset_img conda05.png Anaconda %}


### 4. 镜像源
添加镜像源，为了更快速的下载第三方库，参考[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

### 5. 虚拟环境
创建虚拟环境，可以使用命令，env_name为你的虚拟环境的命名，X.X为python版本（2.7/3.7/3.7/...），也可以在图形界面中创建，上面的图中已经显示了我创建的两个虚拟环境pytorch和tensorflow，分别是3.6和2.7版本的python。

```python
conda create -n env_name python=X.X
```

{% asset_img conda06.png Anaconda %}

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda06.png) -->

### 6. 激活虚拟环境
在命令行中，env_name为你的虚拟环境的名字

```python
source activate env_name
```

成功激活后，命令行变成

```python
(tensorflow) Tao:~ zhoutao$ 
```

前面括号的内容就是你的虚拟环境的名字，这时候就可以在虚拟环境中搞事情了，比如在虚拟环境中安装第三方库，lib_name为库的名字，你也可以先search一下，或者在第三方库的官网上查看conda下载的命令，注意在虚拟环境中install才是虚拟环境中的库，也可以在GUI中安装（注意，macOS可能会遇到权限不够的问题，会提示你安装库失败，这时候需要chmod 777给权限anaconda文件夹，anaconda文件夹在我的用户根目录下），有的库只有pip能安装，不过也没问题

```python
conda install lib_name
pip install lib_name
```

关闭虚拟环境，命令行输入，不用指定env_name，因为你已经在虚拟环境中了

```python
source deactivate
```

但是在命令行中写代码不符合我的懒癌特征，我需要一个IDE，python开发的IDE我用了PyCharm和VSCode，下面就介绍这两个IDE与虚拟环境怎么连接起来。

## 4.1 [PyCharm](http://www.jetbrains.com/pycharm/)安装与配置

版本Professional Edition和Community Edition

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda07.png) -->
{% asset_img conda07.png Anaconda %}


专业的收费，但是功能强劲；社区的免费但是功能少一些

一般来说社区版就够初学者用了，我觉得专业版最有用的是远程连接的功能（怎么用暂时先不告诉你，嘻嘻）

下载安装完成后你会得到

{% asset_img conda08.png Anaconda %}

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda08.png) -->

### 来吧，开始配置相关环境了

### 1. 创建项目
创建一个项目tf_learn（名字你随意），选择interpreter，这里第一次配置下面的existing interpreter应该是空的，所以需要找到前面我们创建虚拟环境的位置

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda09.png) -->
{% asset_img conda09.png Anaconda %}


看路径，一般是你的anaconda目录下，env/env_name/bin/python，最后只要找到这个python就行了

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda10.png) -->
{% asset_img conda10.png Anaconda %}


### 2. 安装库
安装你需要的第三方库，跟上面提到的安装方式不同，但是效果最终都是一样的（在虚拟环境中添加第三方库），在preference中找到project interpreter，这里你就可以发现加载出来你的虚拟环境中的第三方库了，点击加号，搜索库的名字，接着安装就行了。

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda11.png) -->
{% asset_img conda11.png Anaconda %}

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda12.png) -->
{% asset_img conda12.png Anaconda %}

### 3. 运行配置
跑代码的设置，先创建一个py文件，随便写点东西，比如我刚刚安装了tensorflow的库，要验证一下是否安装成功，那么import一下试试。然后配置，点击右上角的add configurations，这里有几个模板，你可以配置一下供以后用，参数自己了解一下，我们创建自己的配置内容。

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda13.png) -->
{% asset_img conda13.png Anaconda %}

script path表明你需要跑的代码，working directory表明你的项目的工作目录，读取文件的时候需要用，随便命个名，其他暂时不用变。

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda14.png) -->
{% asset_img conda14.png Anaconda %}

代码内容如下，随便输出点东西就知道tensorflow安装成功，所以大功告成。

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda15.png) -->
{% asset_img conda15.png Anaconda %}

### 4. 运行
现在已经可以在虚拟环境中跑python代码了，还有一个好东西可以用，那就是jupyter notebook，因为anaconda默认安装jupyter库，所以简单在configuration中配置一下就可以跑，additional options指定额外的参数，这里我加了**--no-browser**，这是为了之后开启服务时不弹出浏览器窗口，同样设置一下工作目录，命个名

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda16.png) -->
{% asset_img conda16.png Anaconda %}

拷贝一下代码测试一下，首先点击右上角**run jupyter**，这是为了开启jupyter服务

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda17.png) -->
{% asset_img conda17.png Anaconda %}

然后点击代码矿上面的**run cell**，这样就可以跑这一个cell中的代码了

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda18.png) -->
{% asset_img conda18.png Anaconda %}

结果，在cell下面显示了‘hello’，完美

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda19.png) -->
{% asset_img conda19.png Anaconda %}

## 4.2 [VSCode](https://code.visualstudio.com/)安装与配置

下载安装完成后你会得到

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda20.png) -->
{% asset_img conda20.png Anaconda %}

### 开始配置了

### 1. 下载插件

在商店中搜索并下载安装

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda22.png) -->
{% asset_img conda22.png Anaconda %}

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda23.png) -->
{% asset_img conda23.png Anaconda %}

### 2. 配置用户环境

只要python.pythonPath和python.venvPath，内容和PyCharm下的差不多，也是找到虚拟环境中的python，以及虚拟环境的目录

<!-- ![image](Mac+Anaconda+PyCharm+VSCode环境搭建/conda24.png) -->
{% asset_img conda24.png Anaconda %}

### 3. 配置configuration
添加python配置就完事了

# 相信你一定会成功的，我就不展示了，嘻嘻