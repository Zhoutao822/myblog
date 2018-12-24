---
title: 深度学习模型-KerasApplications
date: 2018-12-14 17:08:12
categories:
- Deep Learning
tags:
- Code
- Keras
- Model
mathjax: true
---

参考：

> [Keras Applications](https://github.com/keras-team/keras-applications)
> [Models for image classification with weights trained on ImageNet](https://keras.io/applications/#xception)
> [深度学习VGG模型核心拆解](https://blog.csdn.net/qq_40027052/article/details/79015827)
> [Deep Learning Papers Translation(CV)](https://github.com/SnailTyan/deep-learning-papers-translation)
> [ResNet论文翻译——中英文对照](https://blog.csdn.net/Quincuntial/article/details/77263607)
> [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)

ImageNet项目是一个用于视觉对象识别软件研究的大型可视化数据库。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象；在至少一百万个图像中，还提供了边界框。ImageNet包含2万多个类别；一个典型的类别，如“气球”或“草莓”，包含数百个图像。第三方图像URL的注释数据库可以直接从ImageNet免费获得；但是，实际的图像不属于ImageNet。自2010年以来，ImageNet项目每年举办一次软件比赛，即ImageNet大规模视觉识别挑战赛（ILSVRC），软件程序竞相正确分类检测物体和场景。

<!-- more -->

## 1. VGG16 & VGG19

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

{% asset_img vgg.png %}

VGG模型结构：

1. 输入：$224 \times 224$ RGB图片
2. 预处理：每个像素减去RGB均值
3. Filter：一般是$3 \times 3$，也有$1 \times 1$的变体；卷积层stride是1，pad方式为`same`
4. MaxPooling：窗口大小$2\times2$，stride为$2\times2$
5. 三个全连接层FC：前两层units个数为4096，最后一层为分类数（一般是1000）
6. softmax层：最后一层输出概率
7. 激活函数：除最后一层外全是ReLU

VGG模型Tricks：

1. 使用多个$3\times3$卷积层替代一个大size的卷积层（比如$7\times7$），3个$3\times3$卷积层与一个$7\times7$的卷积层的感受野是相同的，都是处理大小为$7\times7$的空间，但是对于相同数量C的输出channel来说，参数数量$3\times(3^2C^2)$是小于$7^2C^2$，同时参数维度更高；
2. 可能使用$1\times1$卷积层，主要是对所有输入进行相同的线性变换；
3. 训练过程使用mini-batch梯度下降，batch_size为256，weight_decay使用$L_2$约束，权重大小为$5\times 10^{-4}$，前两层FC的dropout设为0.5，学习率初始值为$10^{-2}$，当验证集accuracy不再减少时，学习率除以10；
4. 首先训练A网络，A网络训练过程学习率不减少，使用A的前4层卷积层和最后3层FC初始化新的模型；使用均值为0标准差为0.005的高斯分布初始化参数；
5. 图像裁剪以及rescale，为了得到224大小的输入，首先对原始图片进行rescale，rescale的size为$S \geqslant 224$，然后对rescale的图像进行随机裁剪；
6. rescale的大小$S$一般为256或384，预训练时为256；或者动态限制在$[256, 512]$之间，预训练时为384；
7. 测试时rescale大小为$Q$，$Q$可以取$S$中的一个值；测试阶段把网络中原本的三个全连接层依次变为1个conv7x7，2个conv1x1，也就是三个卷积层，测试重用训练时的参数，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入。

VGG模型总结：

* VGG本质上还是在CNN的基础上进行改动，没有脱离卷积网络的基础；
* 由于采取了state-of-the-art，根据实验采取了很多小细节的操作。

## 2. ResNet50 & ResNet101 & ResNet152

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

{% asset_img rest.png %}

$$
\boldsymbol{y} = F(\boldsymbol{x}, \boldsymbol{W}_i) + \boldsymbol{x}
$$

对于输入$\boldsymbol{x}$来说，映射$H(\boldsymbol{x})$是预测输出的最优解，但是实际上我们通过深度学习得到的映射并不一定是完美解，利用残差$F(\boldsymbol{x}) = H(\boldsymbol{x}) - \boldsymbol{x}$这种思想，使用$F(\boldsymbol{x}) + \boldsymbol{x}$作为期望的输出。我觉得采用残差类似于决策树中的残差概念，在ResNet网络中，对于输入到输出中不变的部分或维度，利用残差可以很好的保留这部分稀疏性，训练过程中变化的部分会产生梯度，这种前提是输入输出维度相同；对于输入与输出维度不同的时候，ResNet采取两种策略：补0或使用线性映射$\boldsymbol{W}_s$匹配维度（由1 $\times$ 1卷积完成）。

{% asset_img resnet.png %}

ResNet模型结构：

{% asset_img resnet0.png %}

ResNet模型Tricks：

1. 当输入和输出具有相同的维度时，可以直接使用恒等快捷连接；
2. 当维度增加，考虑两个选项：快捷连接仍然执行恒等映射，额外填充零输入以增加维度；投影快捷连接用于匹配维度（由1 $\times$ 1卷积完成）。对于这两个选项，当快捷连接跨越两种尺寸的特征图时，它们执行时步长为2；
3. 调整图像大小，其较短的边在$[256,480]$之间进行随机采样；
4. 224 $\times$ 224裁剪是从图像或其水平翻转中随机采样，并逐像素减去均值；
5. 在每个卷积之后和激活之前，我们采用批量归一化；
6. 标准颜色增强；
7. 批大小为256；
8. 学习速度从0.1开始，当误差稳定时学习率除以10，并且模型训练高达$60 \times 10^4$次迭代。我们使用的权重衰减为0.0001，动量为0.9；
9. 不使用丢弃；
10. 在测试阶段，为了比较学习我们采用标准的10-crop测试；
11. 采用如$[40, 12]$中的全卷积形式，并在多尺度上对分数进行平均（图像归一化，短边位于$\{224, 256, 384, 480, 640\}$中）；
12. 更深的瓶颈结构，三层是1 $\times$ 1，3 $\times$ 3和1 $\times$ 1卷积，其中1 $\times$ 1层负责减小然后增加（恢复）维度，使3 $\times$ 3层成为具有较小输入/输出维度的瓶颈，两个设计具有相似的时间复杂度。

ResNet模型总结：

* 提出了利用残差优化网络的概念；
* 根据实验说明了，利用残差的神经网络的深度可以很大；
* 最后仅使用了一个全连接层。


## 3. ResNet50V2 & ResNet101V2 & ResNet152V2

[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

ResNetV2在ResNet初代的基础上进行了更深一步的思考

$$
\boldsymbol{y}_l = h(\boldsymbol{x}_l) + F(\boldsymbol{x}_l, \boldsymbol{W}_l)
\\
\boldsymbol{x}_{l+1} = f(\boldsymbol{y}_l)
$$

$h$表示恒等映射，$F$是残差函数，$f$是ReLU

与初代不同的是，V2考虑如果$f$也是恒等映射的情况下，网络的性能

{% asset_img resnetv2-0.png %}

对比了以上几种的$f$，实验结果表明恒等映射的训练误差最小

{% asset_img resnetv2-1.png %}

在此基础上，分析了BN after addition，ReLU before addition，pre-activation方法作为$f$，研究$f$对ResNet效果的影响，效果自然是最后一个full pre-activation最好。

ResNetV2模型总结：

* 在初代ResNet的基础上，通过实验进一步确定了恒等映射对残差网络性能有较好的提升；
* 对于不同的$f$作为恒等映射对性能的影响做出了实验性的结论，并且制定了一套处理流程。

## 4. ResNeXt50 & ResNeXt101

[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

 ResNet的一种变体ResNeXt，它具备以下构建块：

{% asset_img resnetxt.png %}

作者在论文中引入了一个叫作「基数」（cardinality）的超参数，指独立路径的数量，这提供了一种调整模型容量的新思路。实验表明，通过扩大基数值（而不是深度或宽度），准确率得到了高效提升。作者表示，与 Inception 相比，这个全新的架构更容易适应新的数据集或任务，因为它只有一个简单的范式和一个需要调整的超参数，而 Inception 需要调整很多超参数（比如每个路径的卷积层内核大小）。

ResNetXt模型总结：

* ResNetXt与ResNetV2考虑的方向不同，主要考虑卷积层的变换，采取“分裂 - 变换 - 合并”的策略，增加了一个维度Group。

## 5. InceptionV3

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)



## 6. InceptionResNetV2

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)



## 7. Xception

[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)



## 8. MobileNet(alpha=0.25/0.50/0.75/1.0)

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)



## 9. MobileNetV2(alpha=0.35/0.50/0.75/1.0/1.3/1.4)

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)



## 10. DenseNet121 & DenseNet169 & DenseNet201

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)



## 11. NASNetLarge & NASNetMobile

[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)


