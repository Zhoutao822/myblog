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


ImageNet项目是一个用于视觉对象识别软件研究的大型可视化数据库。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象；在至少一百万个图像中，还提供了边界框。ImageNet包含2万多个类别；一个典型的类别，如“气球”或“草莓”，包含数百个图像。第三方图像URL的注释数据库可以直接从ImageNet免费获得；但是，实际的图像不属于ImageNet。自2010年以来，ImageNet项目每年举办一次软件比赛，即ImageNet大规模视觉识别挑战赛（ILSVRC），软件程序竞相正确分类检测物体和场景。

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

## 2. ResNet50 & ResNet101 & ResNet152

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)



## 3. ResNet50V2 & ResNet101V2 & ResNet152V2

[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)



## 4. ResNeXt50 & ResNeXt101

[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)



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


