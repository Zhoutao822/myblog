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
> [深度学习---GoogLeNet](https://blog.csdn.net/qq_38906523/article/details/80061075)
> [Inception-V4和Inception-Resnet论文阅读和代码解析](https://blog.csdn.net/stesha_chen/article/details/82115429)
> [MobileNet 翻译及总结：用于移动视觉应用的高效卷积神经网络](https://blog.csdn.net/just_sort/article/details/79901885)
> [轻量级网络--MobileNetV2论文解读](https://blog.csdn.net/u011974639/article/details/79199588)
> [NASNet](https://blog.csdn.net/qq_14845119/article/details/83050862)

{% asset_img cnn.png %}

<!-- more -->

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

作者在论文中引入了一个叫作**基数**（cardinality）的超参数，指独立路径的数量，这提供了一种调整模型容量的新思路。实验表明，通过扩大基数值（而不是深度或宽度），准确率得到了高效提升。作者表示，与 Inception 相比，这个全新的架构更容易适应新的数据集或任务，因为它只有一个简单的范式和一个需要调整的超参数，而 Inception 需要调整很多超参数（比如每个路径的卷积层内核大小）。

ResNetXt模型总结：

* ResNetXt与ResNetV2考虑的方向不同，主要考虑卷积层的变换，采取“分裂 - 变换 - 合并”的策略，增加了一个维度cardinality。

## 5. InceptionV3

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

首先了解一下Inception结构，其目的是为了既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能

{% asset_img inception0.png %}

1. 采用不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合； 
2. 之所以卷积核大小采用1、3和5，主要是为了方便对齐。设定卷积步长stride=1之后，只要分别设定pad=0、1、2，那么卷积之后便可以得到相同维度的特征，然后这些特征就可以直接拼接在一起了； 
3. 文章说很多地方都表明pooling挺有效，所以Inception里面也嵌入了。 
4. 网络越到后面，特征越抽象，而且每个特征所涉及的感受野也更大了，因此随着层数的增加，3x3和5x5卷积的比例也要增加。

但是，使用5x5的卷积核仍然会带来巨大的计算量。 为此，文章借鉴NIN2，采用1x1卷积核来进行降维。 
例如：上一层的输出为100x100x128，经过具有256个输出的5x5卷积层之后(stride=1，pad=2)，输出数据为100x100x256。其中，卷积层的参数为128x5x5x256。假如上一层输出先经过具有32个输出的1x1卷积层，再经过具有256个输出的5x5卷积层，那么最终的输出数据仍为为100x100x256，但卷积参数量已经减少为128x1x1x32 + 32x5x5x256，大约减少了4倍。

{% asset_img inception1.png %}

下面的准则来源于大量的实验，因此包含一定的推测，但实际证明基本都是有效的

1. 避免表达瓶颈，特别是在网络靠前的地方。 信息流前向传播过程中显然不能经过高度压缩的层，即表达瓶颈。从input到output，feature map的宽和高基本都会逐渐变小，但是不能一下子就变得很小。比如你上来就来个kernel = 7, stride = 5 ,这样显然不合适。 另外输出的维度channel，一般来说会逐渐增多(每层的num_output)，否则网络会很难训练。（特征维度并不代表信息的多少，只是作为一种估计的手段）
2. 高维特征更易处理。 高维特征更易区分，会加快训练。
3. 可以在低维嵌入上进行空间汇聚而无需担心丢失很多信息。 比如在进行3x3卷积之前，可以对输入先进行降维而不会产生严重的后果。假设信息可以被简单压缩，那么训练就会加快。
4. 平衡网络的宽度与深度。

再到InceptionV2中，采用不对称卷积，n×1卷积核替代n×n卷积核，这种结构在前几层效果不太好，但对特征图大小为12~20的中间层效果明显。 

并且引入了Batch normal层，使用3×3替换5×5卷积核

{% asset_img inception2.png %}

InceptionV3，一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块。

{% asset_img inception4.png %}


Inception模型总结：

* 使用了集成方法的思想，将卷积行为变成并行的以增加不同维度特征间的联系；
* 提出了不对称卷积的思路；
* 引入了辅助分类器的概念，以改善非常深的网络的收敛，辅助分类器起着正则化项的作用；
* 使用平行的步长为2的块来缩减特征图的网格大小，缩减网格尺寸的同时扩展滤波器组的Inception模块；
* 通过标签平滑进行模型正则化。

{% asset_img inception3.png %}

## 6. InceptionResNetV2

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

提到InceptionResNetV2就必须提到Inception-v4，作者首先使用纯Inception Block构建了Inception-v4模型，其结构为

{% asset_img inceptionresnet6.png %}

每个block的具体结构如下：（每个block中没有标记v的都表示same padding）

**Stem**，InceptionResNetV2和Inception-v4共用

{% asset_img inceptionresnet0.png %}

**Inception-A**

{% asset_img inceptionresnet1.png %}

**Inception-B**

{% asset_img inceptionresnet2.png %}

**Inception-C**

{% asset_img inceptionresnet3.png %}

**35 to 17 ReductionA**，共用，但是具体filters个数根据不同模型而不同，参考表格

{% asset_img inceptionresnet4.png %}

{% asset_img inceptionresnet17.png %}

**17 to 8 ReductionB**

{% asset_img inceptionresnet5.png %}

---

接下来是InceptionResNetV1和InceptionResNetV2，V1和V2整体结构相同，细节的block有差异，InceptionResNetV2的Stem通用，InceptionResNetV1的Stem与其他不同

{% asset_img inceptionresnet12.png %}

**Stem**，仅限InceptionResNetV1

{% asset_img inceptionresnet11.png %}

**Inception-ResNet-v1 Inception-ResNet-A**

{% asset_img inceptionresnet7.png %}

**Inception-ResNet-v1 Inception-ResNet-B**

{% asset_img inceptionresnet8.png %}

**Inception-ResNet-v1 Inception-ResNet-C**

{% asset_img inceptionresnet10.png %}

**35 to 17 ReductionA**，共用，参考上面

**17 to 8 ReductionB**，注意这里与V2的小区别，filters数量不同

{% asset_img inceptionresnet9.png %}

---

**Inception-ResNet-v2 Inception-ResNet-A**

{% asset_img inceptionresnet13.png %}

**Inception-ResNet-v2 Inception-ResNet-B**

{% asset_img inceptionresnet14.png %}

**Inception-ResNet-v2 Inception-ResNet-C**

{% asset_img inceptionresnet16.png %}

**35 to 17 ReductionA**，共用，参考上面

**17 to 8 ReductionB**，注意这里与V1的小区别，filters数量不同

{% asset_img inceptionresnet15.png %}

---

最后，作者发现如果filter的个数超过1000个，残差网络会变得不稳定，网络会在训练的早期就“死掉”，也就意味着在几万次迭代之后，avg_pool之前的最后几层网络参数全是0。解决方案是要么减小learning rate，要么对这些层增加额外的batch normalization。

作者又发现如果将残差部分缩放后再跟需要相加的层相加，会使网络在训练过程中更稳定。因此作者选择了一些缩放因子在0.1到0.3之间，用这个缩放因子去缩放残差网络，然后再做加法，如下图

{% asset_img inceptionresnet18.png %}

InceptionResNetV2模型总结：

* 结合ResNet的思想，构建了Inception-ResNet模块，既优化了训练过程，又可以扩大特征间的联系；
* 提出了缩小残差的思想。

## 7. Xception

[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

Xception从Inception进化而来，同时Xception是ResNeXt的一个变种

串行式group的module，被起名 separable convolution 

{% asset_img xception.png %}

Xception模型结构：

{% asset_img xception1.png %}

实验结果，Xception在ImageNet上稍优于Inceptionv3，参数数量和Inceptionv3基本一致，速度也差不多。

Xception模型总结：

* 使用串行group替代Inception的并行group。

## 8. MobileNet(alpha=0.25/0.50/0.75/1.0)

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

MobileNet的提出很明显是为了部署在移动设备上，与ImageNet比赛相比，移动设备对内存、运算速度等方面要求很高，精度要求可以适当降低，而传统的ImageNet比赛中取胜的模型的参数基本已经达到一个很高的程度，所以使用MobileNet在保持类似精度的条件下显著的减少模型参数和计算量成为另一个目标。

MobileNet引入了几个重要的技巧以降低运算量：

### 8.1 Deep-wise Separabe 深度可分离卷积

{% asset_img mobilenetv1-1.png %}

传统卷积考虑到通道数，对于输入通道数为$N$，输出通道数为$M$，输入长宽$D_K$，输出长宽$D_F$，则卷积层的计算量为

$$
D_K \times D_K \times M \times N \times D_F \times D_F
$$

若采用深度可分离卷积，首先使用2D卷积核对所有通道进行处理，再使用3D1×1卷积核处理之前输出的特征图，最终得到的输出是与传统卷积是相同的

计算量分为两部分

$$
D_K \times D_K \times M \times D_F \times D_F
\\
M \times N \times D_F \times D_F
$$

显然总计算量为上述两部分之和，那么与传统卷积计算量的比值为

$$
\frac{1}{N} + \frac{1}{D_K^2}
$$

MobileNet对3×3卷积进行这种改变使得计算量减少为原始的$\frac{1}{9}$，准确率仅下降一点。

### 8.2 Network Structure and Training 网络结构和训练

所有层之后都是BatchNormalization和ReLU非线性激活函数，但是最后的全连接层例外，它没有非线性激活函数，直接馈送到softmax层进行分类。

{% asset_img mobilenetv1-2.png %}

{% asset_img mobilenetv1.png %}

与训练大模型相反，我们较少地使用正则化和数据增加技术，因为小模型不容易过拟合。当训练MobileNets时，我们不使用sideheads或者labelsmoothing，通过限制croping的尺寸来减少图片扭曲。另外，我们发现重要的是在depthwise滤波器上放置很少或没有重量衰减（L2正则化），因为它们参数很少

### 8.3 Width Multiplier: Thinner Models(alpha参数：更小的模型)

引入了一个非常简单的参数$\alpha$，称为width multiplier。这个参数widthmultiplier的作用是在每层均匀地减负网络。对于一个给定的层和widthmultiplierα，输入通道的数量从$M$变成$\alpha M$，输出通道的数量从$N$变成$\alpha N$。深度可分离卷积（以widthmultiplier参数$\alpha$为计）的计算复杂度： 
$\alpha \in (0,1]$，通常设为1，0.75，0.5和0.25。$\alpha = 1$表示基准MobileNet，而$\alpha < 1$则表示瘦身的MobileNets。Width multiplier有减少计算复杂度和参数数量（大概$\alpha$二次方）的作用。Width multiplier可以应用于任何模型结构，以定义一个具有合理准确性，延迟和尺寸的新的较小的模型。它用于定义新的简化结构，但需要重新进行训练。

### 8.4 Resolution Multiplier: Reduced Representation

降低神经网络的第二个超参数是resolution multiplier ρ，简而言之就是作用于输出特征图大小（输出大小的系数），$\rho \in (0,1]$，通常设为224,192,160或者128。$\rho=1$是基本MobileNets而$\rho<1$示瘦身的MobileNets。计算量

$$
D_K \times D_K \times \alpha M \times \rho D_F \times \rho D_F + \alpha M \times \alpha N \times \rho D_F \times \rho D_F
$$

MobileNet模型总结：

* 提供了Width Multiplier和Resolution Multiplier两个参数控制模型大小；
* 提出了深度可分离卷积的思想。

## 9. MobileNetV2(alpha=0.35/0.50/0.75/1.0/1.3/1.4)

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

MobileNetv2架构是基于倒置残差结构(inverted residual structure)，原本的残差结构的主分支是有三个卷积，两个逐点卷积通道数较多，而倒置的残差结构刚好相反，中间的卷积通道数(依旧使用深度分离卷积结构)较多，旁边的较小。此外，我们发现去除主分支中的非线性变换是有效的，这可以保持模型表现力。

MobileNetV2提出了manifold of interest（兴趣流形）概念，表示感兴趣的数据内容，但是目前无法定量描述，仅凭经验研究。

我们的目的是在低维依然保持manifold of interest，但是实际上是将矩阵映射到低维后再进行ReLU，最后在求逆投影回原来的维度时，这种manifold of interest会丢失很多的信息。论文针对这个问题使用linear bottleneck(即不使用ReLU激活，做了线性变换)的来代替原本的非线性激活变换。所以通过在卷积模块中后插入linear bottleneck来捕获兴趣流形。 实验证明，使用linear bottleneck可以防止非线性破坏太多信息。

{% asset_img mobilenetv2-1.png %}

MobileNetv2的结构同样是将标准卷积拆分为深度卷积和逐点卷积，在逐点卷积后使用了接1×1卷积，该卷积使用线性变换，总称为一层低维linear bottleneck，其作用是将输入映射回低维空间。

考虑到倒残差结构Inverted residuals，对于Expansion layer(即linear到深度卷积部分)仅是伴随张量非线性变换的部分实现细节，我们可将shortcuts放在linear bottleneck之间连接

{% asset_img mobilenetv2-2.png %}

下表是bottleneck convolution的基本实现：

{% asset_img mobilenetv2-3.png %}

* 首先是1×1 conv2d变换通道，后接ReLU6激活(ReLU6即最高输出为6，超过了会clip下来)
* 中间是深度卷积,后接ReLU
* 最后的1×1 conv2d后面不接ReLU了，而是论文提出的linear bottleneck

{% asset_img mobilenetv2-4.png %}

可以看到相比与之前的残差模块，中间的深度卷积较宽，除了开始的升维的1×1 1×11×1卷积，做shortcut的1×1 1×11×1卷积通道数较少，呈现的是倒立状态，故称为Inverted residuals。

{% asset_img mobilenetv2-5.png %}

训练细节：

1. 训练器：RMSPropOptimizer, decay and momentum都设置0.9
2. 标准的权重衰减：4e-5
3. 学习率：初始学习率为0.045,每个epoch后衰减0.98
4. batch_size：16GPU内设置96
5. 其他细节：每层后使用BN层

## 10. DenseNet121 & DenseNet169 & DenseNet201

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

DenseNet的提出基于快速连接的思想，与ResNet异曲同工，它进一步利用了快捷连接的效果 - 它将所有层直接相互连接。在这种新颖的架构中，每层的输入由所有早期层的特征图组成，其输出传递给每个后续层。特征映射与深度级联聚合在一起。

{% asset_img densenet1.png %}

除了解决消失的渐变问题之外，作者认为这种架构还鼓励特征重用，使网络具有高参数效率。对此的一个简单解释是，身份映射的输出被添加到下一个块，如果两个层的特征映射具有非常不同的分布，则可能阻碍信息流。因此，连接特征映射可以保留所有特征映射并增加输出的方差，从而鼓励特征重用。

{% asset_img densenet2.png %}

遵循这个范例，我们知道第l层将具有$k \times（l-1）+ k_0$个输入要素图，其中$k_0$是输入图像中的通道数。作者使用了一个称为增长率（k）的超参数来防止网络过长，他们还使用1x1卷积瓶颈层来减少昂贵的3x3卷积之前的特征映射数量。整体结构如下表所示：

{% asset_img densenet0.png %}

DenseNet模型总结：

* 更远距离的快速连接。

## 11. NASNetLarge & NASNetMobile

[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

NasNet，是当前图像识别领域的最佳模型，这个模型并非是人为设计出来的，而是通过谷歌很早之前推出的AutoML自动训练出来的。

NasNet的组成由两种网络单元组合而成

{% asset_img nasnet1.png %}

这两种单元的堆叠方案如下：

{% asset_img nasnet0.png %}

搜索过程：

{% asset_img nasnet2.png %}

如上图所示，控制器RNN从搜索空间中以概率p预测网络结构A。worker单元，学习该网络直到收敛，并得到准确性R。最终将梯度p*R传递给RNN控制器进行梯度更新。

{% asset_img nasnet3.png %}

控制器依次搜索隐藏状态，隐藏状态，何种操作，何种操作，何种组合方法，这5个方法和操作的组合。其中，每种方法，每种操作都对应于一个softmax损失。这样重复B次，得到一个最终block模块。最终的损失函数就有5B个。实验中最优的B=5。

{% asset_img nasnet4.png %}

其中，黄色的可选的操作包括上图所示的13种操作。

最终论文得到了3个网络结构，分别为NASNet-A，NASNet-B， NASNet-C。

NASNet-A：

{% asset_img nasnet-a.png %}

NASNet-B：

{% asset_img nasnet-b.png %}

NASNet-C：

{% asset_img nasnet-c.png %}

NasNet模型总结：

* 设计了新的搜索空间，即NASNet search space，并在实验中搜索得到最优的网络结构NASNet

* 不管是乘-加计算量，还是参数量，NASNet都优于目前人工设计的网络结构

* 提出新的正则化技术，ScheduledDropPath，是DropPath方法的改进版，可以大大提高了模型的泛化能力。

DropPath方法在训练过程中以随机概率p进行drop，该概率训练中保持不变。而ScheduledDropPath方法在训练过程线性的提高随机概率p。
