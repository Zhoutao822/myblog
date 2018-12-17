---
title: TensorFlow-CIFAR10
date: 2018-12-13 17:54:58
categories:
- Coding
tags:
- Code
- CNN
- Estimator
- CIFAR-10
mathjax: true
---

参考：

> [CIFAR-10 ResNet](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)

CIFAR10项目下有6个py文件：`cifar10.py, cifar10_main.py, cifar10_model.py, cifar10_utils.py, generate_cifar10_tfrecords.py, model_base.py`

<!-- more -->

先从`generate_cifar10_tfrecords.py`开始

```python
"""这部分代码的功能是生成TFRecords，这是专门提供给TensorFlow的一种数据格式。

代码功能包括下载图片数据并解压，生成train，validation，eval三个.tfrecords文件作为训练集、验证集和测试集
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse # 控制运行时参数，在__main__中使用
import os # os.path.join 连接路径
import sys # 获取系统相关信息

import tarfile # 压缩/解压文件
from six.moves import cPickle as pickle # 序列化数据
from six.moves import xrange  # python3中可以直接使用range，性能比python2中的range强
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME # 从这个url下载原始数据
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py' # 解压到本地的路径


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  # maybe_download已经被1.12版本废弃，替代方法的是直接在keras中load_data，参考 卷积神经网络-coding
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)

# 数据写入到TFRecords需要用到Feature，这是一种key-value的形式，同时我们可以定义value的类型，
# 一般有三种Int64List，BytesList，FloatList，顾名思义，value也必须是List形式
# 这里Int类型保存的是标签，Bytes类型保存图片数据，理论上来说，也可以用其他类型保存图片，
# 但是二进制字符串需要的空间比int或float小很多，而一张图片包含的数据量大，为了减小存储压力，
# 通常做法是将图片写为bytes类型，而label本身只是单值数字，所以可以用int。
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 解压后的文件包括data_batch_[1...5]和test_batch，这里将1到4作为训练集，5为验证集，
# 返回三种集合的文件名，这里的文件名没有后缀.XX
def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names

# 根据文件路径将序列化的原始数据读取出来，使用tf.gfile.Open，rb代表二进制读，
# sys.version_info判断是python2还是python3，调用pickle.load转成python数据结构
# 根据convert_to_tfrecord里的代码可知，是一个字典类型的数据被序列化了
def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict

# 写入TFRecords的具体函数，两个参数都是文件的绝对路径
def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  
  # tf.python_io.TFRecordWriter在1.12中是tf.io.TFRecordWriter，对文件处理时采用的方式是
  # with XX as xx，这很常见，因为可以避免忘记关闭文件
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      # 这里用b'xxxx'是因为原始数据使用了bytes字符串作为key而不是str，也是为了减少存储空间
      # len(labels)可以知道有多少条数据或图片
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      # 注意了，record_writer.write是一条数据一条数据地往tfrecords中写入
      for i in range(num_entries_in_batch):
        # 写入地内容是tf.train.Example类型，对应上面地key-value形式
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()), # tobytes与tostring最终结果相同
                'label': _int64_feature(labels[i])
            }))
        # SerializeToString序列化，必须步骤
        record_writer.write(example.SerializeToString())


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  # 首先下载文件
  download_and_extract(data_dir)
  # 获取文件名
  file_names = _get_file_names()
  # 连接文件路径
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  # 分别对三种集合处理
  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    # 保存的文件名为xxx.tfrecords
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    # 这里先删除已经存在的输出文件，不错
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  # 运行时参数控制，似乎可以使用tf.app.flags替代，只是指定了下载文件的路径
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')

  args = parser.parse_args()
  main(args.data_dir)
```

通过运行上面的py文件，我们得到了三个TFRecords文件，使用TFRecords文件的好处有，与直接使用原始数据相比，TensorFlow模型读取TFReocrds文件更快，内存压力更小，要知道，模型训练速度的瓶颈可能不是运算能力而是IO，配合tf.data.Dataset更快。还有就是生成的TFRecords文件可能比原始文件大。

---

然后是`cifar10.py`

```python
"""这个代码的功能就很简单了，读取TFReocrds，生成Dataset"""
import os

import tensorflow as tf

# 图片数据的原始形状，长宽为32，RGB图所以深度为3
HEIGHT = 32
WIDTH = 32
DEPTH = 3

# 这里定义一个类，超级方便后面的调用，封装了读取、处理数据一系列的方法
class Cifar10DataSet(object):
  """Cifar10 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """
  # __init__是python类的初始化方法，参数包括：data_dir即三个TFRecords文件的路径
  # subset指定是train，validation还是eval，从而生成指定的Dataset，
  # 训练时train和validation，测试时eval，use_distortion指定是否需要扰乱数据集，
  # 图片扰乱一般包括裁剪、旋转、平移、翻转、亮度等等方式调整数据，从而增加模型的鲁棒性
  def __init__(self, data_dir, subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion
  
  # 获取文件路径名，返回了一个List类型
  def get_filenames(self):
    if self.subset in ['train', 'validation', 'eval']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  # 将单个tf.Example还原为float32的图片数据和int32的label数据
  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    # 在1.12版本使用tf.io.parse_single_example，调用方式很简单，
    # FixedLenFeature代表固定长度的数据，图片对应是字节数组，所以按tf.string格式转换
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    # 得到的数据需要解码，按照无符号8bit格式
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    # [depth, height, width]到[height, width, depth]，同时从uint8到float32
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # 调用preprocess方法处理图片数据
    image = self.preprocess(image)

    return image, label

  # 生成batch_size大小的Dataset
  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    # 注意这里使用了tf.data.TFRecordDataset读取TFRecords文件，repeat进行复制
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # 使用map对每一个TFRecordDataset中读取的Example进行parser，
    # num_parallel_calls指定并行处理的数量，这里等于batch_size
    dataset = dataset.map(
        self.parser, num_parallel_calls=batch_size)

    # 训练集需要shuffle，buffer_size等于数据集40%的总数量加上3个batch_size
    if self.subset == 'train':
      min_queue_examples = int(
          Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  # 如果是train数据集，且扰动为真，resize_image_with_crop_or_pad填充图像到40长宽
  # 1.12版本tf.image.random_crop随机裁剪到32长宽
  # random_flip_left_right随机翻转图片
  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

  # 静态方法，保存的是数据集大小，不需要实例
  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 45000 # 这里可能写错了？40000
    elif subset == 'validation':
      return 5000 # 这里可能写错了？10000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
```

上面的代码主要是为了训练模型提供Dataset，生成Dataset的过程中已经进行了数据扰动，而且这个类适用于三种不同的数据集。

---

接着是`cifar10_utils.py`

```python
"""
提供了运行时对模型的run_config，有很多部分是需要替换和丢弃的
"""
import collections
import six

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging        # 1.12版本 tf.logging.info
from tensorflow.core.framework import node_def_pb2                  # 1.12版本 tf.NodeDef
from tensorflow.python.framework import device as pydev             # 1.12版本 tf.DeviceSpec
from tensorflow.python.training import basic_session_run_hooks      # 1.12版本 tf.train.SecondOrStepTimer/tf.train.SessionRunArgs
from tensorflow.python.training import session_run_hook             # 1.12版本 tf.train.SessionRunHook
from tensorflow.python.training import training_util                # 1.12版本 tf.train.get_global_step
from tensorflow.python.training import device_setter                # 1.12版本 tf.train.replica_device_setter
from tensorflow.contrib.learn.python.learn import run_config        # 1.12版本 tf.estimator.RunConfig


# 1.12版本使用tf.estimator.RunConfig，废弃tf.contrib.learn.RunConfig，
# 而且在tf.estimator.RunConfig可以直接调用，不需要重写，此部分略过
class RunConfig(tf.contrib.learn.RunConfig): 
  def uid(self, whitelist=None):
    """Generates a 'Unique Identifier' based on all internal fields.
    Caller should use the uid string to check `RunConfig` instance integrity
    in one session use, but should not rely on the implementation details, which
    is subject to change.
    Args:
      whitelist: A list of the string names of the properties uid should not
        include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
        includes most properties user allowes to change.
    Returns:
      A uid string.
    """
    if whitelist is None:
      whitelist = run_config._DEFAULT_UID_WHITE_LIST

    state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    # Pop out the keys in whitelist.
    for k in whitelist:
      state.pop('_' + k, None)

    ordered_state = collections.OrderedDict(
        sorted(state.items(), key=lambda t: t[0]))
    # For class instance without __repr__, some special cares are required.
    # Otherwise, the object address will be used.
    if '_cluster_spec' in ordered_state:
      ordered_state['_cluster_spec'] = collections.OrderedDict(
         sorted(ordered_state['_cluster_spec'].as_dict().items(),
                key=lambda t: t[0])
      )
    return ', '.join(
        '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state)) 

# ExamplesPerSecondHook提供了在训练时输出xx examples/sec的功能，
# 本质上是tf.train.StepCounterHook乘以batch_size，所以可以使用StepCounterHook替换，
# 或者使用tf.train.LoggingTensorHook记录
# 使用tf.train.SessionRunHook替换继承的父类
class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """
  # Hook的继承都差不多，主要包括几个方法，而且Hook一般用在Estimator中，与keras的callback有点不同
  # __init__，begin，before_run，after_run
  # 从__init__参数可以看出，estimator的log形式是按照steps或secs来输出的，默认为100steps打印一次
  def __init__(
      self,
      batch_size,
      every_n_steps=100,
      every_n_secs=None,):
    """Initializer for ExamplesPerSecondHook.

      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    # 这个判断形式确保二者至少有一个
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    # 1.12版本使用tf.train.SecondOrStepTimer作为触发器，每n步或者n秒触发一次
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)
    # 初始化time和steps
    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  # tf.train.get_global_step替换，用一个_global_step_tensor记录训练steps，训练开始时执行begin方法
  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')
  
  # tf.train.SessionRunArgs替换，相当于把_global_step_tensor添加到Session.run()中
  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  # 在每一个step运行完成后通过触发器判断是否需要打印信息
  def after_run(self, run_context, run_values):
    _ = run_context
    # 触发器与global_step对比
    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      # 获取两次触发器之间间隔的时间和steps
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps
        # 用batch_size乘以steps/secs得到examples/secs得到平均值
        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        # 1.12版本 tf.logging替换
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)

# 1.12版本使用tf.train.replica_device_setter替换，这个方法是为了配置在不同的device上运行
# 比如CPU和GPU或者多GPU，默认只用一个CPU device
def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
  if ps_ops == None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    # 必须替换的方法，返回下一个任务的索引？
    # Returns the next ps task index for placement in round-robin order
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")
  # 获得device的规范名称，在with tf.device(DeviceSpec(job="train", ))中使用
  def _local_device_chooser(op):
    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser
```

这一部分的代码可以直接使用TensorFlow的函数替换，虽然重写也是可以的，了解了部分源码。

---

`model_base.py`里面是ResNet模型，这里不表，准备与其他模型例如VGG16，InceptionV3等等一起写一下。
这里看一下CIFAR10的模型`cifar10_model.py`

```python
"""Model class for Cifar10 Dataset."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_base

# 基于ResNet的CIFAR10模型
class ResNetCifar10(model_base.ResNet):
  """Cifar10 model with ResNetV1 and basic residual block."""
  # num_layers模型层数，is_training模型处于train状态还是eval状态，
  # data_format表示图片数据中Depth处于第几个维度
  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_first'):
    super(ResNetCifar10, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 10 + 1
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
  # 前向传播，这里具体细节和ResNet相关，之后再分析，输出的x的维度是num_classes
  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Image standardization.
    x = x / 128 - 1

    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    res_func = self._residual_v1

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, 3, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)

    x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes)

    return x
```

---

最后是`cifar10_main.py`，包括train和eval部分功能，这里可能大部分代码需要重新写以适配新的版本

```python
"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html
"""
from __future__ import division
from __future__ import print_function

import argparse
import functools # 作用于或返回其他函数的函数
import itertools # chain() 可以把一组迭代对象串联起来，形成一个更大的迭代器
import os

import cifar10
import cifar10_model
import cifar10_utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 设置运行过程中所有函数能打印INFO级别的信息
tf.logging.set_verbosity(tf.logging.INFO)

# 自定义Estimator需要一个model_fn，返回值为一个tf.estimator.EstimatorSpec，
# 这里使用了私有方法_resnet_model_fn，是一种很常见的方式，因为TensorFlow中很多地方的参数是方法名，
# 通过私有方法可以简单的实现，或者用lambda形式也可以
# 三个参数num_gpus使用的GPU数量，variable_strategy使用CPU还是GPU，num_workers多进程处理数据
def get_model_fn(num_gpus, variable_strategy, num_workers):
  """Returns a function that will build the resnet model."""

  def _resnet_model_fn(features, labels, mode, params):
    """Resnet model body.

    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay # 权重衰减
    momentum = params.momentum # 动量影响梯度下降速度，参考 深度学习-优化器

    # tower表示处于不同device的数据，比如使用tower_losses记录分别在不同的device上产生的损失
    # 包括CPU和GPUs
    tower_features = features
    tower_labels = labels
    tower_losses = [] # 损失
    tower_gradvars = [] # 梯度
    tower_preds = [] # 预测值

    # NHWC是TensorFlow的默认设置，并且NCHW是使用cuDNN在NVIDIA GPU上训练时使用的最佳格式。
    # 最佳实践是构建可同时处理两种数据格式的模型。这简化了对GPU的训练，然后在CPU上运行推理。
    # 如果使用英特尔MKL优化编译TensorFlow，则会优化和支持许多操作，尤其是与基于CNN的模型相关的操作NCHW。
    # 如果不使用MKL，则在使用时某些操作在CPU上不受支持NCHW。
    # 这两种格式的简要历史是TensorFlow开始使用，NHWC因为它在CPU上速度稍快。
    data_format = params.data_format
    if not data_format:
      if num_gpus == 0:
        data_format = 'channels_last'
      else:
        data_format = 'channels_first'

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'
    
    # device的名称一般为'/cpu:0'或者'/gpu:1'
    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        # 注意这里需要使用tf.train.replica_device_setter替换
        device_setter = cifar10_utils.local_device_setter(
            worker_device=worker_device)
      # GreedyLoadBalancingStrategy懒加载策略，tf.contrib.training下只有两个策略
      # 另一个是RandomStrategy
      elif variable_strategy == 'GPU':
        device_setter = cifar10_utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            # 这里数据被预先分组了，比如均分给所有GPU，每个device只处理tower_features[i]和tower_labels[i]
            # 调用_tower_fn进行计算
            loss, gradvars, preds = _tower_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                data_format, params.num_layers, params.batch_norm_decay,
                params.batch_norm_epsilon)
            # 使用append把所有device的结果存入List中
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
              # batch_norm更新需要执行UPDATE_OPS操作，不会自动进行
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
      all_grads = {}
      for grad, var in itertools.chain(*tower_gradvars):
        if grad is not None:
          all_grads.setdefault(var, []).append(grad)
      for var, grads in six.iteritems(all_grads):
        # Average gradients on the same device as the variables
        # to which they apply.
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            # 计算平均梯度，根据设备数来计算
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
      # 计算梯度完成后还需要执行梯度下降的运算，默认是cpu或者gpu1号
      # Suggested learning rate scheduling from
      # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
      # 使用多进程读取数据
      num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
          'train') // (params.train_batch_size * num_workers)
      boundaries = [
          num_batches_per_epoch * x
          for x in np.array([82, 123, 300], dtype=np.int64)
      ]
      # 学习率阶段性下降
      staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]
      # 在82steps后学习率减少到0.1倍，123后0.01倍，300后0.002倍
      learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                  boundaries, staged_lr)
      # 均值损失
      loss = tf.reduce_mean(tower_losses, name='loss')
      # ExamplesPerSecondHook打印训练速度
      examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
          params.train_batch_size, every_n_steps=10)
      # 使用LoggingTensorHook打印learning_rate和loss
      tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      train_hooks = [logging_hook, examples_sec_hook]
      # 优化器使用MomentumOptimizer
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=momentum)
      # 分布式优化器，暂时不了解
      if params.sync:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
        train_hooks.append(sync_replicas_hook)

      # Create single grouped train op
      train_op = [
          optimizer.apply_gradients(
              gradvars, global_step=tf.train.get_global_step())
      ]
      train_op.extend(update_ops)
      train_op = tf.group(*train_op)
      # concat横向连接
      predictions = {
          'classes':
              tf.concat([p['classes'] for p in tower_preds], axis=0),
          'probabilities':
              tf.concat([p['probabilities'] for p in tower_preds], axis=0)
      }
      stacked_labels = tf.concat(labels, axis=0)
      metrics = {
          'accuracy':
              tf.metrics.accuracy(stacked_labels, predictions['classes'])
      }
    # 返回EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)

  return _resnet_model_fn

# 计算loss, gradvars, preds的函数
def _tower_fn(is_training, weight_decay, feature, label, data_format,
              num_layers, batch_norm_decay, batch_norm_epsilon):
  """Build computation tower (Resnet).

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.

  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

  """
  # 构建模型
  model = cifar10_model.ResNetCifar10(
      num_layers,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon,
      is_training=is_training,
      data_format=data_format)
  # 前向传播计算结果logits
  logits = model.forward_pass(feature, input_data_format='channels_last')
  tower_pred = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }

  tower_loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=label)
  tower_loss = tf.reduce_mean(tower_loss)

  model_params = tf.trainable_variables()
  # 对loss增加l2范数约束，衰减权重为weight_decay
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])

  tower_grad = tf.gradients(tower_loss, model_params)
  # 返回值包括model_params模型参数
  return tower_loss, zip(tower_grad, model_params), tower_pred

# 定义输入函数
def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=True):
  """Create input graph for model.

  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
    image_batch, label_batch = dataset.make_batch(batch_size)
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return [image_batch], [label_batch]

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    # 把image_batch分解为batch_size个张量，axis为0，刚好是batch_size的维度[i, x, x, x]，
    # 然后将这batch_size个张量根据循环报数规则分配给num_shards个进程
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    return feature_shards, label_shards

# tf.contrib.learn.Experiment全部丢弃，需要用根据tf.estimator重新构建Estimator，
# 使用train_spec和eval_spec，调用tf.estimator.train_and_evaluate
# 同样这里使用了私有方法
# 重写
def get_experiment_fn(data_dir,
                      num_gpus,
                      variable_strategy,
                      use_distortion_for_training=True):
  """Returns an Experiment function.

  Experiments perform training on several workers in parallel,
  in other words experiments know how to invoke train and eval in a sensible
  fashion for distributed training. Arguments passed directly to this
  function are not tunable, all other arguments should be passed within
  tf.HParams, passed to the enclosed function.

  Args:
      data_dir: str. Location of the data for input_fns.
      num_gpus: int. Number of GPUs on each worker.
      variable_strategy: String. CPU to use CPU as the parameter server
      and GPU to use the GPUs as the parameter server.
      use_distortion_for_training: bool. See cifar10.Cifar10DataSet.
  Returns:
      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
      tf.contrib.learn.Experiment.

      Suitable for use by tf.contrib.learn.learn_runner, which will run various
      methods on Experiment (train, evaluate) based on information
      about the current runner in `run_config`.
  """

  def _experiment_fn(run_config, hparams):
    """Returns an Experiment."""
    # Create estimator.
    train_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset='train',
        num_shards=num_gpus,
        batch_size=hparams.train_batch_size,
        use_distortion_for_training=use_distortion_for_training)

    eval_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset='eval',
        batch_size=hparams.eval_batch_size,
        num_shards=num_gpus)

    num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
    if num_eval_examples % hparams.eval_batch_size != 0:
      raise ValueError(
          'validation set size must be multiple of eval_batch_size')

    train_steps = hparams.train_steps
    eval_steps = num_eval_examples // hparams.eval_batch_size
 
    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(num_gpus, variable_strategy,
                              run_config.num_worker_replicas or 1),
        config=run_config,
        params=hparams)

    # Create experiment.
    return tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        eval_steps=eval_steps)

  return _experiment_fn

# tf.contrib.learn.learn_runner.run要丢弃了，这里重写
def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = cifar10_utils.RunConfig(
      session_config=sess_config, model_dir=job_dir)
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(data_dir, num_gpus, variable_strategy,
                        use_distortion_for_training),
      run_config=config,
      hparams=tf.contrib.training.HParams(
          is_chief=config.is_chief,
          **hparams))

# 下面全是配置运行时参数
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='CPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=44,
      help='The number of layers of the model.')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.')
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.')
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """)
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--data-format',
      type=str,
      default=None,
      help="""\
      If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
  parser.add_argument(
      '--batch-norm-decay',
      type=float,
      default=0.997,
      help='Decay for batch norm.')
  parser.add_argument(
      '--batch-norm-epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.')
  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

  main(**vars(args))
```
