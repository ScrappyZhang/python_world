# 常用层Core

常用层对应于core模块，core内部定义了一系列常用的网络层，包括全连接、激活层等

## Dense层

```
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

Dense就是常用的全连接层，所实现的运算是`output = activation(dot(input, kernel)+bias)`。其中`activation`是逐元素计算的激活函数，`kernel`是本层的权值矩阵，`bias`为偏置向量，只有当`use_bias=True`才会添加。

如果本层的输入数据的维度大于2，则会先被压为与`kernel`相匹配的大小。

这里是一个使用示例：

```
# as first layer in a sequential model:
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

### 参数：

- units：大于0的整数，代表该层的输出维度。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：偏置向量初始化方法，为预定义初始化方法名的字符串，或用于初始化偏置向量的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入

形如(batch_size, ..., input_dim)的nD张量，最常见的情况为(batch_size, input_dim)的2D张量

### 输出

形如(batch_size, ..., units)的nD张量，最常见的情况为(batch_size, units)的2D张量

------

Activation层

```
keras.layers.core.Activation(activation)
```

激活层对一个层的输出施加激活函数

### 参数

- activation：将要使用的激活函数，为预定义激活函数名或一个Tensorflow/Theano的函数。参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)

### 输入shape

任意，当使用激活层作为第一层时，要指定`input_shape`

### 输出shape

与输入shape相同

------

Dropout层

```
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
```

为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。

### 参数

- rate：0~1的浮点数，控制需要断开的神经元的比例
- noise_shape：整数张量，为将要应用在输入上的二值Dropout mask的shape，例如你的输入为(batch_size, timesteps, features)，并且你希望在各个时间步上的Dropout mask都相同，则可传入noise_shape=(batch_size, 1, features)。
- seed：整数，使用的随机数种子

### 参考文献

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

------

## Flatten层

```
keras.layers.core.Flatten()
```

Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

### 例子

```
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

------

## Reshape层

```
keras.layers.core.Reshape(target_shape)
```

Reshape层用来将输入shape转换为特定的shape

### 参数

- target_shape：目标shape，为整数的tuple，不包含样本数目的维度（batch大小）

### 输入shape

任意，但输入的shape必须固定。当使用该层为模型首层时，需要指定`input_shape`参数

### 输出shape

`(batch_size,)+target_shape`

### 例子

```
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
```

------

## Permute层

```
keras.layers.core.Permute(dims)
```

Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。

### 参数

- dims：整数tuple，指定重排的模式，不包含样本数的维度。重拍模式的下标从1开始。例如（2，1）代表将输入的第二个维度重拍到输出的第一个维度，而将输入的第一个维度重排到第二个维度

### 例子

```
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

### 输入shape

任意，当使用激活层作为第一层时，要指定`input_shape`

### 输出shape

与输入相同，但是其维度按照指定的模式重新排列

------

## RepeatVector层

```
keras.layers.core.RepeatVector(n)
```

RepeatVector层将输入重复n次

### 参数

- n：整数，重复的次数

### 输入shape

形如（nb_samples, features）的2D张量

### 输出shape

形如（nb_samples, n, features）的3D张量

### 例子

```
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

------

## Lambda层

```
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
```

本函数用以对上一层的输出施以任何Theano/TensorFlow表达式

### 参数

- function：要实现的函数，该函数仅接受一个变量，即上一层的输出
- output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
- mask: 掩膜
- arguments：可选，字典，用来记录向函数中传递的其他关键字参数

### 例子

```
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```

```
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
         output_shape=antirectifier_output_shape))
```

### 输入shape

任意，当使用该层作为第一层时，要指定`input_shape`

### 输出shape

由`output_shape`参数指定的输出shape，当使用tensorflow时可自动推断

------

## ActivityRegularizer层

```
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```

经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值

### 参数

- l1：1范数正则因子（正浮点数）
- l2：2范数正则因子（正浮点数）

### 输入shape

任意，当使用该层作为第一层时，要指定`input_shape`

### 输出shape

与输入shape相同

------

## Masking层

```
keras.layers.core.Masking(mask_value=0.0)
```

使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步

对于输入张量的时间步，即输入张量的第1维度（维度从0开始算，见例子），如果输入张量在该时间步上都等于`mask_value`，则该时间步将在模型接下来的所有层（只要支持masking）被跳过（屏蔽）。

如果模型接下来的一些层不支持masking，却接受到masking过的数据，则抛出异常。

### 例子

考虑输入数据`x`是一个形如(samples,timesteps,features)的张量，现将其送入LSTM层。因为你缺少时间步为3和5的信号，所以你希望将其掩盖。这时候应该：

- 赋值`x[:,3,:] = 0.`，`x[:,5,:] = 0.`
- 在LSTM层之前插入`mask_value=0.`的`Masking`层

```
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

# 卷积层Convolutional

## Conv1D层

```
keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数`input_shape`。例如`(10,128)`代表一个长为10的序列，序列中每个信号为128向量。而`(None, 128)`代表变长的128维向量序列。

该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果`use_bias=True`，则还会加上一个偏置项，若`activation`不为None，则输出为经过激活函数的输出。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
- strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。参考[WaveNet: A Generative Model for Raw Audio, section 2.1.](https://arxiv.org/abs/1609.03499)。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

形如（samples，steps，input_dim）的3D张量

### 输出shape

形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，`steps`的值会改变

【Tips】可以将Convolution1D看作Convolution2D的快捷版，对例子中（10，32）的信号进行1D卷积相当于对其进行卷积核为（filter_length, 32）的2D卷积。【@3rduncle】

------

## Conv2D层

```
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (128,128,3)`代表128*128的彩色RGB图像（`data_format='channels_last'`）

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量

‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的`input_shape`，请参考下面提供的例子。

### 输出shape

‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量

‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量

输出的行列数可能会因为填充方法而改变

------

## SeparableConv2D层

```
keras.layers.convolutional.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

该层是在深度方向上的可分离卷积。

可分离卷积首先按深度方向进行卷积（对每个输入通道分别卷积），然后逐点进行卷积，将上一步的卷积结果混合到输出通道中。参数`depth_multiplier`控制了在depthwise卷积（第一步）的过程中，每个输入通道信号产生多少个输出通道。

直观来说，可分离卷积可以看做讲一个卷积核分解为两个小的卷积核，或看作Inception模块的一种极端情况。

当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (3,128,128)`代表128*128的彩色RGB图像

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- depth_multiplier：在按深度卷积的步骤中，每个输入通道使用多少个输出通道
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- depthwise_regularizer：施加在按深度卷积的权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- pointwise_regularizer：施加在按点卷积的权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- depthwise_constraint：施加在按深度卷积权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- pointwise_constraint施加在按点卷积权重的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量

‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的`input_shape`，请参考下面提供的例子。

### 输出shape

‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量

‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量

输出的行列数可能会因为填充方法而改变

------

## Conv2DTranspose层

```
keras.layers.convolutional.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

该层是转置的卷积操作（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。同时保留与卷积层兼容的连接模式。

当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (3,128,128)`代表128*128的彩色RGB图像

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量

‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的`input_shape`，请参考下面提供的例子。

### 输出shape

‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量

‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量

输出的行列数可能会因为填充方法而改变

### 参考文献

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
- [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
- [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

------

## Conv3D层

```
keras.layers.convolutional.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (3,10,128,128)`代表对10帧128*128的彩色RGB图像进行卷积。数据的通道位置仍然有`data_format`参数指定。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由3个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由3个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
- padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：单个整数或由3个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

‘channels_first’模式下，输入应为形如（samples，channels，input_dim1，input_dim2, input_dim3）的5D张量

‘channels_last’模式下，输入应为形如（samples，input_dim1，input_dim2, input_dim3，channels）的5D张量

这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的`input_shape`。

------

## Cropping1D层

```
keras.layers.convolutional.Cropping1D(cropping=(1, 1))
```

在时间轴（axis1）上对1D输入（即时间序列）进行裁剪

### 参数

- cropping：长为2的tuple，指定在序列的首尾要裁剪掉多少个元素

### 输入shape

- 形如（samples，axis_to_crop，features）的3D张量

### 输出shape

- 形如（samples，cropped_axis，features）的3D张量

------

## Cropping2D层

```
keras.layers.convolutional.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

对2D输入（图像）进行裁剪，将在空域维度，即宽和高的方向上裁剪

### 参数

- cropping：长为2的整数tuple，分别为宽和高方向上头部与尾部需要裁剪掉的元素数
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

形如（samples，depth, first_axis_to_crop, second_axis_to_crop）

### 输出shape

形如(samples, depth, first_cropped_axis, second_cropped_axis)的4D张量

------

## Cropping3D层

```
keras.layers.convolutional.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

对2D输入（图像）进行裁剪

### 参数

- cropping：长为3的整数tuple，分别为三个方向上头部与尾部需要裁剪掉的元素数
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

形如 (samples, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)的5D张量

### 输出shape

形如(samples, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)的5D张量

------

## UpSampling1D层

```
keras.layers.convolutional.UpSampling1D(size=2)
```

在时间轴上，将每个时间步重复`length`次

### 参数

- size：上采样因子

### 输入shape

- 形如（samples，steps，features）的3D张量

### 输出shape

- 形如（samples，upsampled_steps，features）的3D张量

------

## UpSampling2D层

```
keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
```

将数据的行和列分别重复size[0]和size[1]次

### 参数

- size：整数tuple，分别为行和列上采样因子
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

‘channels_first’模式下，为形如（samples，channels, upsampled_rows, upsampled_cols）的4D张量

‘channels_last’模式下，为形如（samples，upsampled_rows, upsampled_cols，channels）的4D张量

------

## UpSampling3D层

```
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), data_format=None)
```

将数据的三个维度上分别重复size[0]、size[1]和ize[2]次

本层目前只能在使用Theano为后端时可用

### 参数

- size：长为3的整数tuple，代表在三个维度上的上采样因子
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘channels_first’模式下，为形如（samples, channels, dim1, dim2, dim3）的5D张量

‘channels_last’模式下，为形如（samples, upsampled_dim1, upsampled_dim2, upsampled_dim3,channels,）的5D张量

------

## ZeroPadding1D层

```
keras.layers.convolutional.ZeroPadding1D(padding=1)
```

对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度

### 参数

- padding：整数，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴1（第1维，第0维是样本数）

### 输入shape

形如（samples，axis_to_pad，features）的3D张量

### 输出shape

形如（samples，paded_axis，features）的3D张量

------

## ZeroPadding2D层

```
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format=None)
```

对2D输入（如图片）的边界填充0，以控制卷积以后特征图的大小

### 参数

- padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴3和轴4（即在'th'模式下图像的行和列，在‘channels_last’模式下要填充的则是轴2，3）
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，形如（samples，channels，first_axis_to_pad，second_axis_to_pad）的4D张量

‘channels_last’模式下，形如（samples，first_axis_to_pad，second_axis_to_pad, channels）的4D张量

### 输出shape

‘channels_first’模式下，形如（samples，channels，first_paded_axis，second_paded_axis）的4D张量

‘channels_last’模式下，形如（samples，first_paded_axis，second_paded_axis, channels）的4D张量

------

## ZeroPadding3D层

```
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

将数据的三个维度上填充0

本层目前只能在使用Theano为后端时可用

### 参数

padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴3，轴4和轴5，‘channels_last’模式下则是轴2，3和4

- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples, channels, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad,）的5D张量

‘channels_last’模式下，为形如（samples, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad, channels）的5D张量

### 输出shape

‘channels_first’模式下，为形如（samples, channels, first_paded_axis，second_paded_axis, third_paded_axis,）的5D张量

‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量



# 池化层Pooling

## MaxPooling1D层

```
keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

对时域1D信号进行最大值池化

### 参数

- pool_size：整数，池化窗口大小
- strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
- padding：‘valid’或者‘same’

### 输入shape

- 形如（samples，steps，features）的3D张量

### 输出shape

- 形如（samples，downsampled_steps，features）的3D张量

------

## MaxPooling2D层

```
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

为空域信号施加最大值池化

### 参数

- pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
- strides：整数或长为2的整数tuple，或者None，步长值。
- border_mode：‘valid’或者‘same’
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

------

## MaxPooling3D层

```
keras.layers.pooling.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

为3D信号（空域或时空域）施加最大值池化

本层目前只能在使用Theano为后端时可用

### 参数

- pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
- strides：整数或长为3的整数tuple，或者None，步长值。
- padding：‘valid’或者‘same’
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘channels_first’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘channels_last’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

------

## AveragePooling1D层

```
keras.layers.pooling.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

对时域1D信号进行平均值池化

### 参数

- pool_size：整数，池化窗口大小
- strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
- padding：‘valid’或者‘same’

### 输入shape

- 形如（samples，steps，features）的3D张量

### 输出shape

- 形如（samples，downsampled_steps，features）的3D张量

------

## AveragePooling2D层

```
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

为空域信号施加平均值池化

### 参数

- pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
- strides：整数或长为2的整数tuple，或者None，步长值。
- border_mode：‘valid’或者‘same’
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

------

## AveragePooling3D层

```
keras.layers.pooling.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

为3D信号（空域或时空域）施加平均值池化

本层目前只能在使用Theano为后端时可用

### 参数

- pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
- strides：整数或长为3的整数tuple，或者None，步长值。
- padding：‘valid’或者‘same’
- data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘channels_first’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘channels_last’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

------

## GlobalMaxPooling1D层

```
keras.layers.pooling.GlobalMaxPooling1D()
```

对于时间信号的全局最大池化

### 输入shape

- 形如（samples，steps，features）的3D张量

### 输出shape

- 形如(samples, features)的2D张量

------

## GlobalAveragePooling1D层

```
keras.layers.pooling.GlobalAveragePooling1D()
```

为时域信号施加全局平均值池化

### 输入shape

- 形如（samples，steps，features）的3D张量

### 输出shape

- 形如(samples, features)的2D张量

------

## GlobalMaxPooling2D层

```
keras.layers.pooling.GlobalMaxPooling2D(dim_ordering='default')
```

为空域信号施加全局最大值池化

### 参数

- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

形如(nb_samples, channels)的2D张量

------

## GlobalAveragePooling2D层

```
keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')
```

为空域信号施加全局平均值池化

### 参数

- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。

### 输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

形如(nb_samples, channels)的2D张量



# 局部连接层LocallyConnceted

## LocallyConnected1D层

```
keras.layers.local.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

`LocallyConnected1D`层与`Conv1D`工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入位置的滤波器是不一样的。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
- strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rata均不兼容
- padding：补0策略，目前仅支持`valid`（大小写敏感），`same`可能会在将来支持。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rata均与任何不为1的strides均不兼容。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

形如（samples，steps，input_dim）的3D张量

### 输出shape

形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，`steps`的值会改变

------

## LocallyConnected2D层

```
keras.layers.local.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

`LocallyConnected2D`层与`Convolution2D`工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入patch的滤波器是不一样的，当使用该层作为模型首层时，需要提供参数`input_dim`或`input_shape`参数。参数含义参考`Convolution2D`。

### 参数

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。
- padding：补0策略，目前仅支持`valid`（大小写敏感），`same`可能会在将来支持。
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- use_bias:布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象

### 输入shape

‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量

‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的`input_shape`，请参考下面提供的例子。

### 输出shape

‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量

‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量

输出的行列数可能会因为填充方法而改变

### 例子

```
# apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image
# with `data_format="channels_last"`:
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# now model.output_shape == (None, 30, 30, 64)
# notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

# add a 3x3 unshared weights convolution on top, with 32 output filters:
model.add(LocallyConnected2D(32, (3, 3)))
# now model.output_shape == (None, 28, 28, 32)
```

# 循环层Recurrent

## Recurrent层

```
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```

这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类`LSTM`，`GRU`或`SimpleRNN`。

所有的循环层（`LSTM`,`GRU`,`SimpleRNN`）都继承本层，因此下面的参数可以在任何循环层中使用。

### 参数

- weights：numpy array的list，用以初始化权重。该list形如`[(input_dim, output_dim),(output_dim, output_dim),(output_dim,)]`
- return_sequences：布尔值，默认`False`，控制返回类型。若为`True`则返回整个序列，否则仅返回输出序列的最后一个输出
- go_backwards：布尔值，默认为`False`，若为`True`，则逆向处理输入序列并返回逆序后的序列
- stateful：布尔值，默认为`False`，若为`True`，则一个batch中下标为i的样本的最终状态将会用作下一个batch同样下标的样本的初始状态。
- unroll：布尔值，默认为`False`，若为`True`，则循环层将被展开，否则就使用符号化的循环。当使用TensorFlow为后端时，循环网络本来就是展开的，因此该层不做任何事情。层展开会占用更多的内存，但会加速RNN的运算。层展开只适用于短序列。
- implementation：0，1或2， 若为0，则RNN将以更少但是更大的矩阵乘法实现，因此在CPU上运行更快，但消耗更多的内存。如果设为1，则RNN将以更多但更小的矩阵乘法实现，因此在CPU上运行更慢，在GPU上运行更快，并且消耗更少的内存。如果设为2（仅LSTM和GRU可以设为2），则RNN将把输入门、遗忘门和输出门合并为单个矩阵，以获得更加在GPU上更加高效的实现。注意，RNN dropout必须在所有门上共享，并导致正则效果性能微弱降低。
- input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)
- input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接`Flatten`层，然后又要连接`Dense`层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果循环层不是网络的第一层，你需要在网络的第一层中指定序列的长度（通过`input_shape`指定）。

### 输入shape

形如（samples，timesteps，input_dim）的3D张量

### 输出shape

如果`return_sequences=True`：返回形如（samples，timesteps，output_dim）的3D张量

否则，返回形如（samples，output_dim）的2D张量

### 例子

```
# as the first layer in a Sequential model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# now model.output_shape == (None, 32)
# note: `None` is the batch dimension.

# the following is identical:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# for subsequent layers, no need to specify the input size:
         model.add(LSTM(16))

# to stack recurrent layers, you must use return_sequences=True
# on any recurrent layer that feeds into another recurrent layer.
# note that you only need to specify the input size on the first layer.
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

### 指定RNN初始状态的注意事项

可以通过设置`initial_state`用符号式的方式指定RNN层的初始状态。即，`initial_stat`的值应该为一个tensor或一个tensor列表，代表RNN层的初始状态。

也可以通过设置`reset_states`参数用数值的方法设置RNN的初始状态，状态的值应该为numpy数组或numpy数组的列表，代表RNN层的初始状态。

### 屏蔽输入数据（Masking）

循环层支持通过时间步变量对输入数据进行Masking，如果想将输入数据的一部分屏蔽掉，请使用[Embedding](http://keras-cn.readthedocs.io/en/latest/layers/embedding_layer)层并将参数`mask_zero`设为`True`。

### 使用状态RNN的注意事项

可以将RNN设置为‘stateful’，意味着由每个batch计算出的状态都会被重用于初始化下一个batch的初始状态。状态RNN假设连续的两个batch之中，相同下标的元素有一一映射关系。

要启用状态RNN，请在实例化层对象时指定参数`stateful=True`，并在Sequential模型使用固定大小的batch：通过在模型的第一层传入`batch_size=(...)`和`input_shape`来实现。在函数式模型中，对所有的输入都要指定相同的`batch_size`。

如果要将循环层的状态重置，请调用`.reset_states()`，对模型调用将重置模型中所有状态RNN的状态。对单个层调用则只重置该层的状态。

------

## SimpleRNN层

```
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

全连接RNN网络，RNN的输出会被回馈到输入

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

### 参考文献

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

------

## GRU层

```
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

门限循环单元（详见参考文献）

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

### 参考文献

- [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

------

## LSTM层

```
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

Keras长短期记忆模型，关于此算法的详情，请参考[本教程](http://deeplearning.net/tutorial/lstm.html)

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- recurrent_activation: 为循环步施加的激活函数（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

### 参考文献

- [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)（original 1997 paper）
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

## ConvLSTM2D层

```
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

ConvLSTM2D是一个LSTM网络，但它的输入变换和循环变换是通过卷积实现的

### 参数

- filters: 整数，输出的维度，该参数含义同普通卷积层的filters
- kernel_size: 整数或含有n个整数的tuple/list，指定卷积窗口的大小
- strides: 整数或含有n个整数的tuple/list，指定卷积步长，当不等于1时，无法使用dilation功能，即dialation_rate必须为1.
- padding: "valid" 或 "same" 之一
- data_format: * data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channels_last”。
- dilation_rate: 单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
- activation: activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- recurrent_activation: 用在recurrent部分的激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
- use_bias: Boolean, whether the layer uses a bias vector.
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

### 输入shape

若data_format='channels_first'， 为形如(samples,time, channels, rows, cols)的5D tensor 若data_format='channels_last' 为形如(samples,time, rows, cols, channels)的5D tensor

### 输出shape

if return_sequences： if data_format='channels_first' ：5D tensor (samples, time, filters, output_row, output_col) if data_format='channels_last' ：5D tensor (samples, time, output_row, output_col, filters) else if data_format ='channels_first' :4D tensor (samples, filters, output_row, output_col) if data_format='channels_last' :4D tensor (samples, output_row, output_col, filters) (o_row和o_col由filter和padding决定)

### 参考文献

[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1) * 当前的实现不包含cell输出上的反馈循环（feedback loop）

## SimpleRNNCell层

```
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

SinpleRNN的Cell类

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例

## GRUCell层

```
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

GRU的Cell类

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

## LSTMCell层

```
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

LSTM的Cell类

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

## StackedRNNCells层

```
keras.layers.StackedRNNCells(cells)
```

这是一个wrapper，用于将多个recurrent cell包装起来，使其行为类型单个cell。该层用于实现搞笑的stacked RNN

### 参数

- cells：list，其中每个元素都是一个cell对象

### 例子

```
cells = [
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
]

inputs = keras.Input((timesteps, input_dim))
x = keras.layers.StackedRNNCells(cells)(inputs)
```

## CuDNNGRU层

```
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

基于CuDNN的快速GRU实现，只能在GPU上运行，只能使用tensoflow为后端

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明

## CuDNNLSTM层

```
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

基于CuDNN的快速LSTM实现，只能在GPU上运行，只能使用tensoflow为后端

### 参数

- units：输出维度
- activation：激活函数，为预定义的激活函数名（参考[激活函数](http://keras-cn.readthedocs.io/en/latest/other/activations)）
- use_bias: 布尔值，是否使用偏置项
- kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- kernel_regularizer：施加在权重上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- activity_regularizer：施加在输出上的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- kernel_constraints：施加在权重上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- recurrent_constraints：施加在循环核上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- bias_constraints：施加在偏置上的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
- 其他参数参考Recurrent的说明



# 嵌入层 Embedding

## Embedding层

```
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]

Embedding层只能作为模型的第一层

### 参数

- input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
- output_dim：大于0的整数，代表全连接嵌入的维度
- embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations)
- embeddings_regularizer: 嵌入矩阵的正则项，为[Regularizer](http://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
- embeddings_constraint: 嵌入矩阵的约束项，为[Constraints](http://keras-cn.readthedocs.io/en/latest/other/constraints)对象
- mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用[递归层](http://keras-cn.readthedocs.io/en/latest/layers/recurrent_layer)处理变长输入时有用。设置为`True`的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 1。
- input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接`Flatten`层，然后接`Dense`层，则必须指定该参数，否则`Dense`层的输出维度无法自动推断。

### 输入shape

形如（samples，sequence_length）的2D张量

### 输出shape

形如(samples, sequence_length, output_dim)的3D张量

### 例子

```
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

### 参考文献

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

# Merge层

Merge层提供了一系列用于融合两个层或两个张量的层对象和方法。以大写首字母开头的是Layer类，以小写字母开头的是张量的函数。小写字母开头的张量函数在内部实际上是调用了大写字母开头的层。

## Add

```
keras.layers.Add()
```

添加输入列表的图层。

该层接收一个相同shape列表张量，并返回它们的和，shape不变。

### Example

```
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## SubStract

```
keras.layers.Subtract()
```

两个输入的层相减。

它将大小至少为2，相同Shape的列表张量作为输入，并返回一个张量（输入[0] - 输入[1]），也是相同的Shape。

### Example

```
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## Multiply

```
keras.layers.Multiply()
```

该层接收一个列表的同shape张量，并返回它们的逐元素积的张量，shape不变。

## Average

```
keras.layers.Average()
```

该层接收一个列表的同shape张量，并返回它们的逐元素均值，shape不变。

## Maximum

```
keras.layers.Maximum()
```

该层接收一个列表的同shape张量，并返回它们的逐元素最大值，shape不变。

## Concatenate

```
keras.layers.Concatenate(axis=-1)
```

该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。

### 参数

- axis: 想接的轴
- **kwargs: 普通的Layer关键字参数

## Dot

```
keras.layers.Dot(axes, normalize=False)
```

计算两个tensor中样本的张量乘积。例如，如果两个张量`a`和`b`的shape都为（batch_size, n），则输出为形如（batch_size,1）的张量，结果张量每个batch的数据都是a[i,:]和b[i,:]的矩阵（向量）点积。

### 参数

- axes: 整数或整数的tuple，执行乘法的轴。
- normalize: 布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。
- **kwargs: 普通的Layer关键字参数

## add

```
keras.layers.add(inputs)
```

Add层的函数式包装

### 参数：

- inputs: 长度至少为2的张量列表A
- **kwargs: 普通的Layer关键字参数

### 返回值

输入列表张量之和

### Example

```
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## subtract

```
keras.layers.subtract(inputs)
```

Subtract层的函数式包装

### 参数：

- inputs: 长度至少为2的张量列表A
- **kwargs: 普通的Layer关键字参数

### 返回值

输入张量列表的差别

### Example

```
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## multiply

```
keras.layers.multiply(inputs)
```

Multiply的函数式包装

### 参数：

- inputs: 长度至少为2的张量列表
- **kwargs: 普通的Layer关键字参数

### 返回值

输入列表张量之逐元素积

## average

```
keras.layers.average(inputs)
```

Average的函数包装

### 参数：

- inputs: 长度至少为2的张量列表
- **kwargs: 普通的Layer关键字参数

### 返回值

输入列表张量之逐元素均值

## maximum

```
keras.layers.maximum(inputs)
```

Maximum的函数包装

### 参数：

- inputs: 长度至少为2的张量列表
- **kwargs: 普通的Layer关键字参数

### 返回值

输入列表张量之逐元素均值

## concatenate

```
keras.layers.concatenate(inputs, axis=-1)
```

Concatenate的函数包装

### 参数

- inputs: 长度至少为2的张量列
- axis: 相接的轴
- **kwargs: 普通的Layer关键字参数

## dot

```
keras.layers.dot(inputs, axes, normalize=False)
```

Dot的函数包装

### 参数

- inputs: 长度至少为2的张量列
- axes: 整数或整数的tuple，执行乘法的轴。
- normalize: 布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。
- **kwargs: 普通的Layer关键字参数



# 高级激活层Advanced Activation

## LeakyReLU层

```
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```

LeakyRelU是修正线性单元（Rectified Linear Unit，ReLU）的特殊版本，当不激活时，LeakyReLU仍然会有非零输出值，从而获得一个小梯度，避免ReLU可能出现的神经元“死亡”现象。即，`f(x)=alpha * x for x < 0`, `f(x) = x for x>=0`

### 参数

- alpha：大于0的浮点数，代表激活函数图像中第三象限线段的斜率

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

### 参考文献

[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

------

## PReLU层

```
keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

该层为参数化的ReLU（Parametric ReLU），表达式是：`f(x) = alpha * x for x < 0`, `f(x) = x for x>=0`，此处的`alpha`为一个与xshape相同的可学习的参数向量。

### 参数

- alpha_initializer：alpha的初始化函数
- alpha_regularizer：alpha的正则项
- alpha_constraint：alpha的约束项
- shared_axes：该参数指定的轴将共享同一组科学系参数，例如假如输入特征图是从2D卷积过来的，具有形如`(batch, height, width, channels)`这样的shape，则或许你会希望在空域共享参数，这样每个filter就只有一组参数，设定`shared_axes=[1,2]`可完成该目标

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

### 参考文献

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)

------

## ELU层

```
keras.layers.advanced_activations.ELU(alpha=1.0)
```

ELU层是指数线性单元（Exponential Linera Unit），表达式为： 该层为参数化的ReLU（Parametric ReLU），表达式是：`f(x) = alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x>=0`

### 参数

- alpha：控制负因子的参数

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

### 参考文献

- [>Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)

------

## ThresholdedReLU层

```
keras.layers.advanced_activations.ThresholdedReLU(theta=1.0)
```

该层是带有门限的ReLU，表达式是：`f(x) = x for x > theta`,`f(x) = 0 otherwise`

### 参数

- theata：大或等于0的浮点数，激活门限位置

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

### 参考文献

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)



# （批）规范化BatchNormalization

## BatchNormalization层

```
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1

### 参数

- axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行`data_format="channels_first`的2D卷积后，一般会设axis=1。
- momentum: 动态均值的动量
- epsilon：大于0的小浮点数，用于防止除0错误
- center: 若设为True，将会将beta作为偏置加上去，否则忽略参数beta
- scale: 若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
- beta_initializer：beta权重的初始方法
- gamma_initializer: gamma的初始化方法
- moving_mean_initializer: 动态均值的初始化方法
- moving_variance_initializer: 动态方差的初始化方法
- beta_regularizer: 可选的beta正则
- gamma_regularizer: 可选的gamma正则
- beta_constraint: 可选的beta约束
- gamma_constraint: 可选的gamma约束

### 输入shape

任意，当使用本层为模型首层时，指定`input_shape`参数时有意义。

### 输出shape

与输入shape相同

### 参考文献

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)

【Tips】BN层的作用

（1）加速收敛 （2）控制过拟合，可以少用或不用Dropout和正则 （3）降低网络对初始化权重不敏感 （4）允许使用较大的学习率



# 噪声层Noise

## GaussianNoise层

```
keras.layers.noise.GaussianNoise(stddev)
```

为数据施加0均值，标准差为`stddev`的加性高斯噪声。该层在克服过拟合时比较有用，你可以将它看作是随机的数据提升。高斯噪声是需要对输入数据进行破坏时的自然选择。

因为这是一个起正则化作用的层，该层只在训练时才有效。

### 参数

- stddev：浮点数，代表要产生的高斯噪声标准差

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

------

## GaussianDropout层

```
keras.layers.noise.GaussianDropout(rate)
```

为层的输入施加以1为均值，标准差为`sqrt(rate/(1-rate)`的乘性高斯噪声

因为这是一个起正则化作用的层，该层只在训练时才有效。

### 参数

- rate：浮点数，断连概率，与[Dropout层](http://keras-cn.readthedocs.io/en/latest/layers/core_layer/#dropout)相同

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

### 参考文献

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

## AlphaDropout

```
keras.layers.noise.AlphaDropout(rate, noise_shape=None, seed=None)
```

对输入施加Alpha Dropout

Alpha Dropout是一种保持输入均值和方差不变的Dropout，该层的作用是即使在dropout时也保持数据的自规范性。 通过随机对负的饱和值进行激活，Alphe Drpout与selu激活函数配合较好。

### 参数

- rate: 浮点数，类似Dropout的Drop比例。乘性mask的标准差将保证为`sqrt(rate / (1 - rate))`.
- seed: 随机数种子

### 输入shape

任意，当使用该层为模型首层时需指定`input_shape`参数

### 输出shape

与输入相同

### 参考文献

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)



# 包装器Wrapper

## TimeDistributed包装器

```
keras.layers.wrappers.TimeDistributed(layer)
```

该包装器可以把一个层应用到输入的每一个时间步上

### 参数

- layer：Keras层对象

输入至少为3D张量，下标为1的维度将被认为是时间维

例如，考虑一个含有32个样本的batch，每个样本都是10个向量组成的序列，每个向量长为16，则其输入维度为`(32,10,16)`，其不包含batch大小的`input_shape`为`(10,16)`

我们可以使用包装器`TimeDistributed`包装`Dense`，以产生针对各个时间步信号的独立全连接：

```
# as the first layer in a model
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

# subsequent layers: no need for input_shape
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

程序的输出数据shape为`(32,10,8)`

使用`TimeDistributed`包装`Dense`严格等价于`layers.TimeDistribuedDense`。不同的是包装器`TimeDistribued`还可以对别的层进行包装，如这里对`Convolution2D`包装：

```
model = Sequential()
model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
```

## Bidirectional包装器

```
keras.layers.wrappers.Bidirectional(layer, merge_mode='concat', weights=None)
```

双向RNN包装器

### 参数

- layer：`Recurrent`对象
- merge_mode：前向和后向RNN输出的结合方式，为`sum`,`mul`,`concat`,`ave`和`None`之一，若设为None，则返回值不结合，而是以列表的形式返回

### 例子

```
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```



# 编写自己的层

对于简单的定制操作，我们或许可以通过使用`layers.core.Lambda`层来完成。但对于任何具有可训练权重的定制层，你应该自己来实现。

这里是一个Keras2的层应该具有的框架结构(如果你的版本更旧请升级)，要定制自己的层，你需要实现下面三个方法

- `build(input_shape)`：这是定义权重的方法，可训练的权应该在这里被加入列表``self.trainable_weights`中。其他的属性还包括`self.non_trainabe_weights`（列表）和`self.updates`（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以参考`BatchNormalization`层的实现来学习如何使用上面两个属性。这个方法必须设置`self.built = True`，可通过调用`super([layer],self).build()`实现
- `call(x)`：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心`call`的第一个参数：输入张量
- `compute_output_shape(input_shape)`：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断

```
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

现存的Keras层代码可以为你的实现提供良好参考，阅读源代码吧！