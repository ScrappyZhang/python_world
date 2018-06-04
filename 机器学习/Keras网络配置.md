# 损失函数loss

## 目标函数objectives

目标函数，或称损失函数，是编译一个模型必须的两个参数之一：

```
model.compile(loss='mean_squared_error', optimizer='sgd')
```

可以通过传递预定义目标函数名字指定目标函数，也可以传递一个Theano/TensroFlow的符号函数作为目标函数，该函数对每个数据点应该只返回一个标量值，并以下列两个参数为参数：

- y_true：真实的数据标签，Theano/TensorFlow张量
- y_pred：预测值，与y_true相同shape的Theano/TensorFlow张量

```
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

真实的优化目标函数是在各个数据点得到的损失函数值之和的均值

请参考[目标实现代码](https://github.com/fchollet/keras/blob/master/keras/objectives.py)获取更多信息

## 可用的目标函数

- mean_squared_error或mse
- mean_absolute_error或mae
- mean_absolute_percentage_error或mape
- mean_squared_logarithmic_error或msle
- squared_hinge
- hinge
- categorical_hinge
- binary_crossentropy（亦称作对数损失，logloss）
- logcosh
- categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如`(nb_samples, nb_classes)`的二值序列
- sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：`np.expand_dims(y,-1)`
- kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
- poisson：即`(predictions - targets * log(predictions))`的均值
- cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数

**注意**: 当使用"categorical_crossentropy"作为目标函数时,标签应该为多类模式,即one-hot编码的向量,而不是单个数值. 可以使用工具中的`to_categorical`函数完成该转换.示例如下:

```
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```

# 优化器optimizers

优化器是编译Keras模型必要的两个参数之一

```
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

可以在调用`model.compile()`之前初始化一个优化器对象，然后传入该函数（如上所示），也可以在调用`model.compile()`时传递一个预定义优化器名。在后者情形下，优化器的参数将使用默认值。

```
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```

## 所有优化器都可用的参数

参数`clipnorm`和`clipvalue`是所有优化器都可以使用的参数,用于对梯度进行裁剪.示例如下:

```
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

## SGD

```
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量

### 参数

- lr：大或等于0的浮点数，学习率
- momentum：大或等于0的浮点数，动量参数
- decay：大或等于0的浮点数，每次更新后的学习率衰减值
- nesterov：布尔值，确定是否使用Nesterov动量

------

## RMSprop

```
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
```

除学习率可调整外，建议保持优化器的其他默认参数不变

该优化器通常是面对递归神经网络时的一个良好选择

### 参数

- lr：大或等于0的浮点数，学习率
- rho：大或等于0的浮点数
- epsilon：大或等于0的小浮点数，防止除0错误

------

## Adagrad

```
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
```

建议保持优化器的默认参数不变

### Adagrad

- lr：大或等于0的浮点数，学习率
- epsilon：大或等于0的小浮点数，防止除0错误

------

## Adadelta

```
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
```

建议保持优化器的默认参数不变

### 参数

- lr：大或等于0的浮点数，学习率
- rho：大或等于0的浮点数
- epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

------

- [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

## Adam

```
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

该优化器的默认值来源于参考文献

### 参数

- lr：大或等于0的浮点数，学习率
- beta_1/beta_2：浮点数， 0<beta<1，通常很接近1
- epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

------

## Adamax

```
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

Adamax优化器来自于Adam的论文的Section7，该方法是基于无穷范数的Adam方法的变体。

默认参数由论文提供

### 参数

- lr：大或等于0的浮点数，学习率
- beta_1/beta_2：浮点数， 0<beta<1，通常很接近1
- epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

------

## Nadam

```
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
```

Nesterov Adam optimizer: Adam本质上像是带有动量项的RMSprop，Nadam就是带有Nesterov 动量的Adam RMSprop

默认参数来自于论文，推荐不要对默认参数进行更改。

### 参数

- lr：大或等于0的浮点数，学习率
- beta_1/beta_2：浮点数， 0<beta<1，通常很接近1
- epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

## TFOptimizer

```
keras.optimizers.TFOptimizer(optimizer)
```

TF优化器的包装器

# 激活函数Activations

激活函数可以通过设置单独的[激活层](http://keras-cn.readthedocs.io/en/latest/layers/core_layer/#activation)实现，也可以在构造层对象时通过传递`activation`参数实现。

```
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

等价于

```
model.add(Dense(64, activation='tanh'))
```

也可以通过传递一个逐元素运算的Theano/TensorFlow/CNTK函数来作为激活函数：

```
from keras import backend as K

def tanh(x):
    return K.tanh(x)

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh))
```

------

## 预定义激活函数

- softmax：对输入数据的最后一维进行softmax，输入数据应形如`(nb_samples, nb_timesteps, nb_dims)`或`(nb_samples,nb_dims)`
- elu
- selu: 可伸缩的指数线性单元（Scaled Exponential Linear Unit），参考[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- softplus
- softsign
- relu
- tanh
- sigmoid
- hard_sigmoid
- linear

## 高级激活函数

对于简单的Theano/TensorFlow/CNTK不能表达的复杂激活函数，如含有可学习参数的激活函数，可通过[高级激活函数](http://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer)实现，如PReLU，LeakyReLU等

# 性能评估

## 使用方法

性能评估模块提供了一系列用于模型性能评估的函数,这些函数在模型编译时由`metrics`关键字设置

性能评估函数类似与[目标函数](http://keras-cn.readthedocs.io/en/latest/other/objectives/), 只不过该性能的评估结果讲不会用于训练.

可以通过字符串来使用域定义的性能评估函数

```
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

也可以自定义一个Theano/TensorFlow函数并使用之

```
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

### 参数

- y_true:真实标签,theano/tensorflow张量
- y_pred:预测值, 与y_true形式相同的theano/tensorflow张量

### 返回值

单个用以代表输出各个数据点上均值的值

## 可用预定义张量

除fbeta_score额外拥有默认参数beta=1外,其他各个性能指标的参数均为y_true和y_pred

- binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率
- categorical_accuracy:对多分类问题,计算再所有预测值上的平均正确率
- sparse_categorical_accuracy:与`categorical_accuracy`相同,在对稀疏的目标值预测时有用
- top_k_categorical_accracy: 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
- sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况

## 定制评估函数

定制的评估函数可以在模型编译时传入,该函数应该以`(y_true, y_pred)`为参数,并返回单个张量,或从`metric_name`映射到`metric_value`的字典,下面是一个示例:

```
(y_true, y_pred) as arguments and return a single tensor value.

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

# 初始化方法

初始化方法定义了对Keras层设置初始化权重的方法

不同的层可能使用不同的关键字来传递初始化方法，一般来说指定初始化方法的关键字是`kernel_initializer` 和 `bias_initializer`，例如：

```
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

一个初始化器可以由字符串指定（必须是下面的预定义初始化器之一），或一个callable的函数，例如

```
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```

## Initializer

Initializer是所有初始化方法的父类，不能直接使用，如果想要定义自己的初始化方法，请继承此类。

## 预定义初始化方法

### Zeros

```
keras.initializers.Zeros()
```

全零初始化

### Ones

```
keras.initializers.Ones()
```

全1初始化

### Constant

```
keras.initializers.Constant(value=0)
```

初始化为固定值value

### RandomNormal

```
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
```

正态分布初始化

- mean：均值
- stddev：标准差
- seed：随机数种子

### RandomUniform

```
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

均匀分布初始化 *minval：均匀分布下边界* maxval：均匀分布上边界 * seed：随机数种子

### TruncatedNormal

```
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

截尾高斯分布初始化，该初始化方法与RandomNormal类似，但位于均值两个标准差以外的数据将会被丢弃并重新生成，形成截尾分布。该分布是神经网络权重和滤波器的推荐初始化方法。

- mean：均值
- stddev：标准差
- seed：随机数种子

### VarianceScaling

```
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

该初始化方法能够自适应目标张量的shape。

当`distribution="normal"`时，样本从0均值，标准差为sqrt(scale / n)的截尾正态分布中产生。其中：

```
* 当```mode = "fan_in"```时，权重张量的输入单元数。
* 当```mode = "fan_out"```时，权重张量的输出单元数
* 当```mode = "fan_avg"```时，权重张量的输入输出单元数的均值
```

当`distribution="uniform"`时，权重从[-limit, limit]范围内均匀采样，其中limit = limit = sqrt(3 * scale / n)

- scale: 放缩因子，正浮点数
- mode: 字符串，“fan_in”，“fan_out”或“fan_avg”fan_in", "fan_out", "fan_avg".
- distribution: 字符串，“normal”或“uniform”.
- seed: 随机数种子

### Orthogonal

```
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

用随机正交矩阵初始化

- gain: 正交矩阵的乘性系数
- seed：随机数种子

参考文献：[Saxe et al.](http://arxiv.org/abs/1312.6120)

### Identiy

```
keras.initializers.Identity(gain=1.0)
```

使用单位矩阵初始化，仅适用于2D方阵

- gain：单位矩阵的乘性系数

### lecun_uniform

```
lecun_uniform(seed=None)
```

LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(3 / fan_in), fin_in是权重向量的输入单元数（扇入）

- seed：随机数种子

参考文献：[LeCun 98, Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### lecun_normal

```
lecun_normal(seed=None)
```

LeCun正态分布初始化方法，参数由0均值，标准差为stddev = sqrt(1 / fan_in)的正态分布产生，其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）

- seed：随机数种子

参考文献：

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### glorot_normal

```
glorot_normal(seed=None)
```

Glorot正态分布初始化方法，也称作Xavier正态分布初始化，参数由0均值，标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）

- seed：随机数种子

参考文献：[Glorot & Bengio, AISTATS 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

### glorot_uniform

```
glorot_uniform(seed=None)
```

Glorot均匀分布初始化方法，又成Xavier均匀初始化，参数从[-limit, limit]的均匀分布产生，其中limit为`sqrt(6 / (fan_in + fan_out))`。fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数。

- seed：随机数种子

参考文献：[Glorot & Bengio, AISTATS 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

### he_normal

```
he_normal(seed=None)
```

He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入

- seed：随机数种子

参考文献：[He et al](http://arxiv.org/abs/1502.01852)

### he_uniform

```
he_normal(seed=None)
```

LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(6 / fan_in), fin_in是权重向量的输入单元数（扇入）

- seed：随机数种子

参考文献：[He et al](http://arxiv.org/abs/1502.01852)

## 自定义初始化器

如果需要传递自定义的初始化器，则该初始化器必须是callable的，并且接收`shape`（将被初始化的张量shape）和`dtype`（数据类型）两个参数，并返回符合`shape`和`dtype`的张量。

```
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, init=my_init))
```

# 正则项

正则项在优化过程中层的参数或层的激活值添加惩罚项，这些惩罚项将与损失函数一起作为网络的最终优化目标

惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但`Dense, Conv1D, Conv2D, Conv3D`具有共同的接口。

这些层有三个关键字参数以施加正则项：

- `kernel_regularizer`：施加在权重上的正则项，为`keras.regularizer.Regularizer`对象
- `bias_regularizer`：施加在偏置向量上的正则项，为`keras.regularizer.Regularizer`对象
- `activity_regularizer`：施加在输出上的正则项，为`keras.regularizer.Regularizer`对象

## 例子

```
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 可用正则项

```
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0.)
```

## 开发新的正则项

任何以权重矩阵作为输入并返回单个数值的函数均可以作为正则项，示例：

```
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)
```

可参考源代码[keras/regularizer.py](https://github.com/fchollet/keras/blob/master/keras/regularizers.py)



# 约束项

来自`constraints`模块的函数在优化过程中为网络的参数施加约束

惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但`Dense, Conv1D, Conv2D, Conv3D`具有共同的接口。

这些层通过一下关键字施加约束项

- `kernel_constraint`：对主权重矩阵进行约束
- `bias_constraint`：对偏置向量进行约束

```
from keras.constraints import maxnorm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 预定义约束项

- max_norm(m=2)：最大模约束
- non_neg()：非负性约束
- unit_norm()：单位范数约束, 强制矩阵沿最后一个轴拥有单位范数
- min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0): 最小/最大范数约束

# 回调函数Callbacks

回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。通过传递回调函数列表到模型的`.fit()`中，即可在给定的训练阶段调用该函数集中的函数。

【Tips】虽然我们称之为回调“函数”，但事实上Keras的回调函数是一个类，回调函数只是习惯性称呼

## Callback

```
keras.callbacks.Callback()
```

这是回调函数的抽象类，定义新的回调函数必须继承自该类

### 类属性

- params：字典，训练参数集（如信息显示方法verbosity，batch大小，epoch数）
- model：`keras.models.Model`对象，为正在训练的模型的引用

回调函数以字典`logs`为参数，该字典包含了一系列与当前batch或epoch相关的信息。

目前，模型的`.fit()`中有下列参数会被记录到`logs`中：

- 在每个epoch的结尾处（on_epoch_end），`logs`将包含训练的正确率和误差，`acc`和`loss`，如果指定了验证集，还会包含验证集正确率和误差`val_acc)`和`val_loss`，`val_acc`还额外需要在`.compile`中启用`metrics=['accuracy']`。
- 在每个batch的开始处（on_batch_begin）：`logs`包含`size`，即当前batch的样本数
- 在每个batch的结尾处（on_batch_end）：`logs`包含`loss`，若启用`accuracy`则还包含`acc`

------

## BaseLogger

```
keras.callbacks.BaseLogger()
```

该回调函数用来对每个epoch累加`metrics`指定的监视指标的epoch平均值

该回调函数在每个Keras模型中都会被自动调用

------

## ProgbarLogger

```
keras.callbacks.ProgbarLogger()
```

该回调函数用来将`metrics`指定的监视指标输出到标准输出上

------

## History

```
keras.callbacks.History()
```

该回调函数在Keras模型上会被自动调用，`History`对象即为`fit`方法的返回值

------

## ModelCheckpoint

```
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

该回调函数将在每个epoch后保存模型到`filepath`

`filepath`可以是格式化的字符串，里面的占位符将会被`epoch`值和传入`on_epoch_end`的`logs`关键字所填入

例如，`filepath`若为`weights.{epoch:02d-{val_loss:.2f}}.hdf5`，则会生成对应epoch和验证集loss的多个文件。

### 参数

- filename：字符串，保存模型的路径
- monitor：需要监视的值
- verbose：信息展示模式，0或1
- save_best_only：当设置为`True`时，将只保存在验证集上性能最好的模型
- mode：‘auto’，‘min’，‘max’之一，在`save_best_only=True`时决定性能最佳模型的评判准则，例如，当监测值为`val_acc`时，模式应为`max`，当检测值为`val_loss`时，模式应为`min`。在`auto`模式下，评价准则由被监测值的名字自动推断。
- save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
- period：CheckPoint之间的间隔的epoch数

------

## EarlyStopping

```
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
```

当监测值不再改善时，该回调函数将中止训练

### 参数

- monitor：需要监视的量
- patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过`patience`个epoch后停止训练。
- verbose：信息展示模式
- mode：‘auto’，‘min’，‘max’之一，在`min`模式下，如果检测值停止下降则中止训练。在`max`模式下，当检测值不再上升则停止训练。

------

## RemoteMonitor

```
keras.callbacks.RemoteMonitor(root='http://localhost:9000')
```

该回调函数用于向服务器发送事件流，该回调函数需要`requests`库

### 参数

- root：该参数为根url，回调函数将在每个epoch后把产生的事件流发送到该地址，事件将被发往`root + '/publish/epoch/end/'`。发送方法为HTTP POST，其`data`字段的数据是按JSON格式编码的事件字典。

------

## LearningRateScheduler

```
keras.callbacks.LearningRateScheduler(schedule)
```

该回调函数是学习率调度器

### 参数

- schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）

------

## TensorBoard

```
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
```

该回调函数是一个可视化的展示器

TensorBoard是TensorFlow提供的可视化工具，该回调函数将日志信息写入TensorBorad，使得你可以动态的观察训练和测试指标的图像以及不同层的激活值直方图。

如果已经通过pip安装了TensorFlow，我们可通过下面的命令启动TensorBoard：

```
tensorboard --logdir=/full_path_to_your_logs
```

更多的参考信息，请点击[这里](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

### 参数

- log_dir：保存日志文件的地址，该文件将被TensorBoard解析以用于可视化
- histogram_freq：计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。
- write_graph: 是否在Tensorboard上可视化图，当设为True时，log文件可能会很大
- write_images: 是否将模型权重以图片的形式可视化
- embeddings_freq: 依据该频率(以epoch为单位)筛选保存的embedding层
- embeddings_layer_names:要观察的层名称的列表，若设置为None或空列表，则所有embedding层都将被观察。
- embeddings_metadata: 字典，将层名称映射为包含该embedding层元数据的文件名，参考[这里](https://keras.io/https__://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)获得元数据文件格式的细节。如果所有的embedding层都使用相同的元数据文件，则可传递字符串。

------

## ReduceLROnPlateau

```
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

当评价指标不在提升时，减少学习率

当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在`patience`个epoch中看不到模型性能提升，则减少学习率

### 参数

- monitor：被监测的量
- factor：每次减少学习率的因子，学习率将以`lr = lr*factor`的形式被减少
- patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
- mode：‘auto’，‘min’，‘max’之一，在`min`模式下，如果检测值触发学习率减少。在`max`模式下，当检测值不再上升则触发学习率减少。
- epsilon：阈值，用来确定是否进入检测值的“平原区”
- cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
- min_lr：学习率的下限

## CSVLogger

```
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

将epoch的训练结果保存在csv文件中，支持所有可被转换为string的值，包括1D的可迭代数值如np.ndarray.

### 参数

- fiename：保存的csv文件名，如`run/log.csv`
- separator：字符串，csv分隔符
- append：默认为False，为True时csv文件如果存在则继续写入，为False时总是覆盖csv文件

## LambdaCallback

```
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

用于创建简单的callback的callback类

该callback的匿名函数将会在适当的时候调用，注意，该回调函数假定了一些位置参数`on_eopoch_begin`和`on_epoch_end`假定输入的参数是`epoch, logs`. `on_batch_begin`和`on_batch_end`假定输入的参数是`batch, logs`，`on_train_begin`和`on_train_end`假定输入的参数是`logs`

### 参数

- on_epoch_begin: 在每个epoch开始时调用
- on_epoch_end: 在每个epoch结束时调用
- on_batch_begin: 在每个batch开始时调用
- on_batch_end: 在每个batch结束时调用
- on_train_begin: 在训练开始时调用
- on_train_end: 在训练结束时调用

### 示例

```
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Plot the loss after every epoch.
import numpy as np
import matplotlib.pyplot as plt
plot_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                      logs['loss']))

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
    p.terminate() for p in processes if p.is_alive()])

model.fit(...,
      callbacks=[batch_print_callback,
         plot_loss_callback,
         cleanup_callback])
```

## 编写自己的回调函数

我们可以通过继承`keras.callbacks.Callback`编写自己的回调函数，回调函数通过类成员`self.model`访问访问，该成员是模型的一个引用。

这里是一个简单的保存每个batch的loss的回调函数：

```
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

### 例子：记录损失函数的历史数据

```
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
```

### 例子：模型检查点

```
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test)
```

# 模型可视化

`keras.utils.vis_utils`模块提供了画出Keras模型的函数（利用graphviz）

该函数将画出模型结构图，并保存成图片：

```
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`接收两个可选参数：

- `show_shapes`：指定是否显示输出数据的形状，默认为`False`
- `show_layer_names`:指定是否显示层名称,默认为`True`

我们也可以直接获取一个`pydot.Graph`对象，然后按照自己的需要配置它，例如，如果要在ipython中展示图片

```
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

【Tips】依赖 pydot-ng 和 graphviz，若出现错误，用命令行输入`pip install pydot-ng & brew install graphviz`

# Scikit-Learn接口包装器

我们可以通过包装器将`Sequential`模型（仅有一个输入）作为Scikit-Learn工作流的一部分，相关的包装器定义在`keras.wrappers.scikit_learn.py`中

目前，有两个包装器可用：

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`实现了sklearn的分类器接口

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`实现了sklearn的回归器接口

## 参数

- build_fn：可调用的函数或类对象
- sk_params：模型参数和训练参数

`build_fn`应构造、编译并返回一个Keras模型，该模型将稍后用于训练/测试。`build_fn`的值可能为下列三种之一：

1. 一个函数
2. 一个具有`call`方法的类对象
3. None，代表你的类继承自`KerasClassifier`或`KerasRegressor`，其`call`方法为其父类的`call`方法

`sk_params`以模型参数和训练（超）参数作为参数。合法的模型参数为`build_fn`的参数。注意，‘build_fn’应提供其参数的默认值。所以我们不传递任何值给`sk_params`也可以创建一个分类器/回归器

`sk_params`还接受用于调用`fit`，`predict`，`predict_proba`和`score`方法的参数，如`nb_epoch`，`batch_size`等。这些用于训练或预测的参数按如下顺序选择：

1. 传递给`fit`，`predict`，`predict_proba`和`score`的字典参数
2. 传递个`sk_params`的参数
3. `keras.models.Sequential`，`fit`，`predict`，`predict_proba`和`score`的默认值

当使用scikit-learn的`grid_search`接口时，合法的可转换参数是你可以传递给`sk_params`的参数，包括训练参数。即，你可以使用`grid_search`来搜索最佳的`batch_size`或`nb_epoch`以及其他模型参数