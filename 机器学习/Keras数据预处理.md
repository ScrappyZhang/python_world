# 序列预处理

## 填充序列pad_sequences

```
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='pre', value=0.)
```

将长为`nb_samples`的序列（标量序列）转化为形如`(nb_samples,nb_timesteps)`2D numpy array。如果提供了参数`maxlen`，`nb_timesteps=maxlen`，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。长于`nb_timesteps`的序列将会被截断，以使其匹配目标长度。padding和截断发生的位置分别取决于`padding`和`truncating`.

### 参数

- sequences：浮点数或整数构成的两层嵌套列表
- maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
- dtype：返回的numpy array的数据类型
- padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
- truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
- value：浮点数，此值将在填充时代替默认的填充值0

### 返回值

返回形如`(nb_samples,nb_timesteps)`的2D张量

------

## 跳字skipgrams

```
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size,
    window_size=4, negative_samples=1., shuffle=True,
    categorical=False, sampling_table=None)
```

skipgrams将一个词向量下标的序列转化为下面的一对tuple：

- 对于正样本，转化为（word，word in the same window）
- 对于负样本，转化为（word，random word from the vocabulary）

【Tips】根据维基百科，n-gram代表在给定序列中产生连续的n项，当序列句子时，每项就是单词，此时n-gram也称为shingles。而skip-gram的推广，skip-gram产生的n项子序列中，各个项在原序列中不连续，而是跳了k个字。例如，对于句子：

“the rain in Spain falls mainly on the plain”

其 2-grams为子序列集合：

the rain，rain in，in Spain，Spain falls，falls mainly，mainly on，on the，the plain

其 1-skip-2-grams为子序列集合：

the in, rain Spain, in falls, Spain mainly, falls on, mainly the, on plain.

更多详情请参考[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

### 参数

- sequence：下标的列表，如果使用sampling_tabel，则某个词的下标应该为它在数据库中的顺序。（从1开始）
- vocabulary_size：整数，字典大小
- window_size：整数，正样本对之间的最大距离
- negative_samples：大于0的浮点数，等于0代表没有负样本，等于1代表负样本与正样本数目相同，以此类推（即负样本的数目是正样本的`negative_samples`倍）
- shuffle：布尔值，确定是否随机打乱样本
- categorical：布尔值，确定是否要使得返回的标签具有确定类别
- sampling_table：形如`(vocabulary_size,)`的numpy array，其中`sampling_table[i]`代表没有负样本或随机负样本。等于1为与正样本的数目相同 采样到该下标为i的单词的概率（假定该单词是数据库中第i常见的单词）

### 输出

函数的输出是一个`(couples,labels)`的元组，其中：

- `couples`是一个长为2的整数列表：`[word_index,other_word_index]`
- `labels`是一个仅由0和1构成的列表，1代表`other_word_index`在`word_index`的窗口，0代表`other_word_index`是词典里的随机单词。
- 如果设置`categorical`为`True`，则标签将以one-hot的方式给出，即1变为[0,1]，0变为[1,0]

------

## 获取采样表make_sampling_table

```
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-5)
```

该函数用以产生`skipgrams`中所需要的参数`sampling_table`。这是一个长为`size`的向量，`sampling_table[i]`代表采样到数据集中第i常见的词的概率（为平衡期起见，对于越经常出现的词，要以越低的概率采到它）

### 参数

- size：词典的大小
- sampling_factor：此值越低，则代表采样时更缓慢的概率衰减（即常用的词会被以更低的概率被采到），如果设置为1，则代表不进行下采样，即所有样本被采样到的概率都是1。



# 文本预处理

## 句子分割text_to_word_sequence

```
keras.preprocessing.text.text_to_word_sequence(text,
                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
```

本函数将一个句子拆分成单词构成的列表

### 参数

- text：字符串，待处理的文本
- filters：需要滤除的字符的列表或连接形成的字符串，例如标点符号。默认值为 '!"#$%&()*+,-./:;<=>?@[]^_`{|}~\t\n'，包含标点符号，制表符和换行符等
- lower：布尔值，是否将序列设为小写形式
- split：字符串，单词的分隔符，如空格

### 返回值

字符串列表

------

## one-hot编码

```
keras.preprocessing.text.one_hot(text,
                                 n,
                                 filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" ")
```

本函数将一段文本编码为one-hot形式的码，即仅记录词在词典中的下标。

【Tips】 从定义上，当字典长为n时，每个单词应形成一个长为n的向量，其中仅有单词本身在字典中下标的位置为1，其余均为0，这称为one-hot。

为了方便起见，函数在这里仅把“1”的位置，即字典中词的下标记录下来。

### 参数

- n：整数，字典长度

### 返回值

整数列表，每个整数是[1,n]之间的值，代表一个单词（不保证唯一性，即如果词典长度不够，不同的单词可能会被编为同一个码）。

------

## 特征哈希hashing_trick

```
keras.preprocessing.text.hashing_trick(text,
                                       n,
                                       hash_function=None,
                                       filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                       lower=True,
                                       split=' ')
```

将文本转换为固定大小的哈希空间中的索引序列

### 参数

- n: 哈希空间的维度
- hash_function: 默认为 python `hash` 函数, 可以是 'md5' 或任何接受输入字符串, 并返回 int 的函数. 注意 `hash` 不是一个稳定的哈希函数, 因此在不同执行环境下会产生不同的结果, 作为对比, 'md5' 是一个稳定的哈希函数.

### 返回值

整数列表

## 分词器Tokenizer

```
keras.preprocessing.text.Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
```

Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。

### 构造参数

- 与`text_to_word_sequence`同名参数含义相同
- num_words：None或整数，处理的最大单词数量。若被设置为整数，则分词器将被限制为待处理数据集中最常见的`num_words`个单词
- char_level: 如果为 True, 每个字符将被视为一个标记

### 类方法

- fit_on_texts(texts)
  - texts：要用以训练的文本列表
- texts_to_sequences(texts)
  - texts：待转为序列的文本列表
  - 返回值：序列的列表，列表中每个序列对应于一段输入文本
- texts_to_sequences_generator(texts)
  - 本函数是`texts_to_sequences`的生成器函数版
  - texts：待转为序列的文本列表
  - 返回值：每次调用返回对应于一段输入文本的序列
- texts_to_matrix(texts, mode)：
  - texts：待向量化的文本列表
  - mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，默认为‘binary’
  - 返回值：形如`(len(texts), nb_words)`的numpy array
- fit_on_sequences(sequences):
  - sequences：要用以训练的序列列表
- sequences_to_matrix(sequences):
  - sequences：待向量化的序列列表
  - mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，默认为‘binary’
  - 返回值：形如`(len(sequences), nb_words)`的numpy array

### 属性

- word_counts:字典，将单词（字符串）映射为它们在训练期间出现的次数。仅在调用fit_on_texts之后设置。
- word_docs: 字典，将单词（字符串）映射为它们在训练期间所出现的文档或文本的数量。仅在调用fit_on_texts之后设置。
- word_index: 字典，将单词（字符串）映射为它们的排名或者索引。仅在调用fit_on_texts之后设置。
- document_count: 整数。分词器被训练的文档（文本或者序列）数量。仅在调用fit_on_texts或fit_on_sequences之后设置。

# 图片预处理

## 图片生成器ImageDataGenerator

```
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```

用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。

### 参数

- featurewise_center：布尔值，使输入数据集去中心化（均值为0）, 按feature执行
- samplewise_center：布尔值，使输入数据的每个样本均值为0
- featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
- samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差
- zca_whitening：布尔值，对输入数据施加ZCA白化
- zca_epsilon: ZCA使用的eposilon，默认1e-6
- rotation_range：整数，数据提升时图片随机转动的角度
- width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
- height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
- shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
- zoom_range：浮点数或形如`[lower,upper]`的列表，随机缩放的幅度，若为浮点数，则相当于`[lower,upper] = [1 - zoom_range, 1+zoom_range]`
- channel_shift_range：浮点数，随机通道偏移的幅度
- fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
- cval：浮点数或整数，当`fill_mode=constant`时，指定要向超出边界的点填充的值
- horizontal_flip：布尔值，进行随机水平翻转
- vertical_flip：布尔值，进行随机竖直翻转
- rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
- preprocessing_function: 将被应用于每个输入的函数。该函数将在图片缩放和数据提升之后运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
- data_format：字符串，“channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是`~/.keras/keras.json`中设置的值，若从未设置过，则为“channel_last”

------

### 方法

- fit(x, augment=False, rounds=1)：计算依赖于数据的变换所需要的统计信息(均值方差等),只有使用`featurewise_center`，`featurewise_std_normalization`或`zca_whitening`时需要此函数。
  - X：numpy array，样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
  - augment：布尔值，确定是否使用随即提升过的数据
  - round：若设`augment=True`，确定要在数据上进行多少轮数据提升，默认值为1
  - seed: 整数,随机数种子
- flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')：接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
  - x：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
  - y：标签
  - batch_size：整数，默认32
  - shuffle：布尔值，是否随机打乱数据，默认为True
  - save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
  - save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了`save_to_dir`时生效
  - save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
  - yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
  - seed: 整数,随机数种子
- flow_from_directory(directory): 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
  - directory: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用.详情请查看[此脚本](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
  - target_size: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
  - color_mode: 颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
  - classes: 可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将从`directory`下的子文件夹名称/结构自动推断。每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性`class_indices`可获得文件夹名与类的序号的对应字典。
  - class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用`model.predict_generator()`和`model.evaluate_generator()`等函数时会用到.
  - batch_size: batch数据的大小,默认32
  - shuffle: 是否打乱数据,默认为True
  - seed: 可选参数,打乱数据和进行变换时的随机数种子
  - save_to_dir: None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
  - save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了`save_to_dir`时生效
  - save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
  - flollow_links: 是否访问子文件夹中的软链接

### 例子

使用`.flow()`的例子

```
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train), epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print 'Epoch', e
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        loss = model.train(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

使用`.flow_from_directory(directory)`的例子

```
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

同时变换图像和mask

```
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```