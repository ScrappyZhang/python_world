# 机器学习中常见距离度量及python实现

## 1. 欧式距离

欧式距离是最易于理解的一种距离计算方法，源自欧式空间中两点间的距离公式。

- 二维平面上两点`a(x1, y1)`与`b(x2, y2)`间的欧式距离

$$
d_{12} =\sqrt [ 2 ]{ (x_1-x_2)^2+(y_1-y_2)^2 }  
$$

- 三维空间两点`a(x1, y1, z1)`与`b(x2, y2, z2)`间的欧式距离

$$
d_{ 12 }=\sqrt [ 2 ]{ (x_{ 1 }-x_{ 2 })^{ 2 }+(y_{ 1 }-y_{ 2 })^{ 2 }+(z_1-z_2)^2 } 
$$

- 两个n维向量`a(x11, x12, …, x1n)`与`b(x21, x22, …, x2n)`间的欧式距离

$$
d_{ 12 }=\sqrt [ 2 ]{ \sum _{ k=1 }^{ n }{ (x_{1k}-x_{2k} )^2}  } 
$$

- 若表示成向量运算的形式

$$
d_{ 12 }=\sqrt [ 2 ]{ (a-b)(a-b)^T} 
$$

### python中实现：

```
import numpy as np
x = np.random.random(10)
y = np.random.random(10)

# x
array([ 0.2027935 ,  0.31394456,  0.1960384 ,  0.27455305,  0.73423524,
        0.49304154,  0.39459916,  0.93357666,  0.25406378,  0.31207493])
# y
array([ 0.58752716,  0.41383598,  0.30174012,  0.74574908,  0.57339749,
        0.32131536,  0.47729764,  0.81684854,  0.24995744,  0.47294776])
        
方式一：
d1=np.sqrt(np.sum(np.square(x-y)))
# 0.70208042137425797
方式二：
from scipy.spatial.distance import euclidean
euclidean(x, y)
# 0.702080421374258
```

## 2. 曼哈顿距离 Manhattan Distance

从一个十字路口开车到另外一个十字路口，驾驶距离往往不是两点之间的距离，而是需要按照街区拐弯的实际距离，因此，也称城市街区距离(**city block)**。

- 二维平面两点`a(x1, y1)`与`b(x2, y2)`间的曼哈顿距离

$$
d_{ 12 }=|x_1-x_2| + |y_1-y_2|
$$

- 两个n维向量`a(x11, x12, …, x1n)`与`b(x21, x22, …, x2n)`间的曼哈顿距离

$$
d_{ 12 }=\sum _{ k=1 }^{ n }{ |x_{1k}-x_{2k}| } 
$$

### python中实现：

```
方式一：
d1=np.sum(np.abs(x-y))

方式二：
from scipy.spatial.distance import cityblock
cityblock(x, y)
```

## 3. 切比雪夫距离Chebyshev Distance

来源于国际象棋中国王从一个格子(x1, y1)到另一个格子(x2, y2)最少需要的步数。它的最少步数为`max(|x2-x1|,|y2-y1|)`步。类似的距离度量方法叫做切比雪夫距离。

- 二维平面两点`a(x1, y1)`与`b(x2, y2)`间的曼哈顿距离

$$
d_{ 12 }=max(\left| x_1 - x_2  \right|, \left| y_1-y_2 \right| )
$$

- 两个n维向量`a(x11, x12, …, x1n)`与`b(x21, x22, …, x2n)`间的切比雪夫距离

$$
d_{ 12 }=\underset { i }{ max} (\left| x_{1i} - x_{2i}  \right|)
$$

以上公式等价于：
$$
d_{ 12 }=\lim _{ k\rightarrow \infty  }{ (\sum _{ i=1 }^{ n }{ (\left| x_{ 1i }-x_{ 2i } \right| ^{ k }) } )^{ \frac { 1 }{ k }  } } 
$$

### python中实现:

```python
方式一：
d1=np.max(np.abs(x-y))

方式二：
from scipy.spatial.distance import chebyshev
chebyshev(x, y)
```

## 4. 闵可夫斯基距离Minkowski Distance

- 两个n维向量`a(x11, x12, …, x1n)`与`b(x21, x22, …, x2n)`间的闵可夫斯基距离

$$
d_{ 12 }=\sqrt [ p ]{ \sum _{ k=1 }^{ n }{ \left| x_{1k}-x_{2k} \right| ^p }  } 
$$

> 当p=1时，就是曼哈顿距离
>
> 当p=2时，就是欧式距离
>
> 当$p \rightarrow \infty $时， 就是切比雪夫距离。

闵可夫距离均存在明显的缺点：例如：

一个二维样本（身高，体重），其中身高范围是`150~190`，体重范围是`50~60`， 有三个样本：`a(180, 50)`, `b(190, 50)`, `c(180,60)`。那么a与b之间的闵式距离等于a与c之间的闵式距离，但是身高10cm真的等价于体重10kg吗？所以衡量样本间相似度，是存在严重缺陷的。可以看到，闵可夫距离仅仅将各个分量的量纲当作单位看待了，而没有考虑各分量的分布（期望、方差等）的不同之处。

### python中实现

```
from scipy.spatial.distance import minkowski
minkowski(x, y, 2)
```

## 5. 标准化欧式距离 Standardized Euclidean distance

针对之前提到的欧式距离的缺点，标准欧式距离就是先将各分量标准化到均值和方差相等，然后进行欧式距离计算。

- 两个n维向量`a(x11, x12, …, x1n)`与`b(x21, x22, …, x2n)`间的标准欧式距离

$$
d_{ 12 }=\sqrt [ 2 ]{ \sum _{ k=1 }^{ n }{ (\frac { x_{1k}-x_{2k} }{ s_k } )^2 }  }
$$

> 其中Sk为标准差

### python中实现

```
X=np.vstack([x,y])
方式一：
sk=np.var(X,axis=0,ddof=1)
d1=np.sqrt(((x - y) ** 2 /sk).sum())

方式二：
from scipy.spatial.distance import pdist
d2=pdist(X,'seuclidean')
```

## 6. 马氏距离 Mahalanobis Distance

