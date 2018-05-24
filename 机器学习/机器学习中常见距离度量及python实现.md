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

有M个样本向量`X1~Xm`，协方差矩阵记为S， 均值记为向量$\mu $。则其中样本向量X到$\mu $的马氏距离表示为:
$$
D(X)=\sqrt [ 2 ]{ (X-\mu )^TS^{-1}(X-\mu )}
$$
而其中向量Xi和Xj之间的马氏距离定义为：
$$
D(X_i,X_j)=\sqrt [ 2 ]{ (X_i-X_j)^TS^{-1}(X_i-X_j )}
$$
若协方差矩阵是单位矩阵，则各个样本向量之间独立同分布，马氏距离就是欧式距离了。若协方差矩阵是对角矩阵，马氏距离就是标准化距离。

因此，

- 马氏距离的计算是建立在总体样本的基础上的，这一点可以从上述协方差矩阵的解释中可以得到。即，若通用取两个样本，放入不同的总体中，计算出来的马氏距离通常是不相同的。
- 在计算马氏距离过程中，要求总体样本数大于样本的维数，否则得到的总体样本协方差矩阵逆矩阵不存在，这种情况下，用欧式距离计算即可。
- 还有一种情况，满足了条件总体样本数大于样本的维数，但是协方差矩阵的逆矩阵仍然不存在，比如三个样本点（3，4），（5，6）和（7，8），这种情况是因为这三个样本在其所处的二维空间平面内共线。这种情况下，也采用欧式距离计算。
- 在实际应用中“总体样本数大于样本的维数”这个条件是很容易满足的，而所有样本点出现3）中所描述的情况是很少出现的，所以在绝大多数情况下，马氏距离是可以顺利计算的，但是马氏距离的计算是不稳定的，不稳定的来源是协方差矩阵，这也是马氏距离与欧式距离的最大差异之处。
- **优点：**它不受量纲的影响，两点之间的马氏距离与原始数据的测量单位无关；由标准化数据和中心化数据(即原始数据与均值之差）计算出的二点之间的马氏距离相同。马氏距离还可以排除变量之间的相关性的干扰。
- **缺点：**它的缺点是夸大了变化微小的变量的作用。

### python中实现

```
#马氏距离要求样本数要大于维数，否则无法求协方差矩阵
#此处进行转置，表示10个样本，每个样本2维
X=np.vstack([x,y])
XT=X.T

#方法一：根据公式求解
S=np.cov(X)   #两个维度之间协方差矩阵
SI = np.linalg.inv(S) #协方差矩阵的逆矩阵
#马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
n=XT.shape[0]
d1=[]
for i in range(0,n):
    for j in range(i+1,n):
        delta=XT[i]-XT[j]
        d=np.sqrt(np.dot(np.dot(delta,SI),delta.T))
        d1.append(d)
        
#方法二：根据scipy库求解
from scipy.spatial.distance import pdist
d2=pdist(XT,'mahalanobis')
```

## 7. 余弦距离 Cosine Distance

几何中夹角余弦可用来衡量两个向量方向的差异，机器学习中借用这一概念来衡量样本向量之间的差异。

余弦距离就是用1减去余弦相似度（余弦公式）

- 二维平面两点`a(x1, y1)`与`b(x2, y2)`间的余弦公式

$$
\cos { \theta  } =\frac { x_{ 1 }x_{ 2 }+y_{ 1 }y_{ 2 } }{ \sqrt { x_1^2+y_1^2 } \sqrt { x_2^2+y_2^2  }  } 
$$

- 两个n维向量`a(x11, x12, …, x1n)`与`b(x21, x22, …, x2n)`间的余弦公式

$$
\cos { \theta  } =\frac { \sum _{ k=1 }^{ n }{ x_{ 1k }x_{ 2k } }  }{ \sqrt { \sum _{ k=1 }^{ n }{ x_{1k}^2 }  } \sqrt { \sum _{ k=1 }^{ n }{ x_{2k}^2 }  }  } 
$$

>  余弦取值范围为[-1,1]。求得两个向量的夹角，并得出夹角对应的余弦值，此余弦值就可以用来表征这两个向量的相似性。夹角越小，趋近于0度，余弦值越接近于1，它们的方向更加吻合，则越相似。当两个向量的方向完全相反夹角余弦取最小值-1。当余弦值为0时，两向量正交，夹角为90度。因此可以看出，余弦相似度与向量的幅值无关，只与向量的方向相关。

**缺点:** 余弦相似度只与向量方向有关，但它会受到向量的平移影响，在夹角余弦公式中如果将 x 平移到 x+1, 余弦值就会改变。

### python中实现

```
代码中为实现余弦相似度

方式一：
d1=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

方式二：
from scipy.spatial.distance import pdist
d2=pdist(X,'seuclidean')

方式三：
from scipy.spatial.distance import cosine
d3 = 1 - cosine(x, y)
```

## 8. 皮尔逊相关系数（Pearson correlation）

怎样才能实现平移不变性？这就要用到**皮尔逊相关系数**（Pearson correlation），有时候也直接叫**相关系数**。
$$
\rho _{ XY }=\frac { Cov(X,Y) }{ \sqrt { D(X) } \sqrt { D(Y) }  } =\frac { E((X-EX)(Y-EY)) }{ \sqrt { D(X) } \sqrt { D(Y) }  } 
$$
夹角余弦公式：
$$
CosSim(x,y)=\frac { \sum { x_iy_i }  }{ \sqrt { \sum { x_i^2 }  } \sqrt {  \sum { y_i^2 }  }  } =\frac { <x,y> }{ \left\| x \right\| \left\| y \right\|  } 
$$
皮尔逊相关系数可表示为:
$$
Corr(x,y)=\frac { \sum { (x_{ i }-\bar { x } )(y_{ i }-\bar { y } ) }  }{ \sqrt { \sum { (x_{ i }-\bar { x } )^{ 2 } }  } \sqrt { \sum { (y_{ i }-\bar { y } ) ^{ 2 } }  }  } =\frac { <x-\bar { x } ,y-\bar { y }  > }{ \left\| x-\bar { x }  \right\| \left\| y-\bar { y }   \right\|  } =CosSim(x-\bar { x } ,y-\bar { y } )
$$
可以很明显看到皮尔逊相关系数具有平移不变性和尺度不变性，计算出了两个向量（维度）的相关性。相关系数是衡量随机变量X与Y相关程度的一种方法，相关系数的取值范围是[-1,1]。相关系数的绝对值越大，则表明X与Y相关度越高。当X与Y线性相关时，相关系数取值为1（正线性相关）或-1（负线性相关）。

### python中实现

```
#方法一：根据公式求解
x_=x-np.mean(x)
y_=y-np.mean(y)
d1=np.dot(x_,y_)/(np.linalg.norm(x_)*np.linalg.norm(y_))

#方法二：根据numpy库求解
X=np.vstack([x,y])
d2=np.corrcoef(X)[0][1]

#方法三：
from scipy.spatial.distance import correlation
d3 = 1 - correlation(x, y)
```

## 9. 汉明距离 Hamming distance

两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数。例如字符串`“1111”`与`“1001”`之间的汉明距离为2。        

应用：信息编码（为了增强容错性，应使得编码间的最小汉明距离尽可能大）。

### python中实现

```
import numpy as np
from scipy.spatial.distance import pdist
x=np.random.random(10)>0.5
y=np.random.random(10)>0.5

x_=np.asarray(x,np.int32)
y_=np.asarray(y,np.int32)

#方法一：根据公式求解
d1=np.mean(x_!=y_)

#方法二：根据scipy库求解
X=np.vstack([x_,y_])
d2=pdist(X,'hamming')

#方法三:
from scipy.spatial.distance import hamming
hamming(x_, y_)
```

## 10. 杰卡德相似系数 Jaccard similarity coefficient

两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。
$$
J(A,B)=\frac { \left| A\cap B \right|  }{ \left| A\cup B \right|  }
$$
杰卡德相似系数是衡量两个集合的相似度一种指标。

 与杰卡德相似系数相反的概念是**杰卡德距离****(**Jaccard distance)。杰卡德距离可用如下公式表示：
$$
J_\delta (A,B)=1-J(A,B)=1-\frac { \left| A\cap B \right|  }{ \left| A\cup B \right|  } 
$$

### python中实现

```
import numpy as np
from scipy.spatial.distance import pdist
x=np.random.random(10)>0.5
y=np.random.random(10)>0.5

x_=np.asarray(x,np.int32)
y_=np.asarray(y,np.int32)

#方法一：根据公式求解
up=np.double(np.bitwise_and((x_ != y_),np.bitwise_or(x_ != 0, y_ != 0)).sum())
down=np.double(np.bitwise_or(x_ != 0, y_ != 0).sum())
d1=(up/down)
           

#方法二：根据scipy库求解
X=np.vstack([x_,y_])
d2=pdist(X,'jaccard')

#方法三:
from scipy.spatial.distance import jaccard
jaccard(x_, y_)
```

## 11. 布雷柯蒂斯距离 Bray Curtis Distance

Bray Curtis距离主要用于生态学和环境科学，计算坐标之间的距离。该距离取值在[0,1]之间。它也可以用来计算样本之间的差异。
$$
d_{ ij }=\frac { \sum _{ k=1 }^{ n }{ \left| x_{ ik }-y_{ jk } \right|  }  }{ \sum _{ k=1 }^{ n }{ x_{jk} } \sum _{ k=1 }^{ n }{ y_{jk} }  } 
$$
例如：

```
    a	b	c	d	e	sum
s29 11	0	7	8	0	26
s30	24	37	5	18	1	85
```

$$
\frac { \left| 11-24 \right| +\left| 0-37 \right| +\left| 7-5 \right| +\left| 8-18 \right| +\left| 0-1 \right|  }{ 26+85 } =0.568
$$

### python中实现

```
import numpy as np
from scipy.spatial.distance import pdist
x=np.array([11,0,7,8,0])
y=np.array([24,37,5,18,1])

#方法一：根据公式求解
up=np.sum(np.abs(y-x))
down=np.sum(x)+np.sum(y)
d1=(up/down)
           
#方法二：根据scipy库求解
X=np.vstack([x,y])
d2=pdist(X,'braycurtis')

#方法三:
from scipy.spatial.distance import braycurtis
braycurtis(x,y)
```

## 12. 编辑距离 Levenshtein Distance

**编辑距离，**又称**Levenshtein距离**，是指两个字串之间，由一个转成另一个所需的最少编辑操作次数。

许可的编辑操作包括：将一个字符替换成另一个字符，插入一个字符，删除一个字符。

例如将eeba转变成abac：

1. eba（删除第一个e）
2. aba（将剩下的e替换成a）
3. abac（在末尾插入c）

所以eeba和abac的编辑距离就是3

它由俄罗斯科学家Vladimir Levenshtein在1965年提出这个概念。

### **算法：**

算法就是简单的线性动态规划（最长上升子序列就属于线性动态规划）。

设我们要将s1变成s2

定义状态矩阵edit[len1][len2]，len1和len2分别是要比较的字符串s1和字符串s2的长度+1（+1是考虑到动归中，一个串为空的情况）

然后，定义edit[i][j]是s1中前i个字符组成的串，和s2中前j个字符组成的串的编辑距离

具体思想是，对于每个i，j从0开始依次递增，对于每一次j++，由于前j-1个字符跟i的编辑距离已经求出，所以只用考虑新加进来的第j个字符即可

**插入操作：**在s1的前i个字符后插入一个字符ch，使得ch等于新加入的s2[j]。于是插入字符ch的编辑距离就是edit[i][j-1]+1

**删除操作：**删除s1[i]，以期望s1[i-1]能与s2[j]匹配（如果s1[i-1]前边的几个字符能与s2[j]前边的几个字符有较好的匹配，那么这么做就能得到更好的结果）。另外，对于s1[i-1]之前的字符跟s2[j]匹配的情况，edit[i-1][j]中已经考虑过。于是删除字符ch的编辑距离就是edit[i-1][j]+1

**替换操作：**期望s1[i]与s2[j]匹配，或者将s1[i]替换成s2[j]后匹配。于是替换操作的编辑距离就是edit[i-1][j-1]+f(i,j)。其中，当s1[i]==s2[j]时，f(i,j)为0；反之为1

于是动态规划公式如下：

- if i == 0 且 j == 0，edit(i, j) = 0
- if i == 0 且 j > 0，edit(i, j) = j
- if i > 0 且j == 0，edit(i, j) = i
- if 0 < i ≤ 1  且 0 < j ≤ 1 ，edit(i, j) == min{ edit(i-1, j) + 1, edit(i, j-1) + 1, edit(i-1, j-1) + f(i, j) }，当第一个字符串的第i个字符不等于第二个字符串的第j个字符时，f(i, j) = 1；否则，f(i, j) = 0。

### python中实现

```
方式一：
按照算法编写，
缺点： 1. 需要一个x，y的矩阵，内存占用多， 2. 若两个样本相同部分很多，做了较多的无效比较。
class arithmetic():  
      
    def __init__(self):  
        pass  
    ''''' 【编辑距离算法】 【levenshtein distance】 【字符串相似度算法】 '''  
    def levenshtein(self,left,right, gap=1):  
        pattern = r'.{%d}' % gap #gap决定编辑时的最小单位
        left = [word for word in re.findall(pattern, left)] 
        right = [word for word in re.findall(pattern, right)] 
        # 左字符串比右字符串大时， 交换左右字符串
        if len(left) > len(right):  
            left,right = right,left  
        len_left =  len(left)
        len_right = len(right)
        # 左字符串长度为0时，直接返回右字符串的长度作为编辑距离
        if len(left) == 0:  
            return len(right)  
        # 右字符串长度为0时，直接返回左字符串的长度作为编辑距离
        if len(right) == 0:  
            return len(left) 


        left_length = len(left) + 1  
        right_length = len(right) + 1  

        distance_matrix = [[x + y for y in range(right_length)] for x in range(left_length)]   
        for i in range(1,left_length):  
           for j in range(1,right_length): 
       			# 删除操作 
                deletion = distance_matrix[i-1][j] + 1  
                # 插入操作
                insertion = distance_matrix[i][j-1] + 1  
                # 替换操作
                substitution = distance_matrix[i-1][j-1]  
                if left[i - 1] != right[j - 1]:  
                   substitution += 1  
                   distance_matrix[i][j] = min(insertion,deletion,substitution)  
        return distance_matrix[left_length-1][right_length-1]  
```

```
方式二：针对算法实现方式1进行优化
class arithmetic():  
      
    def __init__(self):  
        pass  
    ''''' 【编辑距离算法】 【levenshtein distance】 【字符串相似度算法】 '''  
    def levenshtein(self,left,right, gap=1):  
        pattern = r'.{%d}' % gap 
        left = left.upper()
        left = [word for word in re.findall(pattern, left)] 
        right = [word for word in re.findall(pattern, right)] 
        # 左字符串比右字符串大时， 交换左右字符串
        if len(left) > len(right):  
            left,right = right,left  
        len_left =  len(left)
        len_right = len(right)
        # 左字符串长度为0时，直接返回右字符串的长度作为编辑距离
        if len(left) == 0:  
            return len(right)  
        # 右字符串长度为0时，直接返回左字符串的长度作为编辑距离
        if len(right) == 0:  
            return len(left)  

        # 先删除相同位置的相同字符
        for i in range(len_left):
            if left[i] == right[i]:
                left[i] = ' '
                right[i] = ' '
        if len_left != len_right:
            k = 0
            for j in range(len_right - len_left , len_right):
                if left[k] == right[j]:
                    left[k] = ' '
                    right[j] = ' '
                k += 1 

        while ' ' in left:
            left.remove(' ')
        while ' ' in right:
            right.remove(' ')


        if len(left) > len(right):  
            left,right = right,left  
        len_left =  len(left)
        len_right = len(right)
        # 左字符串长度为0时，直接返回右字符串的长度作为编辑距离
        if len(left) == 0:  
            return len(right)  
        # 右字符串长度为0时，直接返回左字符串的长度作为编辑距离
        if len(right) == 0:  
            return len(left)


        left_length = len(left) + 1  
        right_length = len(right) + 1  
        # 创建一个2*right_length矩阵
        distance_matrix = [[ x for x in range(right_length)] for i in range(0 , 2)]
        # pprint(distance_matrix)
        for i in range(1, left_length):
            distance_matrix[1][0] = i
            for j in range(1, right_length):
                # 删除操作 
                deletion = distance_matrix[0][j] + 1  
        #         # 插入操作
                insertion = distance_matrix[1][j-1] + 1  
        #         # 替换操作
                substitution = distance_matrix[0][j-1]  
                if left[i - 1] != right[j - 1]:  
                    substitution += 1  
                distance_matrix[1][j] = min(insertion,deletion,substitution)

            for j in range(right_length):
                distance_matrix[0][j] = distance_matrix[1][j]
            # pprint(distance_matrix)

        return distance_matrix[0][right_length - 1]
```

```
方式三：
使用python-Levenshtein
    import Levenshtein
    Levenshtein.distance(x,y)
```

