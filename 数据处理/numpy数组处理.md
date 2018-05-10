## 1. `numpy`使用与属性

使用`numpy`首先要导入模块：

```python
import numpy as np #为了方便使用numpy 采用np简写
```

列表转矩阵：

```python
array = np.array([[1,2,3],[2,3,4]])  #列表转化为矩阵
print(array)
"""
[[1, 2, 3],
[2, 3, 4]]
"""
```

numpy三种常见属性：

- `ndim`：维度
- `shape`：行数和列数
- `size`：元素个数

```python
print('number of dim:',array.ndim)  # 维度
# number of dim: 2

print('shape :',array.shape)    # 行数和列数
# shape : (2, 3)

print('size:',array.size)   # 元素个数
# size: 6
```

## 2. `numpy`创建`array`

- `array`：创建数组

```
a = np.array([2,23,4])  # list 1d
print(a)
# [2 23 4]
```

- `dtype`：指定数据类型

```python
a = np.array([2,23,4],dtype=np.int)
print(a.dtype)
# int64

a = np.array([2,23,4],dtype=np.int32)
print(a.dtype)
# int32

a = np.array([2,23,4],dtype=np.float)
print(a.dtype)
# float64

a = np.array([2,23,4],dtype=np.float32)
print(a.dtype)
# float32
```

- `zeros`：创建全0数组

```python
a = np.zeros((3,4)) # 数据全为0，3行4列
"""
	[[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]]
"""
```

- `ones`：创建全1数据

```python
a = np.ones((3,4),dtype = np.int)   # 数据为1，3行4列
"""
	 [[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]]
"""
```

- `eye`:创建单位矩阵

```python
np.eye(3)
#[[ 1.  0.  0.]
# [ 0.  1.  0.]
# [ 0.  0.  1.]]
```

- `empty`：创建数据接近0

```python
a = np.empty((3,4)) # 数据为empty，3行4列
"""
	[[  0.00000000e+000,   4.94065646e-324,   9.88131292e-324,
          1.48219694e-323],
       [  1.97626258e-323,   2.47032823e-323,   2.96439388e-323,
          3.45845952e-323],
       [  3.95252517e-323,   4.44659081e-323,   4.94065646e-323,
          5.43472210e-323]]
"""
```

- `arrange`：按指定范围创建数据

```python
a = np.arange(10,20,2) # 10-19 的数据，2步长
"""
[10, 12, 14, 16, 18]
"""
```

- `linspace`：创建线段型数据

```python
a = np.linspace(1,10,20)    # 开始端1，结束端10，且分割成20个数据，生成线段
"""
	[  1.        ,   1.47368421,   1.94736842,   2.42105263,
         2.89473684,   3.36842105,   3.84210526,   4.31578947,
         4.78947368,   5.26315789,   5.73684211,   6.21052632,
         6.68421053,   7.15789474,   7.63157895,   8.10526316,
         8.57894737,   9.05263158,   9.52631579,  10.        ]
"""
```

- `reshape` 改变数据的形状

```python
a = np.arange(12).reshape((3,4))    # 3行4列，0到11
"""
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
"""
```

## 3. `numpy`基础运算

### 3.1 一维数组计算

假设：

```python
import numpy as np
a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
b=np.arange(4)              # array([0, 1, 2, 3])
```

- 矩阵减法

```python
c=a-b  # array([10, 19, 28, 37])
```

- 矩阵加法

```python
c=a+b   # array([10, 21, 32, 43])
```

- 矩阵点乘

```python
c=a*b   # array([  0,  20,  60, 120])
```

- 矩阵元素的乘方

```python
c=b**2  # array([0, 1, 4, 9])
```

- 三角函数

```python
c=10*np.sin(a)  
# array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])
```

- 矩阵元素逻辑判断

```python
print(b<3)   # 返回一个布尔类型的矩阵
# array([ True,  True,  True, False], dtype=bool)
```

### 3.2 多维数组运算

假设：

```python
a=np.array([[1,2],[3,3]])
b=np.arange(4).reshape((2,2))
```

- 矩阵乘法

```python
c_dot = np.dot(a,b)
# [[ 4  7]
#  [ 6 12]]

c_dot_2 = a.dot(b)
# [[ 4  7]
#  [ 6 12]]
```

- 矩阵转置

```Python
print(np.transpose(a))    
print(a.T)
# [[1 3]
#  [2 3]]
```

- 元素之和、最大值、最小值

```python
np.sum(a)   # 9
np.min(a)   # 1
np.max(a)   # 3

np.sum(a,axis=1)  # 按第1维度(二维情况为行)进行相加  [3 6]
np.sum(a,axis=0)  # 按第0维度(二维情况为列)进行相加  [4 5]
```

- 元素最大最小索引

```python
np.argmin(a)   # 0
np.argmax(a)   # 2
```

- 元素均值

```Python
np.mean(a)      # 2.25
np.average(a)     # 2.25
a.mean()   # 2.25
```

- 元素中位数

```
print(np.median(a))       # 2.5
```

### 3.3 其他一些基本操作

- 累加运算

```
print(np.cumsum(a))   # [1 3 6 9]
```

- 累差运算

```
print(np.diff(a))   
#[[1]
# [0]]
```

- 排序

```
import numpy as np
A = np.arange(14,2, -1).reshape((3,4)) 

# array([[14, 13, 12, 11],
#       [10,  9,  8,  7],
#       [ 6,  5,  4,  3]])

print(np.sort(A))    

# array([[11,12,13,14]
#        [ 7, 8, 9,10]
#        [ 3, 4, 5, 6]])
```

- 元素上下限设置

```
print(np.clip(a,2,3))

#[[2 2]
# [3 3]]
```

## 4. numpy索引

### 4.1. 一维索引

```python
import numpy as np
A = np.arange(3,15)

# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
         
print(A[3])    # 6
```

若矩阵为二维呢?

```python
A = np.arange(3,15).reshape((3,4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
         
print(A[2])         
# [11 12 13 14]
```

### 4.2. 二维索引

- 上述二维的情况下，获取单个元素的时候，索引如下：

```
print(A[1][1])      # 8
```

- **切片**获取第二行中间两个元素

```
print(A[1, 1:3])    # [8 9]
```

- for循环时输出为按行输出

```python
for row in A:
    print(row)
"""    
[ 3,  4,  5, 6]
[ 7,  8,  9, 10]
[11, 12, 13, 14]
"""

# 若需按列

for column in A.T:
    print(column)
"""  
[ 3,  7,  11]
[ 4,  8,  12]
[ 5,  9,  13]
[ 6, 10,  14]
"""
```

- 迭代为一维后输出

```python
import numpy as np
A = np.arange(3,15).reshape((3,4))
         
print(A.flatten())   
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

for item in A.flat:
    print(item)
    
# 3
# 4
……
# 14
```

## 5. `numpy array`合并

- `np.vstack()` 上下合并   按第0维合并

```python
import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])
         
print(np.vstack((A,B)))    # vertical stack
"""
[[1,1,1]
 [2,2,2]]
"""
```

- `np.hstack()` 左右合并

```python
D = np.hstack((A,B))       # horizontal stack

print(D)
# [1,1,1,2,2,2]

print(A.shape,D.shape)
# (3,) (6,)
```

- `np.newaxis()`  一维转多维

```
print(A[np.newaxis,:])
# [[1 1 1]]

print(A[np.newaxis,:].shape)
# (1,3)

print(A[:,np.newaxis])
"""
[[1]
[1]
[1]]
"""

print(A[:,np.newaxis].shape)
# (3,1)
```

- `np.concatenate()` 矩阵拼接

```
C = np.concatenate((A,B,B,A),axis=0)

print(C)
"""
array([[1],
       [1],
       [1],
       [2],
       [2],
       [2],
       [2],
       [2],
       [2],
       [1],
       [1],
       [1]])
"""

D = np.concatenate((A,B,B,A),axis=1)

print(D)
"""
array([[1, 2, 2, 1],
       [1, 2, 2, 1],
       [1, 2, 2, 1]])
"""
```

## 6. `numpy array`分割

假设

```python
import numpy as np

A = np.arange(12).reshape((3, 4))
print(A)
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""
```

- 纵向分割

```python
print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""

print(np.hsplit(A, 2)) #等于 print(np.split(A, 2, axis=1))
```

- 横向分割

```python
print(np.split(A, 3, axis=0))

# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]

print(np.vsplit(A, 3))
```

- 错误的分割

```
print(np.split(A, 3, axis=1))  # 只能等量对分

# ValueError: array split does not result in an equal division
```

- 不等量的分割`np.array_split()`

```
print(np.array_split(A, 3, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2],
        [ 6],
        [10]]), array([[ 3],
        [ 7],
        [11]])]
"""
```

## 7.其他汇总

求方阵的特征值特征向量：

```
a,b=numpy.linalg.elg(x) ##方阵x 特征值赋值给a，对应特征向量赋值给b 
```

排序:

```
ndarray.sort(axis=-1, kind='quicksort', order=None)
axis：排序沿着数组的方向，0表示按行，1表示按列
kind：排序的算法，提供了快排、混排、堆排
order：不是指的顺序，以后用的时候再去分析这个
作用效果：对数组a排序，排序后直接改变了a

numpy.sort(a, axis=-1, kind='quicksort', order=None)
a：要排序的数组，其他同1
作用效果：对数组a排序，返回一个排序后的数组（与a相同维度），a不变

numpy.argsort(a, axis=-1, kind='quicksort', order=None)
作用效果：对数组a排序，返回一个排序后索引，a不变

sorted(iterable, cmp=None, key=None, reverse=False)
说明：内置的排序函数，对list，字典等等可以使用
iterable：是可迭代类型;
cmp：用于比较的函数，比较什么由key决定,有默认值，迭代集合中的一项;
key：用列表元素的某个属性和函数进行作为关键字，有默认值，迭代集合中的一项;
reverse：排序规则. reverse = True 或者 reverse = False，默认False（从小到大）。
返回值：是一个经过排序的可迭代类型，与iterable一样;
```



