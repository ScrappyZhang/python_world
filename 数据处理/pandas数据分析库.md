## 1. pandas数据类型

### 1.1 Series

`Series`的字符串表现形式为：索引在左边，值在右边。默认会自动创建一个0到N-1（N为长度）的整数型索引。

```
import pandas as pd
import numpy as np
s = pd.Series([1,3,6,np.nan,44,1])

print(s)
"""
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
"""
```

### 1.2 DataFrame

`DataFrame`是一个表格型的数据结构，它包含有一组有序的列，每列可以是不同的值类型（数值，字符串，布尔值等）。`DataFrame`既有行索引也有列索引， 它可以被看做由`Series`组成的大字典。

```python
dates = pd.date_range('20180401',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])

print(df)
"""
                   a         b         c         d
2018-04-01 -0.843807  0.922770  0.483373 -2.457732
2018-04-02 -0.090238 -1.564450  0.361317 -0.031557
2018-04-03  2.167338  0.013244 -0.398937 -1.338737
2018-04-04  1.056168  1.413110  1.306358  1.411396
2018-04-05  0.278281  0.759915 -0.186652 -1.590543
2018-04-06  1.480363 -0.438411 -0.001964 -0.214940
"""
```

- 获取b列元素

```Python
print(df['b'])
"""
2018-04-01    0.439262
2018-04-02   -0.202394
2018-04-03   -2.306616
2018-04-04    0.328872
2018-04-05   -0.665601
2018-04-06    1.005113
Freq: D, Name: b, dtype: float64
"""
```

Pandas默认会为DataFrame数据类型没有标签时自动创建行列索引，从0开始

```python
df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

"""
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
"""
```

创建方法2：使用字典来创建

```Python
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20180402'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
                    
print(df2)

"""
     A          B    C  D      E    F
0  1.0 2018-04-02  1.0  3   test  foo
1  1.0 2018-04-02  1.0  3  train  foo
2  1.0 2018-04-02  1.0  3   test  foo
3  1.0 2018-04-02  1.0  3  train  foo
"""

print(df2.dtypes)
"""
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object
"""

print(df2.index)
# Int64Index([0, 1, 2, 3], dtype='int64')


print(df2.columns)
# Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')


print(df2.values)
"""
[[1.0 Timestamp('2018-04-02 00:00:00') 1.0 3 'test' 'foo']
 [1.0 Timestamp('2018-04-02 00:00:00') 1.0 3 'train' 'foo']
 [1.0 Timestamp('2018-04-02 00:00:00') 1.0 3 'test' 'foo']
 [1.0 Timestamp('2018-04-02 00:00:00') 1.0 3 'train' 'foo']]
"""

# 数据描述
df2.describe()
"""
         A    C    D
count  4.0  4.0  4.0
mean   1.0  1.0  3.0
std    0.0  0.0  0.0
min    1.0  1.0  3.0
25%    1.0  1.0  3.0
50%    1.0  1.0  3.0
75%    1.0  1.0  3.0
max    1.0  1.0  3.0
"""

# 转置
print(df2.T)
"""
                     0                    1                    2  \
A                    1                    1                    1
B  2018-04-02 00:00:00  2018-04-02 00:00:00  2018-04-02 00:00:00
C                    1                    1                    1
D                    3                    3                    3
E                 test                train                 test
F                  foo                  foo                  foo

                     3
A                    1
B  2018-04-02 00:00:00
C                    1
D                    3
E                train
F                  foo
"""

# 对index排序
print(df2.sort_index(axis=1, ascending=False))
"""
     F      E  D    C          B    A
0  foo   test  3  1.0 2018-04-02  1.0
1  foo  train  3  1.0 2018-04-02  1.0
2  foo   test  3  1.0 2018-04-02  1.0
3  foo  train  3  1.0 2018-04-02  1.0
"""

# 按某一列排序
"""
     A          B    C  D      E    F
0  1.0 2018-04-02  1.0  3   test  foo
1  1.0 2018-04-02  1.0  3  train  foo
2  1.0 2018-04-02  1.0  3   test  foo
3  1.0 2018-04-02  1.0  3  train  foo
"""
```

## 2. Pandas 选择数据

假设有一个6*4矩阵数据：

```Python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])

"""
       A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```

### 行选择

Pandas进行行选择一般有三种方法：

- 连续多行的选择用类似于python的列表切片

```python
print(df[0:3])
 
"""
            A  B   C   D
2013-01-01  0  1   2   3
2013-01-02  4  5   6   7
2013-01-03  8  9  10  11
"""

print(df['20130102':'20130104'])

"""
A   B   C   D
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
"""
```

- 按照指定的索引选择一行或多行，使用loc[]方法

```python
print(df.loc['20130102'])
"""
A    4
B    5
C    6
D    7
Name: 2013-01-02 00:00:00, dtype: int64
"""

print(df.loc['20130102':'20130104'])
"""
             A   B   C   D
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
"""
```

- 按照指定的位置选择一行或多行，使用iloc[]方法

```python
print(df.iloc[0])
"""
A    0
B    1
C    2
D    3
Name: 2013-01-01 00:00:00, dtype: int64
"""

print(df.iloc[0:3])
"""
            A  B   C   D
2013-01-01  0  1   2   3
2013-01-02  4  5   6   7
2013-01-03  8  9  10  11
"""
```

### 列选择

列选择比较简单，只要直接把列名传递过去即可，如果有多列的数据，要单独指出列名或列的索引号

- 获取某一列

```python
print(df['A'])
print(df.A)

"""
2013-01-01     0
2013-01-02     4
2013-01-03     8
2013-01-04    12
2013-01-05    16
2013-01-06    20
Freq: D, Name: A, dtype: int64
"""
```

- 通过列名获取多列

```python
print(df[['A', 'B', 'C']])
"""
             A   B   C
2013-01-01   0   1   2
2013-01-02   4   5   6
2013-01-03   8   9  10
2013-01-04  12  13  14
2013-01-05  16  17  18
2013-01-06  20  21  22
"""


```

- ix选择特定多个位置的行列

```python
print(df.ix[1:3,['A','C']])
"""
           A   C
2013-01-02  4   6
2013-01-03  8  10
"""

print(df.iloc[[1,3,5],1:3])
"""
             B   C
2013-01-02   5   6
2013-01-04  13  14
2013-01-06  21  22
"""
```



### 逻辑判断筛选

采用判断指令 (Boolean indexing) 进行选择. 我们可以约束某项条件然后选择出当前所有数据

```Python
print(df[df.A>8])

"""
             A   B   C   D
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```

## 3. Pandas设置值

- 根据索引或标签设置：

```python
df.iloc[2,2] = 1111
df.loc['20130101','B'] = 2222

"""
             A     B     C   D
2013-01-01   0  2222     2   3
2013-01-02   4     5     6   7
2013-01-03   8     9  1111  11
2013-01-04  12    13    14  15
2013-01-05  16    17    18  19
2013-01-06  20    21    22  23
"""
```

- 根据条件设置

如果现在的判断条件是这样, 我们想要更改`B`中的数, 而更改的位置是取决于 `A` 的. 对于`A`大于4的位置. 更改`B`在相应位置上的数为0.

```python
df.B[df.A>4] = 0
"""
             A     B     C   D
2013-01-01   0  2222     2   3
2013-01-02   4     5     6   7
2013-01-03   8     0  1111  11
2013-01-04  12     0    14  15
2013-01-05  16     0    18  19
2013-01-06  20     0    22  23 
"""
```

- 整列设置

```python
df['F'] = np.nan
"""
             A     B     C   D   F
2013-01-01   0  2222     2   3 NaN
2013-01-02   4     5     6   7 NaN
2013-01-03   8     0  1111  11 NaN
2013-01-04  12     0    14  15 NaN
2013-01-05  16     0    18  19 NaN
2013-01-06  20     0    22  23 NaN
"""

df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101',periods=6)) 
"""
             A     B     C   D   F  E
2013-01-01   0  2222     2   3 NaN  1
2013-01-02   4     5     6   7 NaN  2
2013-01-03   8     0  1111  11 NaN  3
2013-01-04  12     0    14  15 NaN  4
2013-01-05  16     0    18  19 NaN  5
2013-01-06  20     0    22  23 NaN  6
"""
```

- 正行设置

```python
df.loc['20130101'] = [1,2,3,4]
"""
             A   B   C   D
2013-01-01   1   2   3   4
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```

## 4. Pandas处理丢失数据

假设有如下矩阵数据：

```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
"""
             A     B     C   D
2013-01-01   0   NaN   2.0   3
2013-01-02   4   5.0   NaN   7
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""
```

- `pd.dropna()`：如果想直接去掉有 `NaN` 的行或列, 可以使用 `dropna`

```python
df.dropna(
    axis=0,     # 0: 对行进行操作; 1: 对列进行操作
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    ) 
"""
             A     B     C   D
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""
```

- `pd.fillna()`：如果是将 `NaN` 的值用其他值代替, 比如代替成 `0`

```python
df.fillna(value=0)
"""
             A     B     C   D
2013-01-01   0   0.0   2.0   3
2013-01-02   4   5.0   0.0   7
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""
```

- `pd.isnull()`：判断是否有缺失数据 `NaN`, 为 `True` 表示缺失数据

```python
df.isnull() 
"""
                A      B      C      D
2013-01-01  False   True  False  False
2013-01-02  False  False   True  False
2013-01-03  False  False  False  False
2013-01-04  False  False  False  False
2013-01-05  False  False  False  False
2013-01-06  False  False  False  False
"""

# 检测在数据中是否存在 NaN, 如果存在就返回 True:
np.any(df.isnull()) == True  
# True
```

## 5. Pandas导入导出

- 读取csv

```
import pandas as pd #加载模块

#读取csv
data = pd.read_csv('student.csv')

#打印出data
print(data)
```

- 将资料存取成pickle

```
data.to_pickle('student.pickle')
```

## 6. Pandas 合并 concat

`pandas`处理多组数据的时候往往会要用到数据的合并处理,使用 `concat`是一种基本的合并方式.而且`concat`中有很多参数可以调整,合并成你想要的数据形式。

- axis (合并方向)

`axis=0`是预设值，因此未设定任何参数时，函数默认`axis=0`。

```Python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#concat纵向合并
res = pd.concat([df1, df2, df3], axis=0)

#打印结果
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 0  1.0  1.0  1.0  1.0
# 1  1.0  1.0  1.0  1.0
# 2  1.0  1.0  1.0  1.0
# 0  2.0  2.0  2.0  2.0
# 1  2.0  2.0  2.0  2.0
# 2  2.0  2.0  2.0  2.0
```

- ignore_index (重置 index)

```Python
#承上一个例子，并将index_ignore设定为True
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

#打印结果
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0
# 6  2.0  2.0  2.0  2.0
# 7  2.0  2.0  2.0  2.0
# 8  2.0  2.0  2.0  2.0
```

结果的`index`变0, 1, 2, 3, 4, 5, 6, 7, 8。

- join (合并方式)
  - `join='outer'`为预设值，因此未设定任何参数时，函数默认`join='outer'`。此方式是依照`column`来做纵向合并，有相同的`column`上下合并在一起，其他独自的`column`个自成列，原本没有值的位置皆以`NaN`填充。
  - `join='inner'`为预设值 ，只有相同的`column`合并在一起，其他的会被抛弃。

```python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

#纵向"外"合并df1与df2
res = pd.concat([df1, df2], axis=0, join='outer')

print(res)
#     a    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN
# 2  0.0  0.0  0.0  0.0  NaN
# 3  0.0  0.0  0.0  0.0  NaN
# 2  NaN  1.0  1.0  1.0  1.0
# 3  NaN  1.0  1.0  1.0  1.0
# 4  NaN  1.0  1.0  1.0  1.0
```

```python
#承上一个例子

#纵向"内"合并df1与df2
res = pd.concat([df1, df2], axis=0, join='inner')

#打印结果
print(res)
#     b    c    d
# 1  0.0  0.0  0.0
# 2  0.0  0.0  0.0
# 3  0.0  0.0  0.0
# 2  1.0  1.0  1.0
# 3  1.0  1.0  1.0
# 4  1.0  1.0  1.0

#重置index并打印结果
res = pd.concat([df1, df2], axis=0, join='inner', ignore_index=True)
print(res)
#     b    c    d
# 0  0.0  0.0  0.0
# 1  0.0  0.0  0.0
# 2  0.0  0.0  0.0
# 3  1.0  1.0  1.0
# 4  1.0  1.0  1.0
# 5  1.0  1.0  1.0
```

- join_axes (依照 axes 合并)

```python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

#依照`df1.index`进行横向合并
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])

#打印结果
print(res)
#     a    b    c    d    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0

#移除join_axes，并打印结果
res = pd.concat([df1, df2], axis=1)
print(res)
#     a    b    c    d    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 4  NaN  NaN  NaN  NaN  1.0  1.0  1.0  1.0
```

- append (添加数据)
  - `append`只有纵向合并，没有横向合并

```python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])

#将df2合并到df1的下面，以及重置index，并打印出结果
res = df1.append(df2, ignore_index=True)
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0

#合并多个df，将df2与df3合并至df1的下面，以及重置index，并打印出结果
res = df1.append([df2, df3], ignore_index=True)
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0
# 6  1.0  1.0  1.0  1.0
# 7  1.0  1.0  1.0  1.0
# 8  1.0  1.0  1.0  1.0

#合并series，将s1合并至df1，以及重置index，并打印出结果
res = df1.append(s1, ignore_index=True)
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  2.0  3.0  4.0
```

## 7. Pandas 合并 merge

- 依据一组key合并

```python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
#    A   B key
# 0  A0  B0  K0
# 1  A1  B1  K1
# 2  A2  B2  K2
# 3  A3  B3  K3

print(right)
#    C   D key
# 0  C0  D0  K0
# 1  C1  D1  K1
# 2  C2  D2  K2
# 3  C3  D3  K3

#依据key column合并，并打印出
res = pd.merge(left, right, on='key')

print(res)
#     A   B key   C   D
# 0  A0  B0  K0  C0  D0
# 1  A1  B1  K1  C1  D1
# 2  A2  B2  K2  C2  D2
# 3  A3  B3  K3  C3  D3
```

- 依据两组key合并

合并时有4种方法`how = ['left', 'right', 'outer', 'inner']`，预设值`how='inner'`。

```python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
#    A   B key1 key2
# 0  A0  B0   K0   K0
# 1  A1  B1   K0   K1
# 2  A2  B2   K1   K0
# 3  A3  B3   K2   K1

print(right)
#    C   D key1 key2
# 0  C0  D0   K0   K0
# 1  C1  D1   K1   K0
# 2  C2  D2   K1   K0
# 3  C3  D3   K2   K0

#依据key1与key2 columns进行合并，并打印出四种结果['left', 'right', 'outer', 'inner']
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
print(res)
#    A   B key1 key2   C   D
# 0  A0  B0   K0   K0  C0  D0
# 1  A2  B2   K1   K0  C1  D1
# 2  A2  B2   K1   K0  C2  D2

res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print(res)
#     A    B key1 key2    C    D
# 0   A0   B0   K0   K0   C0   D0
# 1   A1   B1   K0   K1  NaN  NaN
# 2   A2   B2   K1   K0   C1   D1
# 3   A2   B2   K1   K0   C2   D2
# 4   A3   B3   K2   K1  NaN  NaN
# 5  NaN  NaN   K2   K0   C3   D3

res = pd.merge(left, right, on=['key1', 'key2'], how='left')
print(res)
#    A   B key1 key2    C    D
# 0  A0  B0   K0   K0   C0   D0
# 1  A1  B1   K0   K1  NaN  NaN
# 2  A2  B2   K1   K0   C1   D1
# 3  A2  B2   K1   K0   C2   D2
# 4  A3  B3   K2   K1  NaN  NaN

res = pd.merge(left, right, on=['key1', 'key2'], how='right')
print(res)
#     A    B key1 key2   C   D
# 0   A0   B0   K0   K0  C0  D0
# 1   A2   B2   K1   K0  C1  D1
# 2   A2   B2   K1   K0  C2  D2
# 3  NaN  NaN   K2   K0  C3  D3
```

- Indicator
  - `indicator=True`会将合并的记录放在新的一列。

```python
import pandas as pd

#定义资料集并打印出
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})

print(df1)
#   col1 col_left
# 0     0        a
# 1     1        b

print(df2)
#   col1  col_right
# 0     1          2
# 1     2          2
# 2     2          2

# 依据col1进行合并，并启用indicator=True，最后打印出
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print(res)
#   col1 col_left  col_right      _merge
# 0   0.0        a        NaN   left_only
# 1   1.0        b        2.0        both
# 2   2.0      NaN        2.0  right_only
# 3   2.0      NaN        2.0  right_only

# 自定indicator column的名称，并打印出
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print(res)
#   col1 col_left  col_right indicator_column
# 0   0.0        a        NaN        left_only
# 1   1.0        b        2.0             both
# 2   2.0      NaN        2.0       right_only
# 3   2.0      NaN        2.0       right_only
```

- 依据index合并

```python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])

print(left)
#     A   B
# K0  A0  B0
# K1  A1  B1
# K2  A2  B2

print(right)
#     C   D
# K0  C0  D0
# K2  C2  D2
# K3  C3  D3

#依据左右资料集的index进行合并，how='outer',并打印出
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print(res)
#      A    B    C    D
# K0   A0   B0   C0   D0
# K1   A1   B1  NaN  NaN
# K2   A2   B2   C2   D2
# K3  NaN  NaN   C3   D3

#依据左右资料集的index进行合并，how='inner',并打印出
res = pd.merge(left, right, left_index=True, right_index=True, how='inner')
print(res)
#     A   B   C   D
# K0  A0  B0  C0  D0
# K2  A2  B2  C2  D2
```

- 解决overlapping的问题

```Python
import pandas as pd

#定义资料集
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

print(boys)
"""
   age   k
0    1  K0
1    2  K1
2    3  K2
"""

print(girls)
"""
   age   k
0    4  K0
1    5  K0
2    6  K3
"""
res = pd.merge(boys, girls, on='k', how='inner')
print(res)
"""
   age_x   k  age_y
0      1  K0      4
1      1  K0      5
"""

#使用suffixes解决overlapping的问题
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res)
"""
   age_boy   k  age_girl
0        1  K0         4
1        1  K0         5
"""
```

## 8. Pandas plot 出图

```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 随机生成1000个数据
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
 
# 为了方便观看效果, 我们累加这个数据
data = data.cumsum()

# pandas 数据可以直接观看其可视化形式
data.plot()

plt.show()
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
    
data = data.cumsum()
data.plot()
plt.show()
```

以上为Pandas两种最常见的数据绘图示例，除此之外，它还有如下方法：

- bar
- hist    ==df.hist()==
- box  ==df.boxplot()==
- kde
- area
- scatter
- hexbin

例如散点图：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
    
data = data.cumsum()

ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')

data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()

# df直接调用绘图散点图
plt = df.plot(kind='scatter', x='kind', y='parent')
plt.savefig('xxx.png')

# 抖动图：让数据带你发生微小的唯一，使得数据点不能完全重合。
def jitter(series, factor):
    # factor 抖动距离
    z = float(series.max()) - float(series.min())
    a = float(factor)*z/50
    return series.apply(lambda x: x+np.random.uniform(-a, a))

```

# 命令汇总

> df：任意的Pandas DataFrame对象
> s：任意的Pandas Series对象

同时我们需要做如下的引入：

> import pandas as pd
> import numpy as np

## 导入数据

- pd.read_csv(filename)：从CSV文件导入数据
  - `encoding`编码
  - `sep`分隔符
- pd.read_table(filename)：从限定分隔符的文本文件导入数据
- pd.read_excel(filename)：从Excel文件导入数据
- pd.read_sql(query, connection_object)：从SQL表/库导入数据
  - `index_col`规定由那一列数据设置为index
  - 写入数据库
    - pd.io.sql.write_frame(df, table_name, connection_object)  将df写入table_name表
- pd.read_json(json_string)：从JSON格式的字符串导入数据
- pd.read_html(url)：解析URL、字符串或者HTML文件，抽取其中的tables表格
- pd.read_clipboard()：从你的粘贴板获取内容，并传给read_table()
- pd.DataFrame(dict)：从字典对象导入数据，Key是列名，Value是数据

## 导出数据

- df.to_csv(filename)：导出数据到CSV文件
- df.to_excel(filename)：导出数据到Excel文件
- df.to_sql(table_name, connection_object)：导出数据到SQL表
- df.to_json(filename)：以Json格式导出数据到文本文件

## 创建测试对象

- pd.DataFrame(np.random.rand(20,5))：创建20行5列的随机数组成的DataFrame对象
- pd.Series(my_list)：从可迭代对象my_list创建一个Series对象
- df.index = pd.date_range('1900/1/30', periods=df.shape[0])：增加一个日期索引

## 查看、检查数据

- df.dtypes: 查看各行的数据格式
- ==df.head(n)：查看DataFrame对象的前n行==
- df.tail(n)：查看DataFrame对象的最后n行
- df.index : 查看索引
- ==df.columns: 查看列名==
- ==df.shape()：查看行数和列数==
- [df.info()](https://link.zhihu.com/?target=http%3A//df.info%28%29)：查看索引、数据类型和内存信息
- df.values 查看数据值
- ==df.describe()：查看数值型列的汇总统计==，返回的也是一个DataFrame数据
- df.T 转置，即行列转换
- df.sort(columns='C')排序
- ==s.value_counts(dropna=False)：查看Series对象的唯一值和计数==计数统计
- ==s.unique()  # 存在哪些值==
- ==df.apply(pd.Series.value_counts)：查看DataFrame对象中每一列的唯一值和计数==

## 数据选取

- ==df[col]：根据列名，并以Series的形式返回列==
- df[[col1, col2]]：以DataFrame形式返回多列
- df[1:3]切片操作（行）
  - df['index1':'index2'] 使用行标签来指定输出的行
- s.iloc[0]：按位置（像数组一样操作）选取数据
  - df.iloc[0,:]：返回第一行
    - df.iloc[0] 返回第一行  一个series类型数据
  - df.iloc[0,0]：返回第一列的第一个元素
    - ==df.iat[0,0] 提取单个数时效率更高==
  - df.iloc[3:5,0:2]: 返回局部位置数据
  - df.iloc[[1,2,4], [0,2]] 返回指定不连续行列数据
  - df.iloc[:, 1:3] 返回某某几列
- ==s.loc['index_one']：按索引选取行数据==
  - 多列数据：df.loc[:, ['A', 'B']]
  - 局部数据: df.loc['index1':'index2', ['A', 'B']]
  - 指定行列: df.loc[index1, 'A']
    - ==Df.at[index1, 'A']==
- df.ix[1:3] #获取1-3行的数据,该操作叫切片操作,获取行数据
- df.ix[columns_index] #获取列的数据
- ==使用索引筛选数据：==
  - df.query('col_name=="col_value"')
- df.sub(row, axis=1)  每一行与row做减法

## 数据清理

- ==df.columns = ['a','b','c']：重命名列名==

- pd.isnull()：检查DataFrame对象中的空值，并返回一个Boolean数组

- pd.notnull()：检查DataFrame对象中的非空值，并返回一个Boolean数组

- ==df.dropna()：删除所有包含空值的行==

  - df.dropna(axis=0)

- ==df.dropna(axis=1)：删除所有包含空值的列==

- ==df.dropna(axis=1,thresh=n)：删除所有小于n个非空值的行==

- df[columns].drop_duplicates() #剔除重复行数据

- ==df.fillna(x)：用x替换DataFrame对象中所有的空值==

  - **0替代缺失值**  df.fillna(0)
  - **一个字符串替代缺失值** df.fillna('字符串')
  - **用前一个数据替代缺失值** df.fillna(method='pad')
  - **用前一个数据替代缺失值** df.fillna(method='bfill')
  - **每列限制替代NaN的数目**（例如限制每列只能替代一个NaN） df.fillna(limit=1)
  - **使用描述性统计量来代替缺失值**，如平均值 df.fillna(df.mean())
  - **指定列缺失值处理** df.fillna(df.mean()['one':'two'])

- 插值法填补缺失值

  - df.interpolate()  默认直线形式插值，即前后两个数的平均值
  - 基于索引值插值
    - 索引为数字df.interpolate(method='values')
    - 索引为time   df.interpolate(method='time')

- ==s.astype(float)：将Series中的数据类型更改为float类型==

- ==s.replace(1,'one')：用‘one’代替所有等于1的值==

  - s.replace([1,3],['one','three'])：用'one'代替1，用'three'代替3
  - s.replace({1:11,2:12}) 使用字典映射：将1替换为11，将2替换为12
  - df['a'].replace()
  - `df[['a']['b']].replace(2,10)`多列进行相同的替换将2替换为10
  - `df.replace({‘a’: 0, 'b':5}, np.nan)`多列替换为不同值 ,例中将缺失值对应的替换为0和5
  - 插值替换`df['a'].replace([1,2,3], method='pad')`

- df.rename(columns=lambda x: x + 1)：批量更改列名

- df.rename(columns={'old_name': 'new_ name'})：选择性更改列名

- df.set_index('column_one')：更改索引列

- df.rename(index=lambda x: x + 1)：批量重命名索引

- #### ==数据标准化==

  - ```
    import pandas as pd
    import numpy as np

    index = pd.date_range('1/1/2018', periods=100)
    ts = pd.Series(np.random.normal(0.5,2,100), index)

    ts.head()
    """
    2018-01-01   -0.305553
    2018-01-02    0.712781
    2018-01-03    4.885561
    2018-01-04   -0.433001
    2018-01-05    0.403765
    Freq: D, dtype: float64
    """

    key = lambda x: x.month  # 分组
    zscore = lambda x: (x-x.mean())/x.std() # 标准化
    trandformed = ts.groupby(key).transform(zscore)  # 数据变换为标准化
    # 查看平均值与方差
    trandformed.groupby(key).mean()
    trandformed.groupby(key).std()
    ```

    ​

## 数据处理：Filter、Sort和GroupBy

- #### Filter

  - df[df[col] > 0.5]：选择col列的值大于0.5的行

    - df[df.col_name > 0]: 选择col_name大于0的行

  - 多条件筛选

    - 与  df[(df.D>0) & (df.C<0)]
    - 或 df[(df.D>0) | (df.C<0)]

  - 筛选并返回指定列

    - `df[['A', 'B']][df.D>0]&(df.C<0)]`以D、C列的筛选条件获取相应的AB列数据

  - 布尔索引

    - ```
      index = [df.D>0]&(df.C<0
      df[index]
      ```

  - `isin`

    - ```
      alist = [1,2,3,4]
      df['D'].isin(alist)  # D列数据是否在alist中
      ```

      ​

- df.sort_values(col1)：按照列col1排序数据，默认升序排列

- df.sort_values(by='B')  按列B排序数据

- ==df.sort_values(col2, ascending=False)：按照列col1降序排列数据==

- df.sort_values([col1,col2], ascending=[True,False])：先按列col1升序排列，后按col2降序排列数据

- df.sort_index(axis=1, ascending=False) 按列名降序排列各列

- #### 分组groupby

  - ==df.groupby(col)：返回一个按列col进行分组的Groupby对象==

    - grouped_df.first()返回每一组的第一行数据
    - grouped_df.last()返回每一组的第一行数据

  - df.groupby([col1,col2])：返回一个按多列进行分组的Groupby对象

    - 也可以由一个函数来对列名进行分组

      - ```
        def get_type(col_name):
        	if col_name.lower() in 'abcd':
        		return 'first'
        	else:
        		return 'sencond'
        grouped_df = df.groupy(get_type, axis=1)
        ```

  - df.groupby(level='index_col_name') 按指定索引来分组

  - Df.groupby(level=['index_col_name1', 'index_col_name2'])按两个索引分类，迭代单个值为元组

  - df.groupby(col1)[col2].mean()：返回按列col1进行分组后，列col2的均值

  - ==df.groupby(col1).agg(np.mean)：返回按列col1分组的所有列的均值==

    - df.groupby(col1).agg([np.mean, np.sum, np.std])

      - 指定返回显示的列名
        - df.groupby(col1).agg({‘agg1’: np.mean, 'agg2': np.sum,  'agg3': np.std})

    - agg自定义匿名函数

      - df.groupby(col1).agg(lambda x: np.mean(abs(x)))

    - ```
      grouped = df.groupby(col1).aggregate(np.mean)

      # 将分组聚类后的新dataframe索引转换为列变量
      grouped.reset_index()

      # 使用as_index参数也可以达到上面的效果
      df.groupby(col1, as_index=False).aggregate(np.mean)
      ```

  - ==分组数据量grouped.size()==

- df.stack()   把列（column）放置到索引位置

- df.unstack()

  ​

- ==df.pivot_table(index=col1, values=[col2,col3], aggfunc=max)：创建一个按列col1进行分组，并计算col2和col3的最大值的数据透视表==

- data.apply(np.mean)：对DataFrame中的每一列应用函数np.mean

- ==data.apply(np.max,axis=1)：对DataFrame中的每一行应用函数np.max==

- **使用DataFrame模糊筛选数据(类似SQL中的LIKE):**

```
df_obj[df_obj['套餐'].str.contains(r'.*?语音CDMA.*')] #使用正则表达式进行模糊匹配,*匹配0或无限次,?匹配0或1次
```

- **使用DataFrame筛选数据(类似SQL中的WHERE):**

```
alist = ['023-18996609823']
df_obj['用户号码'].isin(alist) #将要过滤的数据放入字典中,使用isin对数据进行筛选,返回行索引以及每行筛选的结果,若匹配则返回ture
df_obj[df_obj['用户号码'].isin(alist)] #获取匹配结果为ture的行
```



## 数据合并

- df1.append(df2)：将df2中的==行==添加到df1的尾部

- ==df.concat([df1, df2],axis=1)：将df2中的列添加到df1的尾部==

  - axis：需要合并连接的轴，0是行，1是列；
  - join：连接的参数，inner或outer；
  - ignore=True表示重建索引。

- df1.join(df2,on=col1,how='inner')：对df1的列和df2的列执行SQL形式的join

- ```
  merge(mxj_obj2, mxj_obj1 ,on='用户标识',how='inner')# mxj_obj1和mxj_obj2将用户标识当成重叠列的键合并两个数据集,inner表示取两个数据集的交集.
  ```

## 数据统计

- ==在描述性统计中，Nan都是作为0进行运算==
- df.describe()：查看数据值列的汇总统计
- df.mean()：返回所有列的均值
- df.corr()：返回列与列之间的相关系数
- df.count()：返回每一列中的非空值的个数
- df.max()：返回每一列的最大值
- df.min()：返回每一列的最小值
- df.median()：返回每一列的中位数


- df.std()：返回每一列的标准差

```
count                      非 NA 值的数量
describe                  针对 Series 或 DF 的列计算汇总统计
min , max                最小值和最大值
argmin , argmax      最小值和最大值的索引位置（整数）
idxmin , idxmax       最小值和最大值的索引值
quantile                    样本分位数（0 到 1）
sum                          求和
mean                        均值
median                     中位数
mad                          根据均值计算平均绝对离差
var                             方差
std                             标准差
skew                          样本值的偏度（三阶矩）
kurt                           样本值的峰度（四阶矩）
cumsum                     样本值的累计和
cummin , cummax      样本值的累计最大值和累计最小值

cumprod                     样本值的累计积

diff                              计算一阶差分（对时间序列很有用）

pct_change                  计算百分数变化

```



  

- ==s.value_counts(dropna=False)：查看Series对象的唯一值和计数==计数统计

  - 柱形图

    - ```
      # 计数统计
      counts = df['A'].value_counts()
      # 绘制柱形图
      plt=counts.plot(kind='bar').get_figure()
      plt.savefig('counts.png')
      ```

      ​

- 对数据应用函数

```
对数据应用函数
a.apply(lambda x:x.max()-x.min())
表示返回所有列中最大值-最小值的差
```

- 通过Pandas读取大文件

```python
import pandas as pd

f = open('test_data.csv')
# read_csv()函数的iterator参数等于True时，表示返回一个TextParser以便逐块读取文件
reader = pd.read_csv(f, sep=',', iterator=True)
loop = True
# 每次读取的行数 chunkSize
chunkSize = 100000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
# 合并分次读取的数据
df = pd.concat(chunks, ignore_index=True)
print(df)
```

- 判断Dataframe是否为空

```python
if df.empty:
    print('DataFrame is empty!')
```

- 合并时直接删除重复列名的多余列

```
cols_to_use = df2.columns - df.columns
dfNew = merge(df, df2[cols_to_use], on='Protein_ID', how='left')
```

- 整列操作

  - 增加列

    - ==df['C'] = Series对象  增加C列==
    - ==df.insert(1,'e', df['a'])  将df['a']添加到列索引为1的位置，列名为e==

  - 移动列

    - ```
      将b列移动至第0列
      b = df.pop('b')
      df.insert(0, 'b', b)
      ```

      ​

  - 删除含有空值的行列

    - df.dropna()：删除所有包含空值的行
    - df.dropna(axis=1)：删除所有包含空值的列

  - 删除列

    - 永久删除一列： del df['C']
    - 返回一个新DataFrame  df.drop(['C'], axis=1)

### 字符串操作

```python
import pandas as pd
import numpy as np
s = pd.Series(list('ABCDEF'))

"""
0    A
1    B
2    C
3    D
4    E
5    F
dtype: object
"""
```

- 数据转小写`s.str.lower()`

```python
# 不改变原始数据
"""
0    a
1    b
2    c
3    d
4    e
5    f
dtype: object
"""
```

- 数据转大写`s.str.upper()`
- 数据字符串长度`s.str.len()`
- 数据字符串切割`s.str.split(sep)`
- 获取某个元素
  - s.str.split('sep').str.get(1)
  - s.str.split('sep').str[1]
- 替换字符串
  - s.str.replace(正则表达式，替换的新字符串, case=False)
- 字符串内提取数据
  - 例，extract方法提取数字
    - s.str.extract('(\d)')  括号内为正则表达式语法，可同时提取多个数据，即括号内的表达式匹配字符
- 包含数据序列检测
  - s.str.contains(pattern)
  - s.str.contains(pattern, na=False) na代表遇到NaN数据时匹配成True还是False
- 严格匹配字符串
  - s.str.match(pattern, as_indexer=False)
- 检查字符串开始字符
  - s.str.startswith('a', na=False)
- 检查字符串结束字符
  - s.str.endswith('a', na=False)



- 样本随机抽取
  - df.sample(frac=0.5， random_state=20)  0.5代表抽取的比例
- 编码映射

```
label_mapping = {'汽车':1, '财经':2, '科技':3, '健康':4, '体育':5, '教育':6, '文化':7, '军事':8, '娱乐':9, '时尚':10}
df_train['label'] = df_train['label'].map(label_mapping)
```

