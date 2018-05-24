# XGBoost

参考文献：

1. http://xgboost.readthedocs.io/en/latest/model.html
2. https://blog.csdn.net/sb19931201/article/details/52557382
3. https://blog.csdn.net/u010657489/article/details/51952785

## 1. XGBoost原理



一般，机器学习模型的目标函数由两部分组成——损失误差函数+正则项：
$$
obj(\theta )=L(\theta )+\Omega (\theta )
$$

> 其中，L为训练损失函数、$\Omega$为正则项。
>
> 例如，常见的L函数有：MSE：$L(\theta )=\sum _{ i }^{  }{ (y_i-\hat { y_i } )^2 } $
>
> 逻辑回归的L函数：$L(\theta )=\sum _{ i }^{  }{ [y_i\ln { (1+e^{-\hat { y_i } } )} + (1-y_i)\ln { 1 + e^{\hat { y_i }} }]} $
>
> 正则项常有L0、L1、L2，主要是为了解决模型过于复杂导致过拟合问题。

首先我们先了解下树的集成效果：

![image-20180524135610522](image/树集成.png)

以上为两个单独的数所预测的结果，若将两棵树结合起来，则预测结果分别为2.9、1.9 。即一种简单的树集成效果为多棵树预测结果进行求和平均，如下式所示：
$$
\hat { y_{ i } } =\sum _{ k=1 }^{ K }{ f_{ k }(x_{ i }),f_{ k }\in  \digamma  } 
$$

> 其中，K为树的数量， $f$为预测函数

那么此时，目标函数应该改写为：
$$
obj(\theta )=\sum _{ i }^{ n }{ l(y_{ i },\hat { y_{ i } } ) } +\sum _{ k=1 }^{ K }{ \Omega(f_k)  } 
$$

> 其中，可以看出，树集成的目标函数主要分两大部分：损失函数、所有树的正则项

### 损失函数

![image-20180524140755971](image/boosting模型与rf模型对比.png)

从上图可以看出，对Boosting模型来说，其预测为逐级累加，即如下式:
$$
 \hat { y_{ i } } ^{ (0) }=0\\ \hat { y_{ i } } ^{ (1) }=f_{ 1 }(x_{ i })={ y_{ i } }^{ (0) }+f_{ 1 }(x_{ i })\\ \hat { y_{ i } } ^{ (2) }=f_{ 1 }(x_{ i })+f_{ 2 }(x_{ i })={ y_{ i } }^{ (1) }+f_{ 2 }(x_{ i })\\ \qquad ...\\ \hat { y_{ i } } ^{ (t) }=\sum _{ k=1 }^{ t }{ f_{ k }(x_{ i }) } ={ y_{ i } }^{ (t-11) }+f_{ t }(x_{ i })
$$
那么将预测值$\hat{y_i}^{(t)}$代入目标函数，可得：
$$
obj^{ (t) }=\sum _{ i=1 }^{ n }{ l(y_{ i },\hat { y_{ i } } ^{ (t) }) } +\sum _{ i=1 }^{ t }{ \Omega (f_{ i }) } \\ \qquad =\sum _{ i=1 }^{ n }{ l(y_{ i },\hat { y_{ i } } ^{ (t-1) }+f_t(x_i)) } +\Omega (f_{ t }) +constant
$$
假设我们采用MSE作为损失函数，则代入MSE对应的损失函数，目标函数可以转换为：
$$
obj^{ (t) }=\sum _{ i=1 }^{ n }{ (y_{ i }-(\hat { y_i }^{(t-1)} +f_t(x_i)))^{ 2 } } +\sum _{ i=1 }^{ t }{ \Omega (f_{ i }) } \\ \qquad =\sum _{ i=1 }^{ n }{ [2(\hat { y_i }^{(t-1)}-y_i )f_t(x_i)+f_t(x_i)^2]} +\Omega (f_{ t })+constant
$$
==注意：==此处为XGBoost与普通GBDT的不同点一：

==对6式中做泰勒展开，并保留二阶子式，（普通GBDT仅含有一阶子式）==

可以得到：
$$
obj^{ (t) }=\sum _{ i=1 }^{ n }{ [l(\hat { y_{ i } } ^{ (t-1) },y_{ i })  +g_if_{ t }(x_{ i })+\frac { 1 }{ 2 } h_if_{ t }(x_{ i })^{ 2 }] } +\Omega (f_{ t })+constant
$$

> 其中：
> $$
> g_{ i }=\partial _{ \hat { y_{ i } } ^{ (t-1) } }l(\hat { y_{ i } } ^{ (t-1) },y_{ i })\\ h_{ i }=\partial^2 _{ \hat { y_{ i } } ^{ (t-1) } }l(\hat { y_{ i } } ^{ (t-1) },y_{ i })
> $$
> 

### 正则项

接下来我们考虑正则项$\Omega$

我们定义3式中对应的$f_k$为
$$
f_t(x)=w_{q(x)},w\in R^T, q:R^d\rightarrow {1,2,...,T}
$$

> 其中，$w$为每棵树所有叶子对应的得分向量，$q$为与数据点与叶子对应的函数，T为叶子数目

==注意：==此处为XGBoost与普通GBDT的不同点二：

据此，在XGBoost中，定义正则项$\Omega$为模型复杂度，如下:
$$
\Omega (f)=\gamma T+\frac { 1 }{ 2 } \lambda \sum _{ j=1 }^{ T }{ w_j^2 } 
$$

> 也就是说，正则项包含两项：叶子数目，L2正则项。

那么我们可以得到第t棵树的目标函数为：
$$
obj^{ (t) }\approx \sum _{ i=1 }^{ n }{ [g_{ i }w_{ q(x_{ i }) }+\frac { 1 }{ 2 } h_{ i }w^{ 2 }_{ q(x_{ i }) }] } +\gamma T+\frac { 1 }{ 2 } \lambda \sum _{ j=1 }^{ T }{ w_{ j }^{ 2 } } \\ \qquad =\sum _{ j=1 }^{ T }{ [(\sum _{ i\in I_{ j } }{ g_{ i } } )w_{ j }+\frac { 1 }{ 2 } (\sum _{ i\in I_{ j } }{ h_{ i } } +\lambda )w^2_{ j }] } +\gamma T
$$

> 其中，$I_{ j }=\left\{ i|q(x_i)=j \right\} $代表第j个叶子对应的数据点索引集合

此时，我们令：
$$
G_{ j }=\sum _{ i\in I_{ j } }{ g_{ i } } \\ H_j=\sum_{i\in I_j } { h_i }
$$
可以得到目标函数为：
$$
obj^{ (t) }=\sum _{ j=1 }^{ T }{ [G_{ j }w_{ j }+\frac { 1 }{ 2 } (H_j+\lambda )w^{ 2 }_{ j }] } +\gamma T
$$

> 很容易可以计算出$w_j$的最优解为:
> $$
> w^*_j=-\frac { G_j }{ H_j +\lambda  }
> $$
>

从而得到对应的目标函数为：
$$
obj^{ * }_{ j }=-\frac { 1 }{ 2 } \sum _{ j=1 }^{ T }{ \frac { G_{ j }^{ 2 } }{ H_{ j }^{ 2 }+\lambda  }  } +\gamma T
$$

### 树分裂（树结构）打分算法：

最常见的就是贪心法，对某个叶子进行分裂，若分裂后的目标函数比分裂前的目标函数更优(信息增益大于0)，则进行树分裂。

==XGBoost中的增益计算如下：==
$$
Gain=\frac { 1 }{ 2 } [\frac { G_L^2 }{ H_L+\lambda  } +\frac { G_R^2 }{ H_R+\lambda  } -\frac { (G_L+G_R)^2 }{ H_L+H_R+\lambda  } ]-\gamma 
$$

> 即增益为0.5*（新左子树分数+右子树分数-不分裂叶子分数）-加入新叶子节点引入的复杂度

==此外，作者针对该算法设计对特征进行了排序，分位点划分等。==

![image-20180524152106097](image/分位点划分.png)

根据特征划分有无数可能的树结构，因此采用近似算法（特征分位点，候选分割点）

![image-20180524152223588](image/分为点划分2.png)

### 总结

![image-20180524152307205](image/xgboost特点.png)

- 每轮增加一棵树
- 传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。 **—可以通过booster [default=gbtree]设置参数:gbtree: tree-based models/gblinear: linear models**
- 损失函数引入二阶导数，传统GBDT在优化时只用到一阶导数信息
- 损失函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性  **—正则化包括了两个部分，都是为了防止过拟合，剪枝是都有的，叶子结点输出L2平滑是新增的。**
- shrinkage and column subsampling —**还是为了防止过拟合**
- 树分裂（树打分）为贪心算法寻找切分点
  - exact greedy algorithm—**贪心算法获取最优切分点** 
  - approximate algorithm— **近似算法，提出了候选分割点概念，先通过直方图算法获得候选分割点的分布情况，然后根据候选分割点将连续的特征信息映射到不同的buckets中，并统计汇总信息。**
  - Weighted Quantile Sketch—**分布式加权直方图算法，**
- 添加新树时引入缩减因子避免过拟合
- 对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。 **—稀疏感知算法**

### XGBoost包的特点

- **Built-in Cross-Validation（内置交叉验证)**

- **continue on Existing Model（接着已有模型学习）**

- **High Flexibility（高灵活性）**

- 并行化处理 **—系统设计模块,块结构设计等**

  - > xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。

## 2. XGBoost参数

### 通用参数

这些参数用来控制XGBoost的宏观功能。

#### 1、booster[默认gbtree]

- 选择每次迭代的模型，有两种选择： 
  gbtree：基于树的模型 
  gbliner：线性模型

#### 2、silent[默认0]

- 当这个参数值为1时，静默模式开启，不会输出任何信息。
- 一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。

#### 3、nthread[默认值为最大可能的线程数]

- 这个参数用来进行多线程控制，应当输入系统的核数。
- 如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它。 
  还有两个参数，XGBoost会自动设置，目前你不用管它。接下来咱们一起看booster参数。

### booster参数

尽管有两种booster可供选择，我这里只介绍**tree booster**，因为它的表现远远胜过**linear booster**，所以linear booster很少用到。

#### 1、eta[默认0.3]

- 和GBM中的 learning rate 参数类似。
- 通过减少每一步的权重，可以提高模型的鲁棒性。
- 典型值为0.01-0.2。

#### 2、min_child_weight[默认1]

- 决定最小叶子节点样本权重和。
- 和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小*样本权重的和*，而GBM参数是最小*样本总数*。
- 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。
- 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。

#### 3、max_depth[默认6]

- 和GBM中的参数相同，这个值为树的最大深度。
- 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。
- 需要使用CV函数来进行调优。
- 典型值：3-10

#### 4、max_leaf_nodes

- 树上最大的节点或叶子的数量。
- 可以替代max_depth的作用。因为如果生成的是二叉树，一个深度为n的树最多生成$n^2$个叶子。
- 如果定义了这个参数，GBM会忽略max_depth参数。

#### 5、gamma[默认0]

- 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
- 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。

#### 6、max_delta_step[默认0]

- 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。
- 通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。
- 这个参数一般用不到，但是你可以挖掘出来它更多的用处。

#### 7、subsample[默认1]

- 和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。
- 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
- 典型值：0.5-1

#### 8、colsample_bytree[默认1]

- 和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
- 典型值：0.5-1

#### 9、colsample_bylevel[默认1]

- 用来控制树的每一级的每一次分裂，对列数的采样的占比。
- 我个人一般不太用这个参数，因为subsample参数和colsample_bytree参数可以起到相同的作用。但是如果感兴趣，可以挖掘这个参数更多的用处。

#### 10、lambda[默认1]

- 权重的L2正则化项。(和Ridge regression类似)。
- 这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。

#### 11、alpha[默认1]

- 权重的L1正则化项。(和Lasso regression类似)。
- 可以应用在很高维度的情况下，使得算法的速度更快。

#### 12、scale_pos_weight[默认1]

- 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。

### 学习目标参数

这个参数用来控制理想的优化目标和每一步结果的度量方法。

#### 1、objective[默认reg:linear]

这个参数定义需要被最小化的损失函数。最常用的值有： 

- binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
- multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。 
  - 在这种情况下，你还需要多设一个参数：num_class(类别数目)。
- multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。

#### 2、eval_metric[默认值取决于objective参数的取值]

对于有效数据的度量方法。==对于回归问题，默认值是rmse，对于分类问题，默认值是error。==
典型值有： 

- rmse 均方根误差
- mae 平均绝对误差
- logloss 负对数似然函数值
- error 二分类错误率(阈值为0.5)
- merror 多分类错误率
- mlogloss 多分类logloss损失函数
- auc 曲线下面积

#### 3、seed(默认0)

随机数的种子，设置它可以复现随机数据的结果，也可以用于调整参数

### XGBoost调参步骤

第一步：确定学习速率和tree_based 参数调优的估计器数目

第二步： max_depth 和 min_weight 参数调优

第三步：gamma参数调优

第四步：调整subsample 和 colsample_bytree 参数

第五步：正则化参数调优。

第六步：降低学习速率，增加树数量

循环罔替，直到达到要求。

## 3. XGBoost使用示例

```python
import xgboost as xgb

# 此处省去数据预处理部分
# 1. 转换数据结构
train = xgb.DMatrix(train_xy, label=label_train)
test = xgb.DMatrix(offline_test, label=test_ki['Ki'])
watchlist = [(test,'eval'), (train,'train')]
# 2. 参数
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'gamma': 0.1,
    'min_child_weight': 6,
    'max_depth': 10,  # 默认为6
    'max_leaf_nodes': 80,
    'lambda': 8,
    'subsample': 0.7,
    'eta': 0.1, # 学习率
    'seed': 2000,
    'nthread': 3,
}

# 3. 训练
num_round = 1500
gbm = xgb.train(params,
                train,
                num_boost_round=num_round,
                evals=watchlist,
                early_stopping_rounds=100,
                verbose_eval=50,
                )
                
# 4. 预测
testt_feat = xgb.DMatrix(testt_feat)
preds_offline = gbm.predict(test)
offline_mse =  mean_squared_error(label_test, preds_offline)
print( '线下mse ：', mean_squared_error(label_test, preds_offline))
preds_sub = gbm.predict(testt_feat)
```



## 4. XGBoost sklearn接口示例

```
from xgboost.sklearn import XGBClassifier, XGBRegressor

gbm = xgb.XGBRegressor(
        nthread=3,  # 进程数
        max_depth=max_depth,  # 最大深度
        n_estimators=n_estimators,  # 树的数量
        learning_rate=learning_rate,  # 学习率
        subsample=subsample,  # 采样率
        min_child_weight=min_child_weight,  # 孩子数
        max_delta_step=10,  # 10步不降则停止
        eval_metric='rmse',
        objective="reg:linear",
        reg_lambda=reg_lambda
    )

gbm = xgb.XGBClassifier(
        nthread=4,  # 进程数
        max_depth=max_depth,  # 最大深度
        n_estimators=n_estimators,  # 树的数量
        learning_rate=learning_rate,  # 学习率
        subsample=subsample,  # 采样率
        min_child_weight=min_child_weight,  # 孩子数
        max_delta_step=10,  # 10步不降则停止
        num_class=3,
        eval_metric='auc',
        objective="multi:softprob"
    )
```

