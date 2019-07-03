---
title: 2019-summer-deep_learning
date: 2019-07-02 16:05:36
categories: Deep Learning
toc: true
marked: true
---

## 2019-SUMMER-深度学习

###### 前言：本篇用作记录小学期的深度学习课程

<!-- toc -->

#### 数据集的拆分

##### 留出法

处理简单

不稳定

##### k-fold Cross Validation K- 折交叉验证 

常见：5折

目的：提高评估结果的稳定性

###### Stratified k-fold 分层抽样策略

把原始数据集划分为多个层：每个层的类别及比例完全相同

##### Hyper parameter 超参

Set before train. 调参。

eg: learning rate.

###### Grid search 网格搜索

离散取值对组合，尝试所有网格上的取值对。（连续参数离散化）

具体步骤：

- 选择取值组合
- K-fold cross validation
- 记录分数，循环

<!-- more -->

#### 分类问题

##### 分类性能度量-准确率

positive - 关注的样本

negative - 其他样本

- True Positive(真正, TP)：将正类预测为正类数. 
- True Negative(真负 , TN)：将负类预测为负类数. 
- False Positive(假正, FP)：将负类预测为正类数 → 误报 (Type I error). 
- False Negative(假负 , FN)：将正类预测为负类数 → 漏报 (Type II error). 

###### 准确率（分类正确的比例）

$$
accuracy = \frac{TP+TN}{P+N}
$$

###### 精确率（分类positive样本中真正正确分类的概率）

$$
precision = \frac{TP}{TP+TN}
$$

###### 召回率（正确从positive样本中识别positive样本的概率）

$$
recall = \frac{TP}{P}
$$

Recall rate - precision: 反比—> **P-R curve**

###### P-R curve的绘制

通过对概率score的划分：来得到positive和negative的样本

- **AUC**
  - 弥补曲线单点值局部性，反映全局性能

###### F-value

$$
F_{\beta}-\text {score}=\frac{\left(1+\beta^{2}\right) * \text { precision}*\text{  recall}}{\left(\beta^{2} * \text { precision}+\text {recall}\right)}
$$

精确率和召回率的调和平均

###### ROC (Receiver Operating Characteristic) 受试者工作特征曲线

- FP (x-axis) - TP (y-axis) curve
- 反应了之间的权衡
- **ROC-AUC** ：AUC越大，分类器性能越好

##### 分类性能可视化

###### 混淆矩阵的可视化(Confusion Matrix)

![image-20190702134858657](/Users/yola/myGit/yolayolayolanda.github.io/source/images/image-20190702134858657.png)

用heapmap直观地展示诶别的混淆情况（错误划分为别的类别的情况）

###### 分类报告(Classification Report)

先是每个类别的分类性能：包括精确率、召回率、F值等。



#### 一致性评价

##### Pearson 皮尔森相关系数

$$
\boldsymbol{\rho}_{X, Y}=\frac{\operatorname{cov}(X, Y)}{\sigma_{X} \sigma_{Y}}=\frac{E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]}{\sigma_{X} \sigma_{Y}}
$$

- 两人之间
- 可连续可离散
- 取值区间是：[-1, 1], 0 - 完全不相关，1 - 完全正相关，-1 - 完全负相关

##### Cohen's Kappa 

$$
\kappa=\frac{p_{o}-p_{e}}{1-p_{e}}=1-\frac{1-p_{o}}{1-p_{e}}
$$

$p_o$ 是pearson相关系数，$p_e$ 是 overall random agreement.
$$
P_e = p_{yes}+p_{no} = p_{A_{yes}}\times p_{B_{yes}}+p_{A_{no}}\times p_{B_{no}}
$$


- 两人之间
- 更适用于离散的分类标准
- 更强壮

##### Fleiss's Kappa

对每列：
$$
P_j = \frac{同列数据累加}{该列数据总数}
$$
对每行：
$$
p_i = \frac{1}{n(n-1)}\left(\sum_{j=1}^k n_{ij}^2 - n\right)
$$
n 是最高分数上限值。

- 多人之间
- 更适用于离散的分类标准
- 更强壮





#### Feature Engineering 特征工程

##### 文档表示模型：BOW模型和VSM模型

###### BOW (bag-of-words model)

仅考虑单词的存在性

###### VSM (Vector space model)

将文档表示为一个向量空间上的向量

**二者被认为是等价的**

##### 停用词

and, the… 没什么太大区分意义的词

##### N-gram模型

按每N个词分成一个gram（逗号隔开）

卷积神经网络做文本分类 $\sim$ 用N-gram做文本分类

##### 文档之间的距离

###### 欧式距离

$$
\mathrm{d}_{12}=\sqrt{\sum_{k=1}^{n}\left(x_{1 k}-x_{2 k}\right)^{2}}
$$

###### 余弦距离

$$
\cos (\vartheta)=\frac{\sum_{k=1}^{n}\left(x_{1 k} \times x_{2 k}\right)}{\sqrt{\sum_{k=1}^{n}\left(x_{1 k}\right)^{2}} \times \sqrt{\sum_{k=1}^{n}\left(x_{2 k}\right)^{2}}}=\frac{x_{1} \cdot x_{2}}{\left\|x_{1}\right\| \times\left\|x_{2}\right\|}
$$

> 归一化后的空间就可以想象成一个超球面，
>
> 欧氏距离就是球面上两点的直线距离，而向量余弦值等价于那两点的球面距离。
>
> 本质是一样，但没有归一化的情况下，余弦距离是计算相似程度，而欧氏距离计算的是相同程度

##### Tf-idf 词条权重计算

$$
w(x) = ts(x)*idf(x)
$$

tf - 词条在本文当中出现的频率

idf - 词条出现文档数目的反比

##### Feature Scalar 特征值的缩放 （无量纲处理）

###### 标准化法

![image-20190702152434982](/Users/yola/myGit/yolayolayolanda.github.io/source/images/image-20190702152434982.png)

###### 区间缩放法

以列（特征）为单位对区间进行缩放到某个特定范围内

![image-20190702152556953](/Users/yola/myGit/yolayolayolanda.github.io/source/images/image-20190702152556953.png)

###### 归一化（Normalizer）

以行（样本）为单位进行归一化：单位向量

![image-20190702152618849](/Users/yola/myGit/yolayolayolanda.github.io/source/images/image-20190702152618849.png)



##### 缺失特征值的弥补计算

常见弥补策略：用同一特征（列）的均值进行替代

##### 创建多项式特征

- 基于线性特征的模型不够理想，改用多项式的特征
- $(x, y) \rightarrow (x,y)^2 = (x^2, y^2 , xy, x, y, 1)$
- 只是特征空间的变化，分类器不做变化

##### Feature Selection 特征选择

###### 方法

1. 方差选择法

   - 方差非常小的特征维度对样本的区分作用很小，可以剔除

2. Pearson相关系数法

3. 基于森林的特征选择

   1. mean decrease impurity

      每个特征的选择减少了多少不纯度？

   2. mean decrease accuracy

      直接度量每个特征值对模型精确度的影响

4. 递归式特征消除

   通过学习器返回的 coef_ 属性 或者 feature_importances_ 属性来获得每个特征的重要程度。 然后，从当前的特征集合中移除最不重要的特征。在特征集合上不断的重复递归这个步骤，直到最终达到所需要的特征数量为止。

##### Feature Reduction 特征降维

1. LDA 线性判别分析法（有监督）

   投影后，使类别内的方差最小，类别之间的方差最大

2. PCA 主成分分析法（无监督）

   奇异值分析，把给定的一组相关特征维度线性变换转换成另外一组不相关的维度，按方差依次递减，选择重要的（方差大）：第一主成分，第二主成分… 

   

#### 回归

- 直接输出值

- 可以用回归来做分类
- 也可以用分类模型来做回归：利用分类器的特点，使其输出连续值即可

##### Linear Regression 线性回归

###### 狭义

一次曲线

###### 广义

用联结函数 link function $g(x)$ 使预测值落在响应区间内。eg: Logistic regression.

###### MSE (Mean Square Error)

$$
\operatorname{MSE}(y, \hat{y})=\frac{1}{n_{\text {samples}}} \sum_{i=0}^{n_{\text { samples }}-1}\left|y_{i}-\hat{y}_{\mathrm{i}}\right|^{2}
$$

###### MAE (Mean Absolute Error)

$$
\operatorname{MAE}(y, \hat{y})=\frac{1}{n_{\text {samples}}} \sum_{i=0}^{n_{\text { samples }}-1}\left|y_{i}-\hat{y_{\mathrm{i}}}\right|
$$

###### Logistic Regression 

损失函数：Log loss (cross-entropy loss)

二类（互信息）
$$
L_{\log }(y, p)=-\log (\operatorname{pr}(y | p))=-(y \log (p)+(1-y) \log (1-p))
$$
多类（交叉熵）
$$
L_{\log }(Y, P)=-\frac{1}{N} \sum_{i=0}^{N-1} \sum_{k=0}^{K-1} y_{i, k} \log p_{i, k}
$$

> ##### 机器学习的要素：
>
> 1. 训练数据
> 2. 特征
> 3. 模型
> 4. 参数：优化方法

###### 求解方法

- 求导求极值解方程：很难解出
- 梯度下降法：利用一阶梯度信息找到局部最优解 - 每一步都让loss function值变小
  - Learning rate：超参，0.01之类
  - Residual 残差
  - scikit-learn的一元线性回归
  - 多元线性回归

###### 另一种评价标准

- R方计算步骤

###### 多项式回归

1. 生成多项式的特征（1: bias）
2. 继续线性分类器

###### N次回归

对训练集的拟合程度 & 对测试集的普遍性  之间的 balance

###### overfitting 过拟合

噪声的学习

###### 损失函数的正则化 —> *防过拟合*

向量范数

矩阵范数

**线性回归的正则化**

加入正则化项：1. 是假设更好地拟合训练数据； 2. 正则化处理：使模型避免过于复杂而过拟合

线性回归正则化后梯度更新略有变化

- Lasso回归（使用L1正则化）
- Ridge回归（使用L2正则化）
- 弹性网（使用L1+L2正则化）

##### Logistic Regression

对线性回归的结果+logistic函数（或称sigmoid函数）将其转化为 (0, 1)的区间上的数值

```
solver参数指定优化方法：
- sag: 随机平均梯度下降，每次迭代仅用一部分样本来计算梯度
- liblinear: 坐标轴下降法 CD法
```

###### 坐标下降法

- Sklearn 中逻辑回归的优化方法
- 非梯度的优化算法，在每次迭代中，在当前点处沿一个坐标方向进行**一维搜索**（某个特征维度之外的特征维度全部固定）以求得一个函数的局部最小值。在整个过程循环使用不同的坐标方向（不同特征维度变量）
- 对非平滑函数会有问题
- 当一个变量的值很大程度地影响另一个变量的最优值时，坐标下降不是一个很好的方法

#### 信息熵

**信息论的熵**：描述信源的不确定性的大小

###### 信息熵：

$$
\mathrm{H}(X)=\sum_{i} \mathrm{P}\left(x_{i}\right) \mathrm{I}\left(x_{i}\right)=-\sum_{i} \mathrm{P}\left(x_{i}\right) \log _{b} \mathrm{P}\left(x_{i}\right)\\
\mathrm{I}(x_i)=-\log P(x_i)
$$

###### Cross Entropy 交叉熵 (Loss function)：

$$
H(P, Q) = -\sum_{x\in X}p(x)\log Q(x) = \sum_{x\in X}p(x)\frac{1}{\log Q(x)}
$$

- 描述两个变量之间的差异性

###### 相对熵（KL距离） ：

$$
D_{KL}(P||Q) = \sum_{x\in X} p(x)\frac{1}{\log Q_X}
$$

- 表示用Q来描述P需要的额外的信息
- 不是严格意义上的距离：不满足对称性

###### JS散度(Jensen-Shannon)

**JS散度**度量了两个概率分布的相似度，基于KL散度的变体，解决了KL散度非对称的问题。一般地，JS散度是对称的，其取值是0到1之间。定义如下：
$$
J S\left(P_{1} \| P_{2}\right)=\frac{1}{2} K L\left(P_{1} \| \frac{P_{1}+P_{2}}{2}\right)+\frac{1}{2} K L\left(P_{2} \| \frac{P_{1}+P_{2}}{2}\right)
$$

###### 联合熵

观察一个多个随机变量的随机系统获得的信息量
$$
H(X, Y)=-\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log p(x, y)=E\left[\log \frac{1}{p(x, y)}\right]
$$

###### 条件熵

已知第二个随机变量随X的值的前提下，随机变量 Y的信息熵还有多少
$$
H(Y|X)=\sum_{x \in \mathcal{X}, y \in \mathcal{Y}} p(x, y) \log \frac{p(x)}{p(x, y)}
$$

###### 互信息

变量间相互依赖性的量度
$$
I(X ; Y)=\sum_{y \in Y} \sum_{x \in X} p(x, y) \log \left(\frac{p(x, y)}{p(x) p(y)}\right)
$$

#### Back Propaganda 反向传播 

- 包括了正向传播和反向传播
- **调整对象**：参数
- 预设条件来结束传播：
  - 目标损失函数的阈值
  - 迭代次数

