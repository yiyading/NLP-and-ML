# 一、逻辑回归
在分类问题中，因为结果的有限性，使用线性回归无法很好的进行拟合，因此要使用一种叫做逻辑回归的算法。

在分类问题中，我们尝试预测结果属于某一个类（例如正确或错误）。例如：
> 判断垃圾邮件<br>
> <br>
> 判断金融交易是否涉嫌欺诈<br>
> <br>
> 肿瘤问题

## 1.二元分类
假设使用线性回归算法解决一个分类问题，每个训练实例的y取值为0或1，但是使用线性回归算法得出的结果可能大于1或者小于零，这种算法无法很好拟合输入-输出，因此我们要使用逻辑回归算法
![逻辑回归算法1](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%951.jpg)
> 逻辑回归算法是分类算法，名字里的回归具有迷惑性，这种算法适合标签y有几个固定取值的情况<br>

**总结**：二元分类本质上就是分类问题中，输出结果只可能有两种情况的分类问题，对于这种问题，线性回归无法很好的拟合

## 2.假说表示
对于乳腺癌分类问题，我们可以使用线性回归求出一条适合数据的直线，因为线性回归模型只能预测连续的值，对于分类问题，我们预测：
> h>0.5, y=1<br>
> h<0.5, y=0<br>

但假如有一个非常大size的肿瘤，将其作为实例加入训练集，就能获得一条新的直线
![逻辑回归算法2](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%952.png)

在上图中右边这条直线上，再使用0.5作为判断肿瘤是否恶性的阈值已经不合适

我们引入一个新的模型：逻辑回归。该模型的输出变量的范围始终再[0, 1]之间。

模型假设：
![逻辑回归算法3](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%953.png)
> X代表特征向量<br>
> h -> g ，g代表逻辑函数，是一个常用的逻辑函数Sigmoid<br>
> x = θ.T * X<br>
![逻辑回归算法4](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%954.png)

python实现：
```py
import numpy as np
def sigmoid(z):
	return 1/(1 + np.exp(-z))
```

**重点解释**：h的作用是，对于给定的输入变量，根据选择的参数计算出输出变量=1的概率，即hθ(x) = P(y=1 | x;θ)

若hθ(x) = 0.7，则表示有70%几率y=1，30%几率y=0

## 3.决策边界
**决策边界本质上就是分类线，线的两侧是不同类型的结果**

如下图，在逻辑回归种，经过sigmoid，输出函数被限定在[0,1]

![逻辑回归算法5](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%955.png)

我们假设一个模型，如果想要预测值y=1，则需要z>0

![逻辑回归算法6](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%956.png)

> 假设θ=[-3, 1, 1]，下图展示了z>0 和 z<0分别对应的输出y=1，y=0<br>

![逻辑回归算法7](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%957.png)

**总结**：决策边界本质上就是画出使 θ.T\*X = 0这条直线（也可能使曲线），然后计算出y=1和y=0的概率。

## 4.代价函数
在上边的讨论中，我们已经知道监督学习问题中的逻辑回归模型的拟合与线性回归模型的不同。<br>
![逻辑回归算法8](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%958.png)

如果我们将上图中的hθ(x)带入我们在**线性回归模型中定义的误差平方和代价函数**，则我们得到的损失函数将是一个非凸函数

![逻辑回归算法9](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%959.png)
> 非凸函数有很多局部最小值，影响梯度下降寻找全局最小值

重新定义代价函数

![逻辑回归算法10](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9510.png)
> 这种构建Cost函数的特点是当y=1时，hθ(x)不为1时误差随着hθ(x)变小而变大；当y=0时则相反

上边这种构造方法是将Cost分为两种情况来分别表述出来，我们将其简化，并代入代价函数：

![逻辑回归算法11](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9511.png)

python代码实现代价函数:
```py
import numpy as np
def cost(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X * theta.Y)))
	second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
	return np.sum(first - second) / (len(X))
```
得到代价函数后，用梯度下降算法求min(j(θ))，和线性回归中的梯度下降求法相似
![逻辑回归算法12](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9512.png)
> 这里的hθ(x)是sigmoid函数，与线性回归中的不同
> <br><br>
> 特征缩放很有必要

**总结**：逻辑回归中的代价函数和线性回归中的代价函数差别在于求hθ(x)的过程中多了一步sigmoid的转化，目的是**更快**计算全局最小值。
> 除了梯度下降算法，还有一些其他的算法可用来寻找最小的代价函数<br>
> <br>
> 共轭梯度<br>
> 局部优化法(BFGS)<br>
> 有限内存局部优化发(L-BFGS)<br>

## 5.简化代价函数和梯度下降
逻辑回归和线性回归的更新规则是相同的，不同的是其中关于hθ(x)的定义发生了改变。
![逻辑回归算法13](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9513.png)

当使用梯度下降实现逻辑回归时，我们有不同的参数θ，θ1 θ2 θ3直到θn，我们需要用梯度下降逐个更新或者向量化同时更新。

我们之前在谈线性回归时讲到的特征缩放，我们看到了特征缩放是如何提高梯度下降的收敛速度的，这个特征缩放的方法，也适用于逻辑回归。如果你的特征范围差距很大的话那么应用特征缩放的方法 同样也可以让逻辑回归中梯度下降收敛更快 。

## 6.更高级的优化
梯度下降的本质是计算损失函数J的最小值，在其最小化的过程中需要计算两个东西：
> J(θ)<br>
> <br>
> J(θ)的各个偏导数项<br>

![逻辑回归算法14](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9514.png)
如果完成了实现这两件事的代码，那么梯度下降所作的就是反复执行这些更新

梯度下降并不是唯一的算法，有一些其他更高级的算法能让我们计算J(θ)和J(θ)的各个偏导数项，这些算法为我们提供了不同的优化代价函数的方法
> 共轭梯度<br>
> BFGS<br>
> L-BFGS<br>

这三种算法的比较困难，但有许多优点
> 使用其中任意一种算法，通常情况下不需要手动设置α，所以对于这些算法的一种思路是，给出计算导数项和代价函数的方法，你可以认为算法有一个智能的内部循环，而且，事实上，他们确实有一个智能的内部循环，称为线性搜索(line search)算法，它可以自动尝试不同的学习速率 α，并自动选择一个好的学习速率 α，因此它甚至可以为每次迭代选择不同的学习速率，那么你就不需要自己选择。

**总结**：对于BFGS和L-BFGS的使用，或者梯度下降算法的使用，在实际的网络模型中使用并不需要一步步的搭建，可以直接调用函数，这些方法已经被封装在函数中。

## 7.多类别分类：一对多
上边我们说到二元分类问题可以使用逻辑回归来进行判断，对于多分类问题我们采取一种不同的算法来进行分类

我们用三个不同的符号表示三个类别<br>
![逻辑回归算法15](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9515.png)

现在我们有一个训练集好比上图表示的有三个类别 我们用三角形表示 y=1 方框表示 y=2 叉叉表示 y=3。我们下面要做的就是使用一个训练集将其分成**三个二元分类问题**。

我们先从用三角形代表的类别1开始实际上我们可以创建一个新的 "伪" 训练集类型2和类型3定为负类，类型1设定为正类，我们创建一个新的训练集如下图所示的那样我们要拟合出一个合适的分类器。<br>
![逻辑回归算法16](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9516.png)

同理，再把类型2和类型3分别设置为正类，创建训练集，得到不同的逻辑回归分类器<br>
![逻辑回归算法17](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%9517.png)

不同的分类器再经过训练集大量的训练之后，对测试集进行预测时，将所有分类机制都运行一遍，然后对每一个输入变量，取最高可能性的输出变量
> 这里最高可能性就是hθ(x)最大所对应的类别

```py
#!/usr/bin/env python
# coding:utf-8
# author:yiyading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tensor对数据的形状非常挑剔
plt.style.use('fivethirtyeight')    # 格式美化
# plt.style.use('ggplot')           # 这两个参数对格式的优化差不多

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report  # 这个包是评价报告，显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
print("data.head(): \n", data.head())  # 看前五行
print("\ndata.describe(): \n", data.describe())  # 输出样本数量、平均值等参数

# seaborn库是对matplotlib库更高级别的封装
sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 50}
           )
plt.show()  # 输出数据，可视化


def get_X(df):  # 读取特征
    #     """
    #     use concat to add intersect feature to avoid side effect
    #     not efficient for big dataset though
    #     """
    # ones的作用是使X0=1，是为了梯度下降时更新θ0 -> https://github.com/yiyading/NLP-and-ML/blob/master/ML-second.md
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()  # 这个操作返回 ndarray,不是矩阵


def get_y(df):  # 读取标签
    #     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]


def normalize_feature(df):
    #     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())  # 对df进行特征缩放


X = get_X(data)
print("\nX.shape: ", X.shape)
y = get_y(data)
print("\ny.shape: ", y.shape)

# 激活函数，常见的有sigmoid，relu，tanh，在搭建神经网络时，可以直接使用选择参数来选择
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 输出sigmoid函数
fig, ax = plt.subplots(figsize=(8, 6))  # plt.subplots(1, 3, figsize=(8,6)) 产生一个1行3列8*6大小的子图
ax.plot(np.arange(-10, 10, step=0.01),
        sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim((-0.1, 1.1))
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid function', fontsize=18)
plt.show()

theta = np.zeros(3) # X(m*n) so theta is n*1
print("\ntheta: \n", theta)

# 代价函数
def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
# X @ theta与X.dot(theta)等价，表示矩阵相乘
cost(theta, X, y)

def gradient(theta, X, y):  # 批量梯度下降，这个函数实际上是计算梯度，而没有进行梯度更新，梯度下降在搭建神经网络时，通过参数进行选择
#     '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
gradient(theta, X, y)

# 拟合参数
import scipy.optimize as opt
# minimize内部存在两个操作：1.计算各个变量的梯度。2.用梯度更新这些变量
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
print("\nres: \n", res)

# 用训练集预测和验证
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

final_theta = res.x
print("\nres.x: \n", res.x)
y_pred = predict(X, final_theta)
# classification_report()用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
print("\nclassification: \n", classification_report(y, y_pred))

# 寻找决策边界
print("\n final theta: \n", res.x) # this is final theta

coef = -(res.x / res.x[2])  # find the equation
print("\ncoef: \n",coef)

x = np.arange(130, step=0.1)
y = coef[0] + coef[1]*x

print("\ndata.describe(): \n", data.describe())  # find the range of x and y

# 绘制决策边界线
sns.set(context="notebook", style="ticks", font_scale=1.5)
sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 25}
          )
plt.plot(x, y, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')
plt.show()
```
