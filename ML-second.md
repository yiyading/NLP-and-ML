# 多变量线性回归

## 1.多维特征
多变量/特征回归模型与单变量/特征回归模型的不同之处在于，多变量/特征回归模型中每一个训练实例中的特征有很多。

我们还是以房价模型为例，为其添加例如房间数等特征，模型的特征为(X1, X2,...,Xn)

![房屋多变量模型](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E6%88%BF%E5%B1%8B%E5%A4%9A%E5%8F%98%E9%87%8F%E6%A8%A1%E5%9E%8B.png)

添加了更多特征后，我们引入新的解释：

![多变量回归模型引入更多特征后的新解释](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B%E5%BC%95%E5%85%A5%E6%9B%B4%E5%A4%9A%E7%89%B9%E5%BE%81%E5%90%8E%E7%9A%84%E8%A7%A3%E9%87%8A.png)

引入X0=1作为与θ0相乘的一个特征。其意义是为了计算方便。

此时模型中的参数θ是一个n+1维向量，而每个训练实例也是n+1维向量。特征矩阵X的维度是m\*(n+1)。因此公式可以简化为：<br>

![多维特征参数矩阵与特征矩阵相乘](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E7%BB%B4%E7%89%B9%E5%BE%81%E5%8F%82%E6%95%B0%E7%9F%A9%E9%98%B5%E4%B8%8E%E7%89%B9%E5%BE%81%E7%9F%A9%E9%98%B5%E7%9B%B8%E4%B9%98.png)
> 其中T代表矩阵转置

## 2.多变量梯度下降
多变量线性回归函数的代价函数与单变量线性回归函数的代价函数的构建思想相同，是所有建模误差的平方和；梯度下降的参数更新方法也是相同的。

从下边一系列的函数中可以看出，多变变量与单变量变量唯一不同的地方就在于参数/特征更多了。<br>

> 多变量代价函数<br>
> ![多变量代价函数](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0.png)<br>

> 多变量梯度下降算法<br>
> ![多变量梯度下降算法](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95.png)<br>

> 多变量线性回归梯度下降算法，**注意前面引入了X0=1，所以这里的j从0开始**<br>
> ![多变量线性回归梯度下降算法](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95.png)<br>

代价函数的python实现：<br>
```py
def computeCost(x, y, theta):
	inner = np.power(((x * theta) - y), 2)	// 需要import numpy ad np
	return np.sum(inner)/(2 *len(x))
```

## 3.梯度下降实践
### 1.特征缩放
在面对多维特征问题时，我们要保证这些特征具有相近的尺度，这将帮助梯度下降算法更快收敛。

> 不同特征之间的取值范围相差过大，在构建特征参数所组成的代价函数时，会随着取值范围的差距增大，代价函数J的等高线越扁。<br>
> ![Feature Scaling](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/Feature%20Scaling.png)

最简单的求范围的方法是：Xn = （Xn - 平均值）/ 标准差

### 2.学习率
通过绘制迭代次数和代价函数的图标来表示观测算法何时收敛。

![迭代-J收敛](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E8%BF%AD%E4%BB%A3-J%E6%94%B6%E6%95%9B.png)

梯度下降每次迭代都受学习率α的影响，在[ML-first](https://github.com/yiyading/NLP-and-ML/blob/master/ML-first.md)已经陈述了阿尔法过大或过小会产生的影响。

通常考虑尝试的学习率可以为：
α = 0.01, 0.03, 0.1, 0.3, 1, 3, 10

## 4.特征和多项式回归
线性回归并不适用于所有数据，有些时候，需要用曲线来适应我们的数据，比如说：

> 二次或三次模型<br>
> ![二次或三次模型](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E4%BA%8C%E6%AC%A1%E6%88%96%E4%B8%89%E6%AC%A1%E6%A8%A1%E5%9E%8B.png)

二次模型通常会有一个下降的通道，在房屋size-price模型中，随着size的上升，price大概率上并不会下降。一次可以采用开根号的模型。
![开根号模型](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%BC%80%E6%A0%B9%E5%8F%B7%E6%A8%A1%E5%9E%8B.png)

* 注意：如果采用多项式回归模型，在运行梯度下降算法时，特征缩放非常重要，
> 如果不使用特征缩放，对多项式回归模型，不同特征之间数量级可能差几百或者几百万倍。

## 5.正规方程
对于参数/特征数量较少的问题，使用正规方程是更好的方法。
> 实际中大部分还是梯度下降，因为实际问题中参数可能有几百万个。

以单变量二次代价函数J为例，正规方程本质上是求J的极小值所对应的点，也就是J’=0处为代价函数的局部最小值。

正规方程的原理比较简单，就是使用线性代数的知识，直接求解最优解。举个例子：

> 假设训练集的特征矩阵X（包含x0=1）并且训练集结果为向量y，则参数限量θ：<br>
> ![正规方程解出向量θ](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B%E8%A7%A3%E5%87%BA%E5%90%91%E9%87%8F%CE%B8.png)

从下图梯度下降和正规方程对比图中可以看出，当参数/特征小于1W时，正规方程比较适用，这是应为正规方程的计算中涉及到了对**矩阵的转置和矩阵的逆**的运算，这种计算对计算机算力要求较高。

![梯度下降和正规方程对比图](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%92%8C%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B%E5%AF%B9%E6%AF%94.png)

注意：**正规方程只适用于线性模型**，因为

正规方程的python实现：
```py
import numpy as np

def normalEqn(X, y):
	// pyton中有一个魔幻的特点，就是调用里面有调用->np.linalg.inv
	theta = np.linalg.inv(X.T @ X) @ X.T @ y	// X.T@X = X.T.dot(x)
	return theta
```

注意：**有些矩阵不可逆**，比如m=10，n=101，训练样本远远小于参数，可能会造成X的转置与X相乘不可逆，解决这个问题，有以下机中方法：

> 1.删除特征值中的多余特征，比如说两个特征是线性相关的<br>
> <br>
> 2.删除较少使用的特征，或者正则化<br>
> <br>
