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

从下边一系列的函数中可以看出，多变变量与单变量变量唯一不同的地方就在于参数/特征更多了。

> 多变量代价函数
![多变量代价函数](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0.png)

> 多变量梯度下降算法
![多变量梯度下降算法](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95.png)

> 多变量线性回归梯度下降算法，**注意前面引入了X0=1，所以这里的j从0开始**
![多变量线性回归梯度下降算法](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/%E5%A4%9A%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95.png)


