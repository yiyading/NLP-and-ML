当开发一个机器学习系统时，如何确定机器学习下一步得优化方向？

加大样本数量？加大特征数量？

当我们运用训练好的模型来预测未知数据时发现有较大的误差，可选的优化方法有如下几种：
> 获得更多的训练实例<br>
> <br>
> 尝试减少特征数量<br>
> <br>
> 尝试增加特征数量<br>
> <br>
> 尝试增加多项式特征，即不同的特征相乘X1\*X2，或者某一个特征开方X2^2<br>
> <br>
> 尝试减少正则化程度λ<br>
> <br>
> 尝试增加正则化程度λ<br>

在选择优化方法时，我们需要借助一些“机器学习诊断法”来判断选择什么样的优化方法。

## 1.评估假设
当我们评估一个模型好坏时，可以通过判断模型的拟合情况来判断，对于变量（特征）很少的情况，可以通过画图直接观察。但对于变量（特征）较多的情况，画图就很困难甚至不可能

我们将一个数据集分为train（训练集）和test（测试集），通常时7：3的比例进行划分。

如下图所示，假设我们要在10个不同次数的二项式模型之间进行选择，使用train集选出这10个模型中代价函数J最小的一个模型，然后将该模型应用于test集。

![ML-sixth1](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth1.png)

对于不同的回归模型，我们用test集计算代价函数的方法有些区别。

对于线性回归模型，我们利用test集计算代价函数：

![ML-sixth2](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth2.png)

对于逻辑回归模型，我们除了可以利用test集计算代价函数，还可以应用误分类比率，对于每个test集实例，计算所有误分类，然后对结果求平均值：

![ML-sixth3](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth3.png)

通过test集计算出来的J可以判断模型的好坏。

## 2.模型选择和交叉验证集
在**评估假设**中，我们使用10个不同次数的二项式模型进行模型选择，一般来说次数越高，使用train集训练出模型的J越小。

但是通过train的出来的模型可能很好的拟合train集，但不能拟合test集。

我们使用交叉验证集（cross_validation)来帮助选择模型。

train：cross_validation：test = 6:2:2

模型选择方法：
1. 使用train集训练出10个模型。
2. 用cross_validation分别验证这10个模型，选出交叉验证误差(J)最小的那个模型。
3. 用第2步得出的模型对test集进行计算得出推广误差(J)。

计算这3个数据集的方法类似，如下图所示：

![ML-sixth4](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth4.png)

## 3.偏差（bias）和方差（variance）
当运行一个算法时，如果这个算法的效果不好，那这个模型大概率会出现的问题是**偏差大**或者**方差大**，也就是欠拟合和过拟合。

![ML-sixth5](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth5.png)

下图将train集和cross_validation集的损失函数与多项式的次数绘制在同一张图表上。结合上图分析
> 当特征选取的较少时，无论怎么优化，train集和cross_validation集的J都很高，模型处于underfit状态，high bias。<br>
> <br>
> 当特征选取的较多时，模型很好的拟合了train集，J很小，但不能很好拟合cross_validation集，J很高，模型处于overfit状态，high variance。<br>
> <br>
> 当特征选取适中时，train集和cross_validation集的J都很小。

![ML-sixth6](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth6.png)

bias描述的是样本拟合出的模型的输出效果与真实效果的差距。

variance描述的是train集上拟合出的模型在测试集上的表现。

![ML-sixth7](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth7.png)

## 4.正则化和bias/variance
在训练模型时，我们一般使用正则化的方法来防止过拟合，但是如果正则化程度太高或者太小，也会造成过拟合/欠拟合问题。

![ML-sixth8](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth8.png)

如上图所示，如果正则化程度过高，会对除θ0之外所有的参数进行过大的惩罚，造成underfit情况；如果正则化程度过低，会对所有参数惩罚过小，造成overfit情况。

下图将train集和cross_validation集的损失函数与正则化参数λ绘制在同一张表上。

![ML-sixth9](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth9.png)

## 5.学习曲线
通过学习曲线，我们能判断出一个学习算法是处于**high variance/overfit**还是处于**high bias/underfit**状态。

如下图所示，当处于**high bias/underfit**的状态下，无论如何增加数据量都无法train集和cross_validation的J会在一个较高的线上下而不会下降。

![ML-sixth10](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth10.png)

如下图所示，当处于**high variance/overfit**状态下，增加数据量，train集的J会增高，但是其增高幅度小于cross_validation集的J的下降幅度。

![ML-sixth11](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth11.png)

* 通过学习曲线中train和cv的J的变化程度，来判断模型处于什么状态。

## 6.总结
在模型遇到不同的问题时，我们需要用不同的方法来解决这些问题。

解决高方差
> 1. 获得更多的训练实例
> 2. 尝试减少特征数量
> 3. 增加正则化程度λ

解决高偏差
> 1. 尝试获得更多的特征
> 2. 尝试增加多项式特征（旧特征的组合）
> 3. 减少正则化程度λ

使用较小的神经网络，类似于参数较少的情况，容易出现**high bias/underfit**的情况。

使用较大的神经网络，类似于参数较多的情况，容易出现**high variance/overfit**的情况。但可以通过正则化来调整。

通常选择较大的神经网络并采用正则化比采用较小的神经网络的效果要好。

![ML-sxit12](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-sixth12.png)

对于神经网络的隐藏层的层数选择，通常从一层开始逐渐增加层数。

为了更好的选择，把数据集分为train、cross_validation、test这三个数据集，针对不同隐藏层的神经网络训练神经网络，选择cross_validation代价函数最小的神经网络。


