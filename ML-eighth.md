# 一、聚类（Clustering）
聚类算法是非监督学习算法之一。

非监督学习与监督学习的不同点是：
> 1. 监督学习中，我们有一个有标签的训练集，我们的目标是找到能够区分正负样本的决策边界。
> 2. 非监督学习中，我们的数据没有任何标签，如下图所示。<br>
> ![ML-eighth1](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth1.png)

非监督学习中，我们将没有任何标签的训练集输入到一个算法，让算法找到这个训练集中的内在结构。

一个能找到下图中圈出的点集（簇）的算法，称为聚类算法。

![ML-eighth2](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth2.png)

## 1.K-means algorithm（K-均值算法）
K-means是最普及的聚类算法，其功能是将一个无标签数据集聚类成不同的簇。

K-means是一个迭代算法，假设我们想要将数据集聚类成K个簇，其方法是：
1. 选择k个随机点，称为**聚类中心（cluster centroids）**。
2. 将数据集中每一个数据与距其最近的**聚类中心**关联起来，这些与同一个聚类中心点关联的所有点聚成一类。
3. 计算每一簇的平均值，将该组所关联的中心移动到平均值的位置。
4. 不断重复2、3步骤，直到中心点不在变化。

下面是一个聚类实例：

迭代一次：<br>
![ML-eigith3](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth3.png)

迭代三次：<br>
![ML-eighth4](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth4.png)

迭代十次：<br>
![ML-eigheh5](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth5.png)

c^(i)^ 代表第i个数据距离最近**聚类中心**的距离， μ~k~ 代表**聚类中心**，迭代过程如下列伪代码所示：
- [ ] 1
- [x] 2
> k代表第k个聚类中心，K代表总共有K个聚类中心。

![ML-eighen6](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth6.png)

从上述伪代码中可以看到，算法分为两步：
1. 第一个for循环是**赋值步骤**，即对每一个样例i，计算其应该属于哪一个类。
2. 第二个for循环是**聚类中心的移动**，即对于一个类k，重新计算该类的质心。

K-均值算法每一次迭代过程，都是一个重新分类的过程，样例在这次迭代属于这一类，在下一次迭代过程可能会属于另外一个类。

## 2.Optimization objective（优化目标）
令所有数据点与其关联的聚类中心的距离之和为x。

K-均值最小化的目标是寻最小的x，因此K-均值的代价函数（又称为**畸变函数（Distortion function给）**为：

![ML-eighth7](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth7.png)

> 从上一小节中的伪代码中可以知道，迭代过程的第一个循环用于减少ci引起的代价，第二个循环用于减小ui引起的代价函数。迭代过程一定是每次都在减小代价函数，否则会出现错误。

## 3. Random initialization（随机初始化）
在运行K-均值算法之前，首先要随机初始化所有聚类中心点。

1. 选择K\<m，即聚类中心点的个数\<训练集中实例的个数。
2. 随机选择K个训练实例，令K个聚类中心与这K个训练实例相等。

第2步的本质是从实例中选择聚类中心。

K-均值的一个问题在于，它有可能停留在一个局部最小值，而这取决于初始化的情况。

为了避免局部最小值的问题，我们通常运行多次K-均值算法，每次都重新进行随机初始化，最后再比较多次K-均值的结果，选择畸变函数（代价函数）最小的结果。伪代码如下图所示：

![ML-eighth8](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth8.png)
> 这种方法在K较小（2~10）时还可行，但如果K较大，这么做也可能不会显著改善。

## 4.Choosing the number of clusters（选择聚类数）
没有最好的选择聚类数的方法。

通常是根据不同的问题，人工选择聚类数。

在讨论选择聚类数的方法时，有可能会谈及一个方法叫做**“Elbow method（肘部法则）”**。

![ML-eighth9](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth9.png)

左图这条曲线很像人的肘部，在这种模式下，当K小于3时，畸变值下降的非常快；K大于3时，畸变值下降的就很慢。

左图看起来使用3个聚类是正确的，这是因为K=3时曲线的肘点，当应用“肘部法则”时，得到的左图时选择聚类个数的合理方法。

但有时候会得到右图，在这种情况下”肘部法则“就不太好用了。

# 二、降维（Dimensionality Reduction）
降维就是将数据的维数降低，通俗的理解就是通过一定的手段，减少特征的数量。

降维是无监督学习中的一类问题。
## 1.动机一：数据压缩（Motivation 1：Data Compression）
Data Compression不但能压缩数据，进而使用较少的计算机内存或磁盘空间，而且能够加快学习算法。

将数据从二维降至一维：假设使用两种仪器测量一些东西的尺寸，分别为x1（长度厘米），x2（长度英寸）。我们希望测量结果作为我们机器学习的特征，但问题是两种仪器对同一个东西测量的结果不完全相等（误差、精度），而将两者都作为特征有些重复，因此希望将这个二维数据降至一维。

2D降维至1D的过程如下图所示：

![ML-eighth10](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth10.png)

3D降维至2D的过程与上图类似，是将三维空间中的点投射到二维平面上。如下图所示：
![ML-eighth11](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth11.png)

这种投射的处理过程，可以将任何维度的数据降到任何想要的维度（1000D->100D）。

## 2.动机二：数据可视化（Motivation 2：Visualization）
如下图所示，每一个国家对应50个特征（如GDP，人均GDP，平均寿命等）。

![ML-eighth12](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth12.png)

如果想将这个50维的数据可视化是不可能的。使用降维的方法将其降维至二维，我们便可以将其可视化。如下图所示：

![ML-eighth13](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth13.png)

**注意**：降维的算法只负责减少维数，新产生特征的意义必须由我们自己去挖掘。

## 4.Principal Component Analysis Problem Formulatin（PCA，主成分分析）
**PCA是最常见的降维算法**。

在PCA中，我们要做的是找到一个方向向量（Vector direction），当我们把所有数据投射到该向量上时，我们希望投射平均均方误差尽可能地小。
> 1. 方向向量是一个经过原点的向量。
> 2. 投射误差是从特征向量向该方向向量作垂线的长度。

下图展示了使用PCA将2D降维至1D：

![ML-eighth14](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth14.png)

PCA与线性回归是两种不同的算法，PCA是最小化投射误差（Projected Error），线性回归最小化预测误差。

下面两幅图片中，左图为线性回归，右图为PCA：

![ML-eighth15](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth15.png)

PCA技术的一大好处是对数据进行降维的处理。将n个特征降维到k个，可以用来进行数据压缩，如果100维的向量最后可以用10维来表示，那么压缩率为90%。在图像处理领域的KL变换使用PCA做图像压缩。但PCA要保证降维后，还要保证数据的特性损失最小。

PCA技术的一个很大的优点是它完全无参数限制。在PCA的计算过程中完全不需要人为的设定参数或是根据任何经验模型对计算进行干预，最后的结果只与数据相关，与用户是独立的。**这同样使得用户对观测对象的先验知识完全无效**。

PCA算法过程：PCA从n维降至k维
1. 均值归一化（所有特征均值为0）。特征替换的过程如下图所示：

![ML-eighth16](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth16.png)

2. 计算**协方差矩阵（covariance matrix）Σ**：

![ML-eighth17](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth17.png)

3. 计算**协方差矩阵Σ的特征向量（eigenvectors）**

## 4.Choosing The Number of Principal Components（选择主成分的数量）
PCA是减少投射的平均均方误差。

主成分的数量选择与所选主成分与原特征数量的相关性有关。

确定主成分数量k的步骤如下：

> 我们希望在平均均方误差与训练集方差比例尽可能小的情况下选择k的值。

下图表示了步骤中的数学表达式：

![ML-eighth18](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth18.png)

> 如果我们希望比例小于1%，就意味着原来数据的偏差有99%都保留下来了。如果我们选择保留95%的偏差，便能非常显著的降低模型中特征的维度。

通过选定的不同比例，来确定k的取值。

## 5.Reconstruction from Compressed Representation（压缩表示的重建）
PCA可以将1000D的数据降维到100D。

给定100D的Z，如何回到原来1000D的x。

如下图所示，在从2D到1D的过程中，会经历一个U，在升维的时候，只需U\*Z，即可得到Xapprox，它近似等于x

![ML-eighth19](https://github.com/yiyading/NLP-and-ML/blob/master/img/ML-eighth19.png)

这就是从低维回到未压缩前的过程。

## 6.Advice for Applying PCA
假设我们正在针对一张100\*100像素的图片进行某个计算机视觉的ML，总共10000个特征。

步骤：
1. 运用主成分分析将数据压缩至100个特征
2. 对训练集运行学习算法
3. 在预测时，采用之前学习而来的Ureduce将输入的特征x转换成特征向量z，然后进行预测。

**错误应用1**：PCA用于减少过拟合（减少特征数量）

使用PCA减少特征数量可能会丢失非常重要的特征，因为它并不考虑任何与结果变量有关的信息。

而使用正则化处理时，会考虑到结果变量，不会丢掉重要数据。

**错误应用2**：默认PCA作为学习过程的一部分，有些时候会有效果，但还是从原始特征开始比较好。
