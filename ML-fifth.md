# 神经网络NN模型
在生物学中，每个生物大脑对事物的认知都是通过大量学习所形成的，神经元收到激励信号，经过加工传递给下一个神经元。

神经网络模型类似于神经元的信号传输，接收特征，激活单元激活，输出特征，通过一层有一层的神经元最终输出结果。

神经网络模型的层级分为三大类：
> 输入层（input layer）-> 隐藏层（hidden layer）-> 输出层（output layer）<br>
> 处输出层外，需要为每一层添加一个偏置（bias）。bias类似于y=ax+b中的b，作用是使函数全遍历

除输入层外，每一层的逻辑单元按照不同权重（weight）接受从前一层逻辑单元传入的特征，经过激活单元（激活函数）后输出为下一层的输入。

下图展示一个3层神经网络:<br>
<div align=center>![ML-fifth1](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fifith1.png)<br>

<center>![ML-fifth3](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fifith3.png)</center>

a表示激活单元的输出，θ表示前层到后层映射的权重矩阵，对于上图模型中的激活单元和输出分别表达为：
<center>![ML-fifth4](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith4.png)</center>
> 第二层的bias不从第一层计算得来，在tensor中使用random方法得来，后边会写。

若将所有训练实例喂入神经网络，则需要构建X矩阵，即输入特征矩阵。我们把这样的从左到右的算法称作前向传播算法（forward propagation）。

把x，θ，a分别用矩阵表示：<br>
<center>![ML-fifth2](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith2.png)</center>

神经网络和普通的逻辑回归类似，逻辑回归使用的是原始特征，而神经网络使用的是“进化”后的特征，如果只看最后一层hidden layer和output layer，神经网络和logistic Regression是相同的

<center>![ML-fifith5](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith5.png)</center>

当我们的target不止两种分类（y=1,2,3...)，在输出层我们有与类别数相同的值。

举个例子：
> 训练一个神经网络识别**路人、汽车、摩托车、卡车**，在输出层有4个值，即输出层有四个神经元用来表示这4类神经网络的可能结构如下：<br>
> <center>![ML-fifth6](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith6.png)</center><br>
> 神经网络的算法输出结果为下列四种可能情形之一：<br>
> <center>![ML-fifth7](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith7.png)</center>

# 反向传播（back propagation Algorithm）
假设神经元有m个训练样本，每个包含一组输入x和一组输出信号y，L表示神经元网络层数；Sl表示每层neuron个数，SL表示输出层neuron个数。

神经网络的分类定义为两种情况：
> 二分类：SL=1，y=0 or 1表示哪一类 
> <br><br>
> K类分类：SL=K，yi=1表示分到第i类（K > 2）

上述假设如下图所示：
<center>![ML-fifth8](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith8.png)</center>

使用梯度下降的方法进行反向传播需要使用代价函数，logistic Regression和二分类NN的代价函数相同，但K分类NN的代价函数需要计算K个损失值，也就是K个输出的损失求和。

LR：
<center>![ML-fifth9](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith9.png)</center>

K分类的NN
<center>![ML-fifth10](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith10.png)</center>

上述两个损失函数都是使用了L2正则化。在K分类的NN中，正则化计算的是每一层中所有weight的平方和的叠加。

LR和NN正则化均不对θ0和bias进行处理。

## 反向传播梯度下降
在计算完损失函数J之后，需要使用J的偏导数进行梯度下降，目的是对整个神经网络中的参数进行更新。

我们使用一种反向传播算法来进行计算，我们引入误差额概念，即先计算最后一层的误差，再逐层计算每层误差，直到倒数第二层。

误差是激活单元与实际值之间的误差，我们假设一个四层神经网络前向传播如下：
<center>![ML-fifth11](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith11.png)</center>

计算最后一层的误差
<center>![ML-fifth12](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith12.png)</center>

利用这一误差计算前一层误差，其中g'是sigmoid函数的导数，点乘前边的那一项是权值导致的误差的和
<center>![ML-fifth13](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith13.png)</center>

当计算过所有误差之后，我们可以计算代价函数的偏导数，假设不使用正则化进行偏导数计算
<center>![ML-fifth14](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith14.png)</center>

简单的理解上边式子中i，j的含义：在全连接前向传播中，每一层的任意一个神经元对下一层的所有神经元都有贡献，每个贡献的所占的weight就是我们需要更新的θ参数。

如果考虑正则化，且训练集是一个特征矩阵而非向量，我们需要计算每一层的误差单元将构成一个矩阵，使用如下算法可计算误差单元：（三角形那个特殊符号表示误差单元）
<center>![ML-fifth15](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith15.png)</center>

当计算完误差单元，我们便可以利用以下算法计算代价函数偏导数
<center>![ML-fifth16](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith17.png)</center>

最后进行梯度下降

## 梯度检验
当我们对一个较为复杂的模型（例如神经网络）使用梯度下降算法时，可能会存在一些
不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是最优解。

我们采用梯度的数值检验（Numerical Gradient Checking）来避免这样的问题。

对梯度的估计是沿着切线方向选择离两个非常紧的点然后计算两点的平均值用以估计梯度（二次函数中是斜率）
<center>![ML-fifth17](https://github.com/yiyading/NLP-and-ML/img_ML/ML-fifith17.png)</center>

## 随机初始化
任何优化算法都需要一些初始的参数，我们常采用随机初始化的方法进行初始参数设置。

如果我们不采用随机初始化而是令所有参数都为0，则第二层所有激活单元都有相同的值，那么进行NN算法将没有任何意义。
