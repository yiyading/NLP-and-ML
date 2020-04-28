# 正则化
正则化的目的是为了解决overfitting。
1. L1正则化是通过稀疏参数（减少参数的数量）来降低复杂度。
2. L2正则化通过使参数接近零但不为零来降低复杂度。在梯度下降时，加大对θj的惩罚。
> 也可以这样说，L1减少特征的数量；L2保留全部特征，但是降低其某些参数的值。
> <br>
> 如果能确定哪个特征造成了overfitting，就惩罚与之对应的θj，否则惩罚所有θ，但不惩罚θ0，因为正则化对θ0的影响非常小。

![ML-fourth1](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth1.png)

## 1.代价函数
L2正则化对θ的惩罚是在代价函数中体现的。

![ML-fourth3](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth3.png)<br>
![ML-fourth2](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth2.png)<br>

λ又称为正则化参数（regularization parameter），第一个式子中确定θ3和θ4所对应的特征造成了overfiting，对其参数进行惩罚，第二个式子无法确定哪个特征造成正则化，惩罚所有参数。

经过正则化处理的模型与原模型可能的对比如下：<br>
![ML-fourth4](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth4.png)

正则化参数λ如果选择的过大，会把所有参数最小化，导致模型hθ(x)=θ0，也就是上图中红线所示情况，造成欠拟合。

> 下文会详述正则化后的代价函数如何在线性/逻辑回归中使用。

## 2.正则化线性回归和逻辑回归
正则化后的回归算法和未正则化的回归算法的区别在于**代价函数中是否对参数的惩罚**。

使用正则化后的代价函数对参数进行更新时，对θ0的更新不使用正则化。

下边是线性回归和逻辑回归的J(θ)对比图
> 线性回归<br>
> ![ML_fourth5](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth5.png)

> 使用sigmoid的逻辑回归，在[ML-third](https://github.com/yiyading/NLP-and-ML/blob/master/ML-third.md)有关于损失函数的重定义<br>
> ![ML-fourth6](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth6.png)

使用以上两个式子显示的损失函数J(θ)进行参数迭代（梯度下降）
> 线性回归<br>
> ![ML-fourth7](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth7.png)

> 逻辑回归<br>
> ![ML-fourth7](https://github.com/yiyading/NLP-and-ML/blob/master/img_ML/ML-fourth7.png)

逻辑回归看上去和线性回归有些一样，但是其中的h经过了sigmoid过程

python实现逻辑回归
```py
import numpy as np
def costReg(theta, X, y, lr):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	
	# 重定义的代价函数
	first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
	second = np.multiply((1-y), np.log(1 - simoid(X * theta.T)))

	# 添加惩罚
	reg = (lr / (2*len(X))) * np.sum(np.power(theta[ : , 1:theta.shape[1]], 2))

	return np.sum(first - second) / (len(X)) + reg
```

