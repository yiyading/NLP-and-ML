# 卷积神经网络
对于RGB图片，因为待优化参数过多，如果使用全连接会导致模型过拟合。对全连接进行改进->卷积神经网络（CNN）对原始图像进行特征提取，然后再将提取的特征喂给全连接网络。

## 1.卷积的概念
卷积可认为是一种有效提取图像信息中特征的方法。一般使用一个正方形的卷积核遍历图片上的所有点，图像区域每一个像素值乘以卷积核内相对应点的weight，求和，再加上偏置。

卷积核通道数与输入特征一致。使用RGB图片作为输入，采用三个卷积核分别对R、G、B进行特征提取，三通道的提取出的特征相加，然后映射到下一层channel。

多个卷积核可实现对同一输入层多次特征提取，卷积核个数决定输出层channel数。
> 如图使用6个卷积核，对输入层进行两次特征提取，output层channel数为2.<br>
> ![CNN1]()

**感受野**：卷积神经网络各输出层每个像素点在原始图像上的映射区域大小。
![CNN2]()

**输出特征尺寸计算**：输出图片边长=（输入图片边长 - 卷积核长 + 1） / 步长<br>
此图的输出图片边长 = (5-3+1)/1 = 3
![CNN3]()

**全零填充（padding）**：在输入层周围填上一圈零。<br>
在Tensor中
> 参数padding='same'表示全零填充，输出图片边长=输入图片变长/步长<br>
> <br>
> 参数padding='valid'表示不全零填充，输出图片边长计算无变化，但结果向上取整数

python代码实现
```py
tf.keras.layers.Conv2D(
	input_shape=(高, 宽, 通道数),	# 仅在第一层有
	filters=卷积核个数,
	kernel_size=卷积核尺寸,
	strides=卷积核步长,
	padding='valid' or 'same',
	activation='relu' or 'sigmoid' or 'tanh' or 'softmax',	# 如果有BN此层不写
)

# 在实际的网路搭建中，神经网络会有很多层，因此会采用tf.keras.models.Sequential函数进行搭建
model = tf.keras.models.Sequential([
	Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(5,5), padding='same', activation='relu'),
	MaxPooling2D(),
	Conv2D(32, 3, padding='same', activation='relu'),	
	MaxPooling2D(),

	Flatten(),
	Dense(512, activation='relu',
	Dense(10, activation='softmax',
])
```

**批归一化（Batch Normalization，BN）**：对一小批数据在网络各层进行归一化处理。<br>

将每层的数据输入减去其均值再除以标准差：<br>
![CNN4]()

上式中各参数的含义如下：<br>
![CNN5]()

在激活函数中，当数值超过一定的范围，激活函数图像的斜率为0，梯度消失。BN的作用是把每一层的输入调整到均值0，方差为1的标准正态分布，解决梯度消失。

对于sigmoid函数，BN会造成激活函数在[-2,2]区间内近似线性函数，深层网络能力下降：<br>
![CNN6]()

解决办法是给每个卷积核引入可训练参数γ核β，调整BN的力度。
![CNN7]()

输入特征与卷积核乘加计算 -> BN -> 激活层

**池化（pooling）**：用于减少特征数量
> 最大值池化可提取图片纹理。<br>
> <br>
> 均值池化可保留背景特征。<br>

tensorflow中的池化层
```py
# 最大值池化
tf.keras.layers.MaxPool2D(
pool_size=池化核尺寸，
strides=池化步长，
padding=‘valid’or‘same’
)

# 平均值池化
tf.keras.layers.AveragePool2D(
pool_size=池化核尺寸，
strides=池化步长，
padding=‘valid’or‘same’
)
```

**舍弃（Dropout）**：神经网络训练时，按照**概率**舍弃部分神经元的训练，使用时恢复，提高训练速度。

```py
tf.keras.layers.Dropout(舍弃概率)

Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(5, 5), padding='same'),  # 卷积层
BatchNormalization(),  # BN层
Activation('relu'),  # 激活层
MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
Dropout(0.2),  # dropout层
```

总结：
> 1. CNN就是借助卷积核提取特征后送入全连接网络
> 2. Conv2D -> BatchNormalization -> Activation -> Pool2D -> Dropout

## 2.数据增强
目的是针对小数据量的扩展
```py
image_gen_train = ImageDataGenerator(
	rescale=1./255, 	#原像素值0～255归至0～1
	rotation_range=45, 	#随机45度旋转
	width_shift_range=.15, 	#随机宽度偏移[-0.15,0.15)
	height_shift_range=.15, #随机高度偏移[-0.15,0.15)
	horizontal_flip=True, 	#随机水平翻转
	zoom_range=0.5 		#随机缩放到[1-50％，1+50%])

# 对x_train进行数据增强
image_fen_train.fit(x_train)
```

## 3.绘制loss曲线和acc曲线
```py
history=model.fit(训练集数据, 训练集标签, batch_size=, epochs=,validation_split=用作测试数据的比例,validation_data=测试集,shuffle=True, validation_freq=测试频率)

history:
loss：训练集loss
val_loss：测试集loss

sparse_cateforical_accuracy：训练集准确率
val_sparse_cateforical_accuracy：测试集准确率
```
