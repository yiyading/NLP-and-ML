# 全连接网络

# 1.tf.keras搭建网络八股
六步搭建网络，总结起来就3句话，导入库并设置数据集，设置网络结构，选择数据集
```py
# 导入库并设置数据集
import
x_train,y_train

# 设置网络结构
model = tf.keras.modules.Sequential
model.compile

# 选择数据集并plot
model.fit
model.summary
```

## 1.描述网络结构
model = tf.keras.models.Sequential([网络结构])
> 拉直层：tf.keras.layers.Flatten(input_shapt=输入数量)

> 全连接层：tf.keras.layer.Dense(神经元个数, activation='', kernel-regularizer=正则化)<br>
> activation: relu, softmax, sigmoid, tanh<br>
> kernel_regularizer: tf.keras.regularizers.l1(), tf.keras.regularizersl2()

> 卷积层：tf.keras.layers.Conv2D(filters = 输出维数,kernel_size= 卷积核尺寸, strides = 卷积步长, padding = " valid" or "same")

> LSTM层：tf.keras.layer.LSTM()

## 2. 反向传播选择
model.compile(optimizer=优化器,loss=损失函数, metrics=['损失函数'])<br>

sgd是随机梯度下降，这些都是不同的梯度下降的方法，一个模型只能有一个optimizer
> optimizer:<br>
> ‘sgd’ortf.optimizers.SGD(lr=学习率,decay=学习率衰减率,momentum=动量参数)<br>
> ‘adagrad’ortf.keras.optimizers.Adagrad(lr=学习率,decay=学习率衰减率)<br>
> ‘adadelta’ortf.keras.optimizers.Adadelta(lr=学习率,decay=学习率衰减率)<br>
> ‘adam’ortf.keras.optimizers.Adam(lr=学习率,decay=学习率衰减率)<br>

loss损失函数可以使用预定义的，也可以使用自定义的
> loss:<br>
> 'mse' or tf.keras.losses.MeanSquaredError()<br>
> 'sparse_categorical_crossentropy' or tf.keras.losses.SparseCategoricalCrossentripy()

metrics是一个列表，包含评价模型在训练和测试时的性能指标
> metrics:<br>
> sparse_categorical_accuracy=1/2<br>
> accuracy=3/4

## 3.设定训练测试集
model.fit(训练集的数据,训练集的标签,batch_size= , epochs= , validation_split= 用作测试数据的比例, validation_data= (测试集的数据，测试集的标签), shuffle = True or False, validation_freq= 多少epoch测试一次)

# 实例
```py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test /255.0

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatter = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatter(x)
        x = self.d1(x)
        y = self.d2(x)
        return y

model = MnistModel()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
model.summary()
```
