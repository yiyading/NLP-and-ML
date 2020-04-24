# 优化器
# 1.SGD（随机梯度下降）
每次从训练集中随机选择一个batch来进行学习<br>
![AI-second1](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second1.png)

**SGD通过一阶动量和二阶动量计算出每次梯度的下降值，以此来实现随机梯度下降**

缺点：
> 1.学习率难确定<br>
> 2.对所有参数更新时应用同样的学习率，对于稀疏数据，理想情况使对出现频率低的特征进行较大更新<br>
> 3.对于非凸函数，易陷于局部极小值

## 一阶动量与二阶动量
![AI-second2](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second2.png)

一阶动量是各个时刻梯度方向的指数滑动平均值，是最近一段实践梯度和的平均值
> β<1接近1，一般小于等于0.9

二阶动量为之前所有梯度的平方和

## 优化器框架
待优化参数w，损失函数loss，初始学习率lr，每次迭代一个batch，t表示当前第几次batch迭代

1. 计算t时刻损失函数关于当前参数的梯度<br>
![AI-second3](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second3.png)

2. 计算t时刻一阶动量mt和二阶动量Vt

3. 计算t时刻下降梯度<br>
![AI-second4](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second4.png)

4. 计算t+1时刻的梯度w<br>
![AI-second5](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second5.png)


## momentum
**在SGD的基础上增加一阶动量**

在等高线的某些区域（某些方向教另一些方向上陡峭的多，常见于局部极值点），SGD会在这些地方附近震荡，从而导致收敛速度慢。

上述情况，通过动量（momentum）便可以解决。

> 无momentum的SGD没有动量的概念<br>
> ![AI-second6](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second6.png)

> 有momentum的SGD<br>
> ![AI-second8](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second8.png)

> 有无momentum的差别图<br>
> ![AI-second9](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second9.png)
# 2.Adagrad（自适应梯度算法）
**在SGD基础上增加二阶动量**

引入二阶动量，对每个参数分配自适应学习速率，对**低频参数更新大，对高频参数跟新小**，较SGD有更高的鲁棒性

优点：
> 减少学习率的手动调节

缺点：
> 仍需要手动设置全局学习率<br>
> 分母不断积累，导致学习率收缩，最终迫使训练提前结束

## 优化器框架
和SGD类似，但是**为了避免分母为零，在分母上加一个小的平滑项**<br>

![AI-second7](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second7.png)

# 3.Adadelta（自适应增量算法）
**在SGD基础上增加二阶动量**

Adagrad的学习率变化过于激进，可考虑不累计全部历史梯度，只关注指数滑动平均。即用一阶动量的定义方法，定义二阶动量

优点：
> 避免了二阶动量持续累积导致训练提前结束

## 优化器框架
和SGD中动量定义方法不同

![AI-second10](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second10.png)

## 4.Adam
Momentum在SGD基础上增加一阶动量，即Momentum
AdaGrad和Adadelta在SGD基础上增加二阶动量，Adaptive
Adam把一阶动量和二阶动量结合起来，即融合Adaptive + Momentum。

![AI-second11](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/optimizing/img/AI-second11.png)

Adam的python实现
```py
# adam

import tensorflow as tf
import os
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息


def normalize(train):
    # 线性归一化，使数据处理更加便捷快速，避免数据太大造成梯度爆炸
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i])) / (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T


def norm_nonlinear(train):
    # 非线性归一化（log）
    x_data = train.T
    for i in range(4):
        x_data[i] = np.log10(x_data[i]) / np.log10(tf.reduce_max(x_data[i]))
    return x_data.T


def standardize(train):
    # 数据标准化（标准正态分布），使每个特征中的数值平均值为0，标准差为1
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - np.mean(x_data[i])) / np.std(x_data[i])
    return x_data.T


x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

x_data = standardize(x_data)

# 随机打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

print("x.shape:", x_data.shape)
print("y.shape:", y_data.shape)
print("x.dtype:", x_data.dtype)
print("y.dtype:", y_data.dtype)
print("min of x:", tf.reduce_min(x_data))
print("max of x:", tf.reduce_max(x_data))
print("min of y:", tf.reduce_min(y_data))
print("max of y:", tf.reduce_max(y_data))

# from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)

# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))



learning_rate_step = 10
learning_rate_decay = 0.8
train_loss_results = []
test_acc = []
lr = []
epoch = 500
loss_all = 0
learning_rate_base=1
delta_w, delta_b = 0, 0
beta = 0.9
global_step = tf.Variable(0, trainable=False)
m_w, m_b = 0, 0
v_w, v_b = 0, 0
beta1, beta2 = 0.9, 0.999

for epoch in range(epoch):
    learning_rate = learning_rate_base * learning_rate_decay ** (epoch / learning_rate_step)
    lr.append(learning_rate)
    for step, (x_train, y_train) in enumerate(train_db):
        global_step = global_step.assign_add(1)

        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.sigmoid(y)
            y_onehot = tf.one_hot(y_train, depth=3)
            # mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_onehot - y))
            loss_all += loss.numpy()
        # compute gradients
        grads = tape.gradient(loss, [w1, b1])
        
        # adam
        m_w = beta1 * m_w + (1 - beta1) * grads[0]
        m_b = beta1 * m_b + (1 - beta1) * grads[1]
        v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
        v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])
        
        m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
        m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
        v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
        v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))
        
        w1.assign_sub(learning_rate * m_w_correction / tf.sqrt(v_w_correction))
        b1.assign_sub(learning_rate * m_b_correction / tf.sqrt(v_b_correction))

        if step % 10 == 0:
            print("step=", step, 'loss:', float(loss))
            print("lr=", learning_rate)
    train_loss_results.append(loss_all / 3)
    loss_all = 0
    
    
    # test(做测试）
    total_correct, total_number = 0, 0
    for step, (x_test, y_test) in enumerate(test_db):
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.sigmoid(y)
        
        pred = tf.argmax(y, axis=1)
        
        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int64)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc:", acc)
    print("---------------------")


# 绘制 loss 曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()

# 绘制 Learning_rate 曲线
plt.title('Learning Rate Curve')
plt.xlabel('Global steps')
plt.ylabel('Learning rate')
plt.plot(range(epoch+1), lr, label="$lr$")
plt.legend()
plt.show()

```
