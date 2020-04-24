# -*- coding: UTF-8 -*-
# 读入红蓝点，画出分割线，包含正则化

# 1 导入所需模块
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他多余信息

df = pd.read_csv('dot.csv')
x_data=np.array(df[['x1','x2']])
y_data=np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1,2)
y_train = np.vstack(y_data).reshape(-1,1)

Y_c = [['red' if y else 'blue'] for y in y_train]


x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(30)


# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
# 打印训练集的每个batch中输入特征的shape和标签的shape，分别为(32, 4) 和(32,)
print('batch:', sample[0].shape, sample[1].shape)


w1=tf.Variable(tf.random.normal([2,11]),dtype=tf.float32)
b1=tf.Variable(tf.constant(0.01, shape=[11]))

w2=tf.Variable(tf.random.normal([11,1]),dtype=tf.float32)
b2=tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01
epoch = 4000

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        
        with tf.GradientTape() as tape:

            h1 = tf.matmul(x_train, w1) + b1
            h1=tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_train - y))
            #添加l2正则化
            loss_regularization = []
            #tf.nn.l2_loss(w)=sum(w ** 2) / 2
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            #求和
            #例：x=tf.constant(([1,1,1],[1,1,1]))
            #   tf.reduce_sum(x)
            #>>>6
            #loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss + 0.01 * loss_regularization


        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)


        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])


    if epoch%200==0:
        print(epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
probs=[]
for x_predict in grid:

    h1 = tf.matmul([x_predict], w1) + b1
    h1=tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

x1=x_data[:,0]
x2=x_data[:,1]
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
