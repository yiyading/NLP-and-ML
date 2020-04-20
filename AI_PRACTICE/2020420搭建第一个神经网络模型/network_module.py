 # from_tensor_slicesimport tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

os.environ['Tf_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv("./project_1/iris.txt",header=None)

# replace()字符替换
df.replace("Iris-setosa",0,inplace=True)
df.replace("Iris-versicolor",1,inplace=True)
df.replace("Iris-virginica",2,inplace=True)

# 按列切片：切出0到3列
x_data0 = df.iloc[:, 0:4]   
y_data0 = df.iloc[:, 4]

# 生成列表
x_data = np.array(x_data0)  
y_data = np.array(y_data0)

# 随机打乱数据
# 使用相同的seed，使输入特征/标签一一对应
np.random.seed(116)
np.random.shuffle(x_data)  #对数组重新洗牌
np.random.seed(116)
np.random.shuffle(y_data)

#分出训练集和测试集
x_train= x_data[:-30]  #训练集的输入特征
y_train= y_data[:-30]  #训练集的标签
x_test=x_data[-30:]
y_test=y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test=tf.cast(x_test, tf.float32)

print("x.shape:", x_data.shape)
print("y.shape:", y_data.shape)
print("x.dtype:", x_data.dtype)
print("y.dtype:", y_data.dtype)

# from_tensor_slices：切分传入的 Tensor 的第一个维度，生成输入特征/标签对，生成相应的 dataset
# 配成[输入特征，标签]对，每次读入一批(batch)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# iter用来生成迭代器,next()返回迭代器的下一个项目
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

w1 = tf.Variable(tf.random.truncated_normal([4, 32], stddev=0.1, seed=1)
# b1是一行32列
b1 = tf.Variable(tf.random.truncated_normal([32], stddev=0.1, seed=1))  
w2 = tf.Variable(tf.random.truncated_normal([32, 3], stddev=0.1, seed=2))
b2 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=2))

lr = 0.1
train_loss_results = []
epoch = 120  # 1个epoch等于使用训练集中的全部样本训练一次


loss_all=0	# 损失
for epoch in range(epoch):
	for step, (x_train, y_train) in enumerate(train_db):
	# enumerate:将可遍历的数据对象(如列表、元组 或字符串)组合为索引和元素。

		# 前向传播，tf.matmul:矩阵相乘
		# python的broadcast机制，numpy矩阵相加会自动扩展
	        with tf.GradientTape() as tape:
		        h1 = tf.matmul(x_train, w1) + b1
        		h1 = tf.nn.relu(h1)  #Relu函数
	#       	h1 = tf.nn.sigmoid(h1)  #sigmoid函数
	#       	h1 = tf.nn.tanh(h1)  #tanh函数
        	 	y = tf.matmul(h1,w2) + b2

			y_onehot = tf.one_hot(y_train, depth=3)
			# 均方误差MSE = (sum(y-out)^2)/n计算损失函数
			loss = tf.reduce_mean(tf.square(y_onehot - y))
			loss_all+=loss.numpy()

		# compute gradients
		grads = tape.gradient(loss, [w1, b1, w2, b2])
			#梯度下降法
		w1.assign_sub(lr * grads[0])
	        b1.assign_sub(lr * grads[1])
		w2.assign_sub(lr * grads[2])
	        b2.assign_sub(lr * grads[3])

		if step%100==0:
        		print(epoch, step, 'loss:', float(loss))

	train_loss_results.append(loss_all/3)
	loss_all=0

# test(做测试）
    total_correct, total_number = 0, 0
    for step,(x_train, y_train) in enumerate(test_db):

        h1 = tf.matmul(x_train, w1) + b1
        h1 = tf.nn.relu(h1)
#       h1 = tf.nn.sigmoid(h1)  #sigmoid函数
#       h1 = tf.nn.tanh(h1)  #tanh函数
        y = tf.matmul(h1, w2) + b2

        pred=tf.argmax(y, axis=1)  #返回张量沿指定维度最大值的索引，axis=1沿纬度


        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)
        y_train = tf.cast(y_train, dtype=tf.int32)
        correct=tf.cast(tf.equal(pred, y_train), dtype=tf.int32)
        correct=tf.reduce_sum(correct)  #计算测试集的正确数
        total_correct += int(correct)
        total_number += x_train.shape[0] #计算测试集的总数
    acc=total_correct/total_number
    print("test_acc:",acc)


# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results)
plt.show()# test(做测试）
    total_correct, total_number = 0, 0
    for step,(x_train, y_train) in enumerate(test_db):

        h1 = tf.matmul(x_train, w1) + b1
        h1 = tf.nn.relu(h1)
#       h1 = tf.nn.sigmoid(h1)  #sigmoid函数
#       h1 = tf.nn.tanh(h1)  #tanh函数
        y = tf.matmul(h1, w2) + b2

        pred=tf.argmax(y, axis=1)  #返回张量沿指定维度最大值的索引，axis=1沿纬度


        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)
        y_train = tf.cast(y_train, dtype=tf.int32)
        correct=tf.cast(tf.equal(pred, y_train), dtype=tf.int32)
        correct=tf.reduce_sum(correct)  #计算测试集的正确数
        total_correct += int(correct)
        total_number += x_train.shape[0] #计算测试集的总数
    acc=total_correct/total_number
    print("test_acc:",acc)


# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results)
plt.show()
