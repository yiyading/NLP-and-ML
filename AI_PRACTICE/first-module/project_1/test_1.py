
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

# # 创建一个张量常量
# import tensorflow as tf
# a = tf.constant([[1, 2, 5], [2, 3, 4]], dtype=tf.int32)
# print(a)
# print(a.dtype)
# print(a.shape)

# # 将numpy的数据类型转换为Tensor数据类型
# import numpy as np
# import tensorflow as tf
# a = np.arange(0, 5)         # 左闭右开
# aa = tf.convert_to_tensor(a, dtype=tf.int32)
# print("a is:\n %s" %a)
# print("aa is:\n",   aa)

# import tensorflow as tf
# a = tf.zeros([2, 4])
# print(a)
# b = tf.ones(1)
# print(b)
# c = tf.fill([2, 3], 5)          # 写维度并填充数据
# print(c)

# # 正态分布
# import tensorflow as tf
# a = tf.random.normal([2, 3], mean=0, stddev=1)          # 正态分布
# print(a)
# b = tf.random.truncated_normal([2, 3], mean=0, stddev=1)        # 截断式正态分布
# print(b)

# # 均匀分布
# import tensorflow as tf
# a = tf.random.uniform([3, 3], minval=0, maxval=1)
# print(a)

# # 其他常用函数
# import tensorflow as tf
# a = tf.constant([1, 2, 3], dtype=tf.float32)
# print(a)
# x2 = tf.cast(a, dtype=tf.int32)
# print(x2)
# x2_min = tf.reduce_min(x2)
# x2_max = tf.reduce_max(x2)
# print("x2_min is %s\nx2_max is %s" %(x2_min, x2_max))

# # 求tensor的平均值
# import tensorflow as tf
# x1 = tf.constant([[2, 3, 4], [4, 5, 7]], dtype=tf.float32)
# print(x1)
# print(tf.reduce_mean(x1))
# print(tf.reduce_mean(x1, axis=1))       # 1表示求行（纬度）
# print(tf.reduce_mean(x1, axis=0))       # 0表示求列（经度）

# # tf.Variable()将数据标记为“可训练”
# import tensorflow as tf
# w = tf.Variable(tf.random.normal([2, 3], mean=0, stddev=1))
# print(w)

# # tensor的四则运算是对应位置的运算
# import tensorflow as tf
# x1 = tf.constant([2, 3, 4, 5], dtype=tf.int32)
# x2 = tf.constant([3, 5, 1, 5], dtype=tf.int32)
# print(tf.add(x1, x2))
# print(tf.subtract(x1, x2))
# print(tf.multiply(x1, x2))
# print(tf.divide(x1, x2))

# # 平方、次方、开方
# import tensorflow as tf
# a = tf.fill([2, 4], 4.)
# print(a)
# print(tf.square(a))         # 平方
# print("\n")
# print(tf.pow(a, 2))         # 平方
# print("\n")
# print(tf.sqrt(a))           # 开方

# # 矩阵乘
# import tensorflow as tf
# a = tf.ones([2, 3])
# print(a)
# b = tf.fill([3, 2], 3.)
# c = tf.matmul(a, b)
# print(c)
# # print("a matmul b is:\n%s" %c)

# 生成输入特征/标签对
import tensorflow as tf
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 2, 3])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)

# # tf.GradientTape
# import tensorflow as tf
# with tf.GradientTape() as tape:
#     x = tf.Variable(tf.constant(3.0))
#     y = tf.pow(x, 2)
# grad = tape.gradient(y, x)
# print(grad)

# # enumerate将可遍历的数据对象组合成为索引和元素
# seq = ['one', 'two', 'three']
# for i,element in enumerate(seq):
#     print(i, element)

# # 独热码tf.one_hot：1表示是，0表示非
# import tensorflow as tf
# # classes = 3
# # labels = tf.constant([1, 0, 2])
# # output = tf.one_hot(labels, depth=classes)
# # print(output)
# classes = 3
# labels = tf.constant([1, 4, 2])             # 输入的元素值4超出depth-1
# output = tf.one_hot(labels, depth=classes)  # depth给出列数，待转换元素的个数给出行数
# print(output)

# # tf.assign_sub赋值操作，更新参数的值并返回
# # 在调用assign_sub方法前，一定要先用tf.Variable对变量x初始化。
# # 直接调用tf.assign_sub会报错，必须要用x.assign_sub的格式。
# import tensorflow as tf
# x = tf.Variable(4)
# x.assign_sub(1)
# print(x)

# # 返回张量沿指定维度最大值的索引
# import tensorflow as tf
# import numpy as np
# test = np.array([[1, 2, 3], [2 3, 4], [5, 4, 3], [8, 7, 2]])
# print(test)
# print("\n")
# print(tf.argmax(test, axis=0))          # 返回每列（经度）最大值的索引
# print(tf.argmax(test, axis=1))          # 返回每行（纬度）最大值的索引


# from sklearn import datasets
# x = datasets.load_iris().data
# y = x[:-1]
# print(x)
# print("y:\n", y)


# import numpy as np
# np.random.seed(1)
# x = np.random.randn(2, 3)
# print(x)
# np.random.seed()
# y = np.random.randn(2, 3)
# print(y)