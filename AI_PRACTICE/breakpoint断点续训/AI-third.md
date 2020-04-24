# 断点续训
实现网络模型的断点续训功能

保存、恢复模型
```py
保存模型
save_weights(路径文件名,
	     overwrite=是否重写, #默认True
	     save_format=存储格式)

save_format：
	‘h5’ : HDF5格式，保存模型结构及参数
	‘tf’ : TensorFlow格式，保存参数

恢复模型
load_weights(路径文件名）
``` 

提取可训练参数
```py
model.trainable_variables	# 模型中可训练的参数

np.set_printoptions(precision=小数点后留几位,
		    threshold=超过多少位省略)

np.set_printoptions(threshold=np.inf)	# np.inf表示无穷大，
```

实例说明断点续训
```py
#!/usr/bin/env python
# coding:utf-8
# author:yiyading

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置显示模型参数的全部内容
np.set_printoptions(threshild=np.inf)
# 设置模型路径
model_save_path = './checkpoint/mnist.tf'
# 是否续训
load_pretrain_model = False

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['sparse_categorical_accuracy'])

# 恢复模型
if load_pretrain_model:
    print('---------load the model----------')
    model.load_weights(model_save_path)

# 每10个epochs保存一次模型
for i in range(50):
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
    model.save_weights(model_save_path, save_format='tf')

model.summary()

# 打印模型参数
print(model.trainable_variables)
```
