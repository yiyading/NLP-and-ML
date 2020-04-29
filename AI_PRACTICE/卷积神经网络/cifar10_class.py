# 利用class结构训练并测试cifar10

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout, Flatten,Dense
from tensorflow.keras import Model

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train=tf.convert_to_tensor(x_train)
x_test=tf.convert_to_tensor(x_test)
y_train=tf.squeeze(y_train, axis=1)
y_test=tf.squeeze(y_test, axis=1)


class Cifar10_Model(Model):
    def __init__(self):
        super(Cifar10_Model, self).__init__()
        self.c1 = Conv2D(input_shape=(32, 32, 3), filters=32,kernel_size=(5,5),padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.m1 = MaxPool2D(pool_size=(2, 2),strides=2,padding='same')  # 池化层
        self.drop1=Dropout(0.2)  # dropout层

        self.c2 = Conv2D(64, kernel_size=(5,5), padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.m2 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')
        self.drop2 = Dropout(0.2)

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.drop3 = Dropout(0.2)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x=self.c1(x)
        x=self.b1(x)
        x=self.a1(x)
        x=self.m1(x)
        x=self.drop1(x)

        x=self.c2(x)
        x=self.b2(x)
        x=self.a2(x)
        x=self.m2(x)
        x=self.drop2(x)

        x=self.flatten(x)
        x=self.d1(x)
        x=self.drop3(x)
        y=self.d2(x)
        return y


model = Cifar10_Model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test), validation_freq=2)
model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

