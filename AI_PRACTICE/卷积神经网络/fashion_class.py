# 利用class结构训练并测试fashion_mnist
import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout, Flatten,Dense
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_save_path = './checkpoint/fashion.tf'

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

class Fashion_Model(Model):
    def __init__(self):
        super(Fashion_Model, self).__init__()
        self.c1 = Conv2D(input_shape=(28, 28, 1), filters=32,kernel_size=(5,5),padding='same')  # 卷积层
        self.bn1 = BatchNormalization()  # BN层
        self.ac1 = Activation('relu')  # 激活层
        self.s1 = MaxPool2D(pool_size=(2, 2),strides=2,padding='same')  # 池化层
        self.drop1=Dropout(0.2)  # dropout层

        self.c2 = Conv2D(64, kernel_size=(5,5), padding='same')
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')
        self.s2 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')
        self.drop2 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.drop3 = Dropout(0.2)
        self.d1 = Dense(10, activation='softmax')

    def call(self, x):
        x=self.c1(x)
        x = self.bn1(x)
        x=self.ac1(x)
        x=self.s1(x)
        x=self.drop1(x)

        x=self.c2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x=self.s2(x)
        x = self.drop2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x=self.drop3(x)
        y = self.d1(x)
        return y

model = Fashion_Model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if os.path.exists(model_save_path+'.index'):
    print('-------------load the model-----------------')
    model.load_weights(model_save_path)

for i in range(5):
    model.fit(x_train, y_train, epochs=1,batch_size=32, validation_data=(x_test, y_test), validation_freq=2)
    model.save_weights(model_save_path, save_format='tf')

model.summary()

# file = open('./weights.txt', 'w')  # 参数提取
# for v in model.trainable_variables:
#     file.write(v.name + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()
