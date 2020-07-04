tf.keras.layers.Dropout(舍弃概率)

Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(5, 5), padding='same'),      # 卷积层
BatchNormalization(),  # BN层
Activation('relu'),  # 激活层
MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
Dropout(0.2),  # dropout层
