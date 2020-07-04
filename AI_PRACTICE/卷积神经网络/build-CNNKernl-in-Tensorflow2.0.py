tf.keras.layers.Conv2D(
        input_shape=(高, 宽, 通道数),   # 仅在第一层有
        filters=卷积核个数,
        kernel_size=卷积核尺寸,
        strides=卷积核步长,
        padding='valid' or 'same',
        activation='relu' or 'sigmoid' or 'tanh' or 'softmax',  # 如果有BN此层不写
)

# 在实际的网路搭建中，神经网络会有很多层，因此会采用tf.keras.models.Sequential函数进行搭建
model = tf.keras.models.Sequential([
        Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(5,5), padding='sa    me', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(512, activation='relu',
        Dense(10, activation='softmax',
])

