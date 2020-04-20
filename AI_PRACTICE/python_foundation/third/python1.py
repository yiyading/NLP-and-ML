from PIL import Image
import numpy as np

# 从mnist.txt中读取图片名，并拼接处mnist_data_jpg的路径
f = open("./mnist.txt", "r")
image_path = "./mnist_data_jpg/"
contents = f.readlines()
print(contents)

# 逐个读取图片并二值化
for content in contents:
    value = content.split()             # 从空格处隔开
    img_path = image_path + value[0]
    print(img_path)
    # 读取mnist_data_jpg中的图片，并作处理
    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)         # 设定大小和质量
    img = np.array(img.convert('L'))                    # 灰度处理
    img_arr = img
    for i in range(28):                                 # 二值化
        for j in range(28):
            if img_arr[i][j] > 230:
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 255
    im = Image.fromarray(img_arr)                   # 将array转换成Image
    im.save('./输出结果/%s.jpg' %content.split())
