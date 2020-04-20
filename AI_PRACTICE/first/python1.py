import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取当前目录下dot.csv文件
df = pd.read_csv('./dot.csv')      

# 将dot.csv中x1列索引到编号x1
x1 = df['x1']                      
x2 = df["x2"]
y_c = df["y_c"]
print("x1:\n", x1, "\nx2:\n", x2, "\ny_c:\n", y_c)

# y_c=1，给他'red'字符串
Y_c = [['red' if y else 'blue'] for y in y_c]

# np.squeeze(x)将x降维，plt.scatter(x,y,color)画出x,y的散点图，并标出颜色
print("Y_c:\n", Y_c, "\nnp.squeeze(Y_c):\n", np.squeeze(Y_c))
plt.scatter(x1, x2, color=np.squeeze(Y_c))

# 画等高线
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]   
probs = pd.read_csv('./probs.csv')
probc = probs.values

# 把坐标xx，yy和对应的值probs放入contour()函数，给probs中值为0.5的所有点上色，plt.show()后就显示为红蓝点的分界线
plt.contour(xx, yy, probc, .8)
plt.show()
