import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.scatter(x, y, color)  画散点图
# plt.show()                显示以上工作

df = pd.read_csv('./dot.csv')       # 读取当前目录下dot.csv文件
# df[]括号内的元素用双引号和单引号都可以
x1 = df['x1']                       # 将x1索引到编号和x1
x2 = df["x2"]
y_c = df["y_c"]
print("x1:\n", x1, "\nx2:\n", x2, "\ny_c:\n", y_c)

Y_c = [['red' if y else 'blue'] for y in y_c]
# np.squeeze(array)降低矩阵array的一个维度，即去除[[]]的内框，但是[[元素]]内框中只能包含一个元素
print("Y_c:\n", Y_c, "\nnp.squeeze(Y_c):\n", np.squeeze(Y_c))
plt.scatter(x1, x2, color=np.squeeze(Y_c))                  # plt.scatter()画出数据集中第一列x1和第二列x2所对应的点和颜色

# 以上代码实现带颜色的散点图，下边代码的作用是画等高线
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]               # 逗号之前定义的是行的范围，逗号之后定义列的范围——都是前闭后开；绘制网格
probs = pd.read_csv('./probs.csv')
probc = probs.values

# plt.contour()第三个参数使用probs.values和probs在图形上似乎差别不大；如果不加第三个参数，得到的图形是一条直线
# 等高线——三位图像在二维空间的显示
# 把坐标xx，yy和对应的值probs放入contour()函数，给probs中值为0.5的所有点上色，plt.show()后就显示为红蓝点的分界线
# xx和yy对应的是网格
plt.contour(xx, yy, probc, levels=[.5])
plt.show()
