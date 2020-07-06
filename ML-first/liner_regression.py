# # 1. J(θ) = 1/mΣ[(h(θ) - y)^2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集
path = 'data1.txt'
# header默认第0行作为表头，如果数据中没有表头，设置header=None
data = pd.read_csv(path, header=None, names=['Interest', 'Price'])
print(data.head(), '\n')
print(data.describe())

# 输出数据图
data.plot(kind='scatter', x='Interest', y='Price', figsize=(12,8))
plt.show()

# 创建J(θ)
def computCost(X, y, theta):
	tmp = np.power(((X * theta.T) - y) ,2)	
	return np.sum(tmp) / (2 * len(X))	

# 加入x0
data.insert(0, 'Ones', 1)

cols=data.shape[1]
print('cols:' ,cols)

X = data.iloc[ :, 0:cols-1]
y = data.iloc[ :, cols-1:]

print(X.head())
print(y.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
theta =  np.matrix(np.array([0,0]))

print('theta.init:', theta)	# out [[0,0]]
print('X.shape:', X.shape, '\ny.shape:', y.shape, '\ntheta.shape:', theta.shape)

print('J(θ):', computCost(X, y, theta))

# # 2. gradient decent
def gradientDescent(X, y, theta, alpha, iters):
	temp = np.matrix(np.zeros(theta.shape))
	parameters = int(theta.ravel().shape[1])
	cost = np.zeros(iters)

	for i in range(iters):
		error = (X * theta.T) - y

		for j in range(parameters):
			term = np.multiply(error, X[ :, j])
			temp[0, j] = theta[0, j] -((alpha /len(X)) * np.sum(term))
		
		theta = temp
		cost[i] = computCost(X, y, theta)
	return theta, cost

alpha = 0.01
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)
print('\ng:', g, '\n')

# 使用你和参数计算训练模型的代价函数（误差）
print('\nCost-output:', computCost(X, y, g))

# 绘制线性模型以及数据，观察拟合效果
x = np.linspace(data.Interest.min(), data.Interest.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fix, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', lable='Interest')
ax.scatter(data.Interest, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Interest')
ax.set_ylabel('Price')
ax.set_title('Predicted Price vs. Interest Size')
plt.show()

# 输出损失函数的下降过程
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
