import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("SH_600519_high_low.csv")
x = np.arange(0, 47)
y1 = df["high"]
y2 = df["low"]
print("x:\n", x, "\ny1:\n", y1, "\ny2:\n", y2)

plt.xlabel("day")
plt.ylabel("price")
plt.title("Kweichow Moutai")
plt.plot(x, y1, label="high", color="blue")
plt.plot(x, y2, label="low", color="orange")

plt.legend()	# 给图像加上图例
plt.show()
