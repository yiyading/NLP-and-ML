

# from sklearn import datasets
# x = datasets.load_iris().data
# print(x)
# y = datasets.load_iris().target
# print(y)

fp1 = open("iris.txt", "r")
fp2 = open("iris2.txt", "w")
for s in fp1.readlines():  # 先读出来
    fp2.write(s.replace("Iris-setosa", "0").replace("Iris-versicolor", "1").replace("Iris-virginica", "2").replace(",", " "))  # 替换 并写入
fp1.close()
fp2.close()

# import re
# path = "./iris2.txt"        # 你的文件路径
# file = open(path, encoding="utf-8")     # 读取文件
# seq = re.compile("\s+")     # 定义一个用于切割字符串的正则
#
# result = []
# # 逐行读取
# for line in file:
#     lst = seq.split(line.strip())
#     item = {
#         "name": lst[0],
#         "val": lst[1:]
#     }
#     result.append(item)
# # 关闭文件
# file.close()
# print(result)


# f1 = open("iris2.txt", "r+")
# list_row = f1.readlines()
#
# print(list_row)
# print(x)