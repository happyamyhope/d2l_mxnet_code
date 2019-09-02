#线性回归
#线性回归的基本要素
#模型
#模型训练
#训练数据
#损失函数
#优化算法
#模型预测
#线性回归的表示方法
#神经网络图
#矢量计算表达式
from mxnet import nd
from time import time

a = nd.ones(shape=1000)
b = nd.ones(shape=1000)
#向量相加
#按元素逐一做标量加法
start = time()
c = nd.zeros(shape=1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)
#矢量加法
start = time()
d = a + b
print(time() - start)
#结论：尽量采用矢量计算以提升计算效率。

a = nd.ones(shape=3)
b = 10
a + b

