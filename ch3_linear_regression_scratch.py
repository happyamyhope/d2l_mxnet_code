#线性回归的从零开始实现
#%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1]*features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
print(features[0])
print(labels[0])

def use_svg_display():
    #用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    #设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);#加分号只显示图

#读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i+batch_size, num_examples)])
        yield features.take(j), labels.take(j) #take函数根据索引返回对应元素

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

#初始化模型参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1, ))
w.attach_grad()
b.attach_grad()

#定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b

#定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
#定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
#训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs): #训练模型需要num_epochs个迭代周期
    #在每个迭代周期中会使用训练数据集中所有样本一次(假设样本数能够被批量大小整除)
    #X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y) #l是有关小批量X和y的损失
        l.backward()#小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)#使用小批量随机梯度下降迭代模型参数
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch+1, train_l.mean().asnumpy()))

print('true_w, w')
print(true_w, w)

print('true_b, b')
print(true_b, b)



