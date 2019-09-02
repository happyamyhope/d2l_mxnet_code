#多层感知机的从零开始实现
#%matplotlib inline
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss
#3.9.1 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#3.9.2 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

#3.9.3 定义激活函数
def relu(X):
    return nd.maximum(X, 0)
#3.9.4 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1)+b1)
    return nd.dot(H, W2) + b2
#3.9.5 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

#3.9.6 训练模型 
num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


