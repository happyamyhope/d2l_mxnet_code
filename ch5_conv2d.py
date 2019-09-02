#5.1-二维卷积层
#5.1.1-二维互相关运算
from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
corr2d(X, K)

#5.1.2-二维卷积层
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bais = self.params.get('bias', shape=(1, ))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

#5.1.3-图像中物体边缘检测
X = nd.ones((6, 8))
X[:, 2:6] = 0
X

K = nd.array([[1, -1]])
Y = corr2d(X, K)
Y

#5.1.4-通过数据学习核数组
#构造一个输出通道数为1，核数组形状为(1, 2)的二维卷积层
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()
#二维卷积层使用4维输入输出，格式为(样本，通道，高，宽)，这里批量大小和通道数均为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y)**2
    l.backward()
    #简单起见这里忽略了偏差
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i+1) % 2 == 0:
        print('batch %d, loss %.3f' % (i+1, l.sum().asscalar()))

conv2d.weight.data().reshape((1, 2))
#5.1.5-互相关运算和卷积运算
#为了与大多数深度学习文献一致，如无特别说明，本书中提到的卷积运算均指互相关运算。
#5.1.6-特征图和感受野

