#5.2-填充和步幅
#5.2.1-填充
from mxnet import nd
from mxnet.gluon import nn

#定义一个函数老计算卷积层，初始化卷积层权重，并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    #(1, 1)代表批量大小和通道数
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])#排除不关心的前两维：批量和通道

#注意这里是两侧分别填充1行或者列
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
#使用高为5、宽为3的卷积核，在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2,1))
comp_conv2d(conv2d, X).shape 

#5.2.2-步幅
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0,1), strides=(3,4))
comp_conv2d(conv2d, X).shape


#5.3-多输入通道和多输出通道
#5.3.1-多输入通道
import d2lzh as d2l
from mxnet import nd

def corr2d_multi_in(X, K):
    #首先沿着X和K的第0维遍历，然后使用*将结果列表编程add_n函数的位置参数
    #(positional argument)来进行相加
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [6, 7, 8]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
corr2d_multi_in(X, K)

#5.3.2-多输出通道
def corr2d_multi_in_out(X, K):
    #对K的第0维遍历，每次同输入X做互相关计算，所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

K = nd.stack(K, K+1, K+2)
K.shape

corr2d_multi_in_out(X, K)

#5.3.3-1*1卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X) #全连接层的矩阵乘法
    return Y.reshape((c_o, h, w))

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
(Y1-Y2).norm().asscalar() < 1e-6
print('End...')
