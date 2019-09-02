#5.4-池化层
#5.4.1-二维最大池化层和平均池化层
from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0]-p_h+1, X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))

pool2d(X, (2, 2), 'avg')

#5.4.2-填充和步幅
X = nd.arange(16).reshape((1, 1, 4, 4))
X

#默认情况下，MaxPool2D实例中步幅和池化窗口形状相同
pool2d = nn.MaxPool2D(3)
pool2d(X)
#手动指定步幅和填充
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
#指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)

#5.4.3-多通道
X = nd.concat(X, X+1, dim=1)
X
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
print('end...')


