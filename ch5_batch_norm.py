#5.10-批量归一化
#5.10.1-批量归一化层
#对全连接层做批量归一化
#对卷积层做批量归一化
#预测时的批量归一化
#5.10.2-从零开始实现
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

#通过NDArray实现批量归一化层
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #通过autograd来判断当前模式是训练模式还是预测模式
    if not autograd.is_training():
        #如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X-moving_mean) / nd.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(axis=0)
            var = ((X-mean)**2).mean(axis=0)
        else:
            # 使用二维卷积层的情况，计算通道维上(axis=1)的均值和方差，
            # 这里需要保持X的形状以便后面可以做广播运算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X-mean)**2).mean(axis=(0, 2, 3), keepdims=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X-mean) / nd.sqrt(var+eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0-momentum)*mean
        moving_var = momentum * moving_var + (1.0-momentum)*var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var

# 自定义BatchNorm层
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else: 
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化为0和1
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta  = self.params.get('beta' , shape=shape, init=init.Zero())
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = nd.zeros(shape)
        self.moving_var  = nd.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在的显存上
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var  = self.moving_var.copyto(X.context)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
#使用批量归一化层的LeNet
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
print('gpu')
lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print("start train...\n")
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
print("start end...\n")
#查看学习到的批量归一化参数
net[1].gamma.data().reshape((-1, )), net[1].beta.data().reshape((-1, ))
#5.10.3-简洁实现
#BatchNorm类参数值通过延后初始化自动获取
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))

net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
print("start gluon train...\n")
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
print("end gluon train...\n")

