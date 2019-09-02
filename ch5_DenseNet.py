#5.12-稠密连接网络DenseNet
#5.12.1-稠密块
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)#在通道维上将输入和输出连结
        return X

blk = DenseBlock(2, 10)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 8, 8))
Y = blk(X)
Y.shape

#5.12.2-过渡层
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1), 
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk

blk = transition_block(10)
blk.initialize()
blk(Y).shape

#5.12.3-DenseNet模型
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

num_channels, growth_rate = 64, 32 #num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]
for i, num_convs in enumerate(num_convs_in_dense_blocks): #这里i表示什么意思
    net.add(DenseBlock(num_convs, growth_rate))
    #上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    #在稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks)-1:
        net.add(transition_block(num_channels // 2))

net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(), nn.Dense(10))

#5.12.4-获取数据并训练模型
print('train...')
lr, num_epochs, batch_size, ctx = 0.1, 5, 50, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
print("start train...")
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
print("start end...")

