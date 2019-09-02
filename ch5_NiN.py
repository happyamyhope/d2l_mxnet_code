#5.8-网络中的网络NiN
#5.8.1-NiN块
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk

#5.8.2-NiN模型
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(0.5),
        #标签类别数是10
        nin_block(10, kernel_size=3, strides=1, padding=1),
        #全局平均池化层将窗口形状自动设置成输入的高和宽
        nn.GlobalAvgPool2D(),
        #将四维的输出转成二维的输出，其形状为(批量大小，10)
        nn.Flatten())

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

#5.8.3-获取数据和训练模型
print('try_gpu')
lr, num_epochs, batch_size, ctx = 0.1, 5, 50, d2l.try_gpu()  # 128
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
print("start train...\n")
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
print("end train...\n")
