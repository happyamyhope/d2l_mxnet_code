#5.6-深度卷积神经网络(AlexNet)
#5.6.1-学习特征表示
#缺失要素：数据和硬件
#5.6.2-AlexNet
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import os
import sys

net = nn.Sequential()
#使用较大的11x11窗口来捕获物体，同时使用步幅4来较大幅度减小输出高和宽，这里使用的输出通道数比LeNet中的要大得多
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        #减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        #连续3个卷积层，且使用更小的卷积窗口，除了最后的卷积层外，进一步增大了输出通道数
        #前两个卷积层后不使用池化层来减小输入的高和宽
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        #这里全连接层的输出个数比LeNet中的大数倍，使用丢弃层来缓解过拟合
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        #输出层，使用Fashion-Mnist数据集。
        nn.Dense(10))

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

#5.6.3-读取数据
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)#展开用户路径‘~’
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter =gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

batch_size = 50  # 128
#如果出现'out of memory'的报错信息，可以减少batch_size或者resize
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

#5.6.4-训练
lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
print("train start...\n")
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
print("end...\n")
