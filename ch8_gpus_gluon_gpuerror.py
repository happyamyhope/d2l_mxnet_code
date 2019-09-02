#8.5-多GPU计算的简洁实现
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
import time

#8.5.1-多GPU上初始化模型参数
#与ResNet-18构造稍有不同
def resnet18(num_classes):
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk
    
    net = nn.Sequential()
    #这里使用了较小的卷积核、步幅和填充，并去掉了最大池化层
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D, nn.Dense(num_classes))
    return net

net = resnet18(10)

#ctx = [mx.gpu(0), mx.gpu(1)]
ctx = [mx.gpu(0)]
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)

x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctx)
#net(gpu_x[0]), net(gpu_x[1])
net(gpu_x[0])

#相同的模型参数在不同显卡上的显存上的值是一样的。
weight = net[0].params.get('weight')
try:
    weight.data()
except RuntimeError:
    print('not initialized on', mx.cpu())
weight.data(ctx[0])[0], weight.data(ctx[1])[0]

#8.5.2-多GPU训练模型
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.spilt_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y)
                      for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        train_time = time.time() - start
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time %.1f sec, test_acc %.2f' % (epoch+1, train_time, test_acc))
print("start train...\n")
train(num_gpus=1, batch_size=50, lr=0.1)  # 256
print("end train...\n")
# train(num_gpus=2, batch_size=256, lr=0.1)
