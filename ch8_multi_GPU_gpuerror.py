#8.4-多GPU计算
#8.4.1-数据并行
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

#8.4.2-定义模型
#初始化模型参数
scale = 0.01
W1 = nd.random.normal(scale=scale, shape=(20, 1, 3, 3))
b1 = nd.zeros(shape=20)
W2 = nd.random.normal(scale=scale, shape=(50, 20, 5, 5))
b2 = nd.zeros(shape=50)
W3 = nd.random.normal(scale=scale, shape=(800, 128))
b3 = nd.zeros(shape=128)
W4 = nd.random.normal(scale=scale, shape=(128, 10))
b4 = nd.zeros(shape=10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]
#定义模型
def lenet(X, params):
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1],
                             Kernel=(3, 3), num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_tpe='avg', kernel=(2, 2),
                    strides=(2, 2))
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3],
                             kernel=(5, 5), num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                    stride=(2, 2))
    h2 = nd.flatten(h2)
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = nd.relu(h3_linear)
    y_hat = nd.dot(h3, params[6]) + params[7]
    return y_hat

#交叉熵损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

#8.4.3-多GPU之间同步数据
def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params

#尝试把模型参数复制到gpu(0)上
new_params = get_params(params, mx.gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)

#allreduce函数将各块显卡的显存上的数据加起来，然后广播到所有的显存上。
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])

#测试allreduce函数
data = [nd.ones((1, 2), ctx=mx.gpu(i)) * (i+1) for i in range(2)]
print('before allreduce:', data)
allreduce(data)
print('after allreduce:', data)

#split_and_load函数将数据样本划分并复制到各块显卡的显存上
def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k #简单起见，假设可以整除
    assert m*k == n, '# examples is not divides by # devices.'
    return [data[i*m:(i+1)*m].as_in_context(ctx[i]) for i in range(k)]

#使用split_and_load函数将6个数据样本平均分给2块显卡的显存。
batch = nd.arange(24).reshape((6, 4))
ctx = [mx.gpu(0), mx.gpu(1)]
splitted = split_and_load(batch, ctx)
print('input: ', batch)
print('load into', ctx)
print('output:', splitted)

#8.4.4-单个小批量上的多GPU训练
def train_batch(X, y, gpu_params, ctx, lr):
    #当ctx包含多块GPU及相应的显存时，将小批量数据样本划分并复制到各个显存上
    gpu_Xs, gpu_ys = split_and_load(X, ctx), split_and_load(y, ctx)
    with autograd.record(): #在各块GPU上分别计算损失
        ls = [loss(lenet(gpr_X, gpu_W), gpu_y)
              for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params)]
    for l in ls: #在各块GPU上分别反向传播
        l.backward()
    #把各块显卡的显存上的梯度加起来，然后广播到所有显存上
    for i in range(len(gpu_params[0])):
        allreduce(gpu_params[c][i].grad for c in range(len(ctx)))
    for param in gpu_params: #在各块显卡的显存上分别更新模型参数
        d2l.sgd(param, lr, X.shape[0]) #这里使用了完整批量大小
#8.4.5-定义训练函数
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('runing on:', ctx)
    #将模型参数复制到num_gpus块显卡的显存上
    gpu_params = [get_params(params, c) for c in ctx]
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            #对单个小批量进行多GPU训练
            train_batch(X, y, gpu_params, ctx, lr)
            nd.waitall()
        train_time = time.time() - start
        
        def net(x): #在gpu(0)上验证模型
            return lenet(x, gpu_params[0])
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time %.1f sec, test_acc %.2f' % (epoch+1, train_time, test_acc))
#8.4.6-多GPU训练实验
print("start train...\n")
train(num_gpus=1, batch_size=50, lr=0.2)  # 256
print("end train...\n")
# train(num_gpus=2, batch_size=256, lr=0.2)
