#4.6-GPU计算
#介绍如何使用单块GPU计算
#运行本节程序需要至少2块GPU
#4.6.1-计算设备
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import time

mx.cpu(), mx.gpu()#, mx.gpu(1)
#4.6.2-NDArray的GPU计算
x = nd.array([1, 2, 3])
x
x.context#查看变量所在的设备

#GPU上的存储
print('mx-gpu-1')
a = nd.array([1, 2, 3], ctx=mx.gpu())
time.sleep(10)
print('time')
B = nd.random.uniform(shape=(2, 3), ctx=mx.gpu())
print('mx-gpu-2')

#通过copyto和as_in_context函数在设备之间传输数据
y = x.copyto(mx.gpu())
z = x.as_in_context(mx.gpu())

#如果源变量和目标变量的context一致，as_in_context函数使目标变量和源变量共享源变量的内存或者显存
y.as_in_context(mx.gpu()) is y
#而copyto函数总是为目标变量开辟新的内存或者显存
y.copyto(mx.gpu()) is y

#GPU上的计算
(z+2).exp()*y
#4.6.3-Gluon的GPU计算
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())

net(y)
net[0].weight.data()
print('end...\n')






