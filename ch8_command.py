#8.1-命令式编程和符号式混合编程
#命令式编程
def add(a, b):
    return a+b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

fancy_func(1, 2, 3, 4)

#符号式编程
def add_str():
    return '''
def add(a, b):
    return a+b
'''
def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3,4))
'''
prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)

#8.1.1-混合式编程取两者之长
#8.1.2-使用HybridSequential类构造模型
from mxnet import nd, sym
from mxnet.gluon import nn
import time

def get_net():
    net = nn.HybridSequential() #这里创建HybridSequential实例
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)

net.hybridize()
net(x)
#计算性能
def benchmark(net, x):
    start = time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall() #等待所有计算完成方便计时
    return time.time()-start
net = get_net()
print('before hybridizing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('after hybridizing: %.4f sec' % (benchmark(net, x)))
#获取符号式程序
net.export('my_mlp') #将符号式程序和模型参数保存到硬盘

x = sym.var('data')
net(x)

#8.1.3-使用HybridBlock类构造模型
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)
    def hybrid_forward(self, F, x):
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)

net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
net(x)

net.hybridize()
net(x)


