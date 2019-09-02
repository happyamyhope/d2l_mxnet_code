#4.4 自定义层
#4.4.1 不含模型参数的自定义层
from mxnet import gluon, nd
from mxnet.gluon import nn
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))

net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())

net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()

#4.4.2 含模型参数的自定义层
params = gluon.ParameterDict()
params.get('param2', shape=(2,3))
params

#实现一个含权重参数和偏差参数的全连接层
class MyDense(nn.Block):
    #units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bais = self.params.get('bais', shape=(units, ))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bais.data()
        return nd.relu(linear)
#实例化MyDense类并访问模型参数
dense = MyDense(units=3, in_units=5)
dense.params

#直接使用自定义层做前向计算
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))

#直接使用自定义层构造模型
net = nn.Sequential()
net.add(MyDense(8, in_units=64), MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))




