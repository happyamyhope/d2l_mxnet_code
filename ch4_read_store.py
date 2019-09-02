#4.5 读取和存储
#4.5.1读写NDArray
from mxnet import nd
from mxnet.gluon import nn
x = nd.ones(3)
nd.save('x', x)
x2 = nd.load('x')
x2

y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
(x2, y2)

mydict = {'x': x, 'y':y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
mydict2

#4.5.2 读写Gluon模型参数
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)

#将模型参数存成文件
filename = 'mlp.params'
net.save_parameters(filename)

net2 = MLP()
net2.load_parameters(filename)

Y2 = net2(X)
Y2 == Y

