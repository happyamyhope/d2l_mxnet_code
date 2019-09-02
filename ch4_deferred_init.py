#模型参数的延后初始化
from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        #实际的初始化逻辑在此省略了
    
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))

net.initialize(init=MyInit())

X = nd.random.uniform(shape=(2, 20))
Y = net(X)

#Init dense0_weight (256, 20)
#Init dense1_weight (10, 256)

Y = net(X)

#避免延后初始化
#1-对已初始化的模型重新初始化
net.initialize(init=MyInit(), force_reinit=True)
#Init dense0_weight (256, 20)
#Init dense1_weight (10, 256)

#2-创建层时指定输入个数
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())

#Init dense2_weight (256, 20)
#Init dense3_weight (10, 256)


