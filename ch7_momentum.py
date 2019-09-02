#7.4-动量法
#7.4.1-梯度下降的问题
#%matplotlib inline
import d2lzh as d2l
from mxnet import nd

eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1-eta*0.2*x1, x2-eta*4*x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

#学习率调大，自变量在竖直方向上不断越过最优解并逐渐发散
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

#7.4.2-动量法
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4* x2
    return x1-v1, x2-v2, v1, v2

eta, gamma = 0.4, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

eta =0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
#指数加权移动平均-推导过程
#由指数加权移动平均理解动量法
#7.4.3-从零开始实现
features, labels = d2l.get_data_ch7()
def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum']*v + hyperparams['lr'] * p.grad
        p[:] -= v

d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr':0.02, 'momentum':0.5}, features, labels)
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr':0.02, 'momentum':0.9}, features, labels)
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr':0.004, 'momentum':0.5}, features, labels)
#7.4.4-简洁实现
d2l.train_gluon_ch7('sgd', {'learning_rate':0.004, 'momentum':0.9}, features, labels)

