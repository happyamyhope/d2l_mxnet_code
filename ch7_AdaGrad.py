#7.5-AdaGrad算法
#AdaGrad算法根据自变量在每个维度的梯度值大小来调整各个维度上的学习率，
#从而避免统一的学习率难以适应所欲维度的问题。
#7.5.1-算法
#7.5.2-特点
#%matplotlib inline
import d2lzh as d2l
import math
from mxnet import nd

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2*x1, 4*x2, 1e-6 #前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1+eps) * g1
    x2 -= eta / math.sqrt(s2+eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

#7.5.3-从零开始实现
features, labels = d2l.get_data_ch7()
def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s+eps).sqrt()

d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
#7.5.4-简洁实现
d2l.train_gluon_ch7('adagrad', {'learning_rate': 0.1}, features, labels)


