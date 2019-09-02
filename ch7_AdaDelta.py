#7.7-AdaDelta算法
#7.7.1-算法
#算法添加一个额外的状态变量-自变量的变化量
#7.7.2-从零开始实现
#%matplotlib inline
import d2lzh as d2l
from mxnet import nd

features, labels = d2l.get_data_ch7()
def init_adadelta_states():
   s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
   delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
   return ((s_w, delta_w), (s_b, delta_b))
def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta+ eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g

d2l.train_ch7(adadelta, init_adadelta_states(), {'rho':0.9}, features, labels)

#7.7.3-简洁实现
d2l.train_gluon_ch7('adadelta', {'rho': 0.9}, features, labels)

#########################################################################################
#7.8-Adam算法
#Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均
#7.8.1-算法
#7.8.2-从零开始实现
#%matplotlib inline
import d2lzh as d2l
from mxnet import nd

features, labels = d2l.get_data_ch7()
def init_adam_states():
    v_w, v_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (s_bias_corr.sqrt() + eps)
    hyperparams['t'] += 1

d2l.train_ch7(adam, init_adam_states(), {'lr':0.01, 't':1}, features, labels)

#7.8.3-简洁实现
d2l.train_gluon_ch7('adam', {'learning_rate': 0.01}, features, labels)

