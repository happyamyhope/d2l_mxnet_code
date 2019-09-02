#3.8.1
#%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')

x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')

y.backward()
xyplot(x, x.grad, 'grad of relu')

with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')

y.backward()
xyplot(x, x.grad, 'grad of sigmoid')

with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')

y.backward()
xyplot(x, x.grad, 'grad of tanh')


