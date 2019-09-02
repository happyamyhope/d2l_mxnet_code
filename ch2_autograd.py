#自动求梯度
from mxnet import autograd, nd
x = nd.arange(4).reshape((4, 1))
#调用attach_grad函数申请存储梯度所需的内存.
x.attach_grad()
#为了减少计算和内存开销，默认条件下MXNet不会记录用于求梯度的计算，
#需要调用record函数来要求MXNet记录与求梯度有关的计算。
with autograd.record():
    y = 2 * nd.dot(x.T, x)
#通过调用backward函数自动求梯度。
#如果y不是一个标量，MXNet将默认先对y中元素求和得到新的变量，
#再求该变量有关的梯度。
y.backward()
assert(x.grad-4*x).norm().asscalar()==0
x.grad

#训练模式和预测模式
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
#对Python控制流求梯度
def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()

print(a.grad == c / a)

