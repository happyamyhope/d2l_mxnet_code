#8.3-自动并行计算
import d2lzh as d2l
import mxnet as mx
from mxnet import nd
#8.3.1-CPU和GPU的并行计算
def run(x):
    return [nd.dot(x,x) for _ in range(10)]
#分别在内存和显存上创建NDArray
x_cpu = nd.random.uniform(shape=(2000, 2000))
print('x_gpu')
x_gpu = nd.random.uniform(shape=(6000, 6000), ctx=mx.gpu(0))
print('dayin')
#打印
run(x_cpu)#预热开始
run(x_gpu)
nd.waitall() #预热结束

with d2l.Benchmark('Run on CPU.'):
    run(x_cpu)
    nd.waitall()

with d2l.Benchmark('Then run on GPU.'):
    run(x_gpu)
    nd.waitall()

#自动并行不同任务
with d2l.Benchmark('Run on both CPU and GPU in parallel.'):
    run(x_cpu)
    run(x_gpu)
    nd.waitall()

#计算和通信的并行计算
def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU.'):
    y = run(x_gpu)
    nd.waitall()

with d2l.Benchmark('Then copy to CPU.'):
    copy_to_cpu(y)
    nd.waitall()

with d2l.Benchmark('Run and copy in parallel.'):
    y = run(x_gpu)
    copy_to_cpu(y)
    nd.waitall()
