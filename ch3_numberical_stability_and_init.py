#数值稳定性和模型初始化
#数值稳定性的典型问题是衰减(vanishing)和爆炸(explosion)。
#衰减和爆炸
#使用恒等映射(identity mapping)解释衰减和爆炸.
#随机初始化模型参数
#解释需要随机初始化模型参数的原因.
#MXNet的默认随机初始化
#随机采样[-0.07, 0.07]之间的均匀分布
#Xavier随机初始化
#Xavier随机初始化后，每层输出的方差不该受该层输入个数影响，
#且每层梯度的方差也不该受该层输出个数影响。
