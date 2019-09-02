#9.2-微调
#应用迁移学习微调模型参数
#当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力
#9.2.1-热狗识别
#%matplotlib inline
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
from mxnet.gluon import utils as gutils
import os 
import zipfile

#获取数据集
data_dir = '../data'
base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
fname = gutils.download(
    base_url + 'gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
with zipfile.ZipFile(fname, 'r') as z:
    z.extractall(data_dir)

#创建两个ImageFolderDataset实例分别读取训练数据及和测试数据集中的所有图像文件。
train_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog/train'))
test_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog/test'))

hotdogs = [train_imgs[i][0] for i in range(8)]#前8张正类图像
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]#最后8张负类图像
d2l.show_images(hotdogs+not_hotdogs, 2, 8, scale=1.4);

#标准化
#指定RGB三个通道的均值和方差来将图像通道归一化
normalize = gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    normalize])
test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize])

#定义和初始化模型
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.output

finetune_net = model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
#output中的模型参数将在迭代中使用10倍大的学习率
finetune_net.output.collect_params().setattr('lr_mult', 10)
#微调模型
#定义使用微调的训练函数train_fine_tuning以便多次调用
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gdata.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    ctx = d2l.try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
        {'learning_rate':learning_rate, 'wd':0.001})
    d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
print('try_gpu')
print("start finetune train...\n")
train_fine_tuning(finetune_net, 0.01, 50, 2)
print("end train...\n")

#将所有模型参数都初始化为随机值，从头训练，且使用较大的学习率。
scratch_net = model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
print("start scratch train...\n")
train_fine_tuning(scratch_net, 0.1, 30, 2)
print("end train...\n")
