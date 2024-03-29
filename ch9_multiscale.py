#9.5-多尺度目标检测
#%matplotlib inline
import d2lzh as d2l
from mxnet import contrib, image, nd

img = image.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w

d2l.set_figsize()
def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h)) #前两维的取值不影响输出结果
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])

display_anchors(fmap_w=2, fmap_h=2, s=[0.4])

display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

##########################################################################################
#9.6-目标检测数据集(皮卡丘)
#9.6.1-下载数据集
#%matplotlib inline
import d2lzh as d2l
from mxnet import gluon, image
from mxnet.gluon import utils as gutils
import os

def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url+k, os.path.join(data_dir, k), sha1_hash=v)

#9.6.2-读取数据集
def load_data_pikachu(batch_size, edge_size=256): #edge_size:输出图像的宽和高
    data_dir = '../data/pikachu'
    _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
            path_imgrec = os.path.join(data_dir, 'train.rec'),
            path_imgidx = os.path.join(data_dir, 'train.idx'),
            batch_size = batch_size,
            data_shape = (3, edge_size, edge_size), #输出图像的形状
            shuffle = True, #以随机顺序读取数据集
            rand_crop = 1, #随机裁剪的概率为1
            min_object_covered = 0.95, max_attempts = 200)
    val_iter = image.ImageDetIter(
            path_imgrec = os.path.join(data_dir, 'val.rec'), batch_size = batch_size, 
            data_shape = (3, edge_size, edge_size), shuffle = False)
    return train_iter, val_iter

batch_size, edge_size = 32, 256
train_iter, _ = load_data_pikachu(batch_size, edge_size)
batch = train_iter.next()
batch.data[0].shape, batch.label[0].shape

#9.6.3-图示数据
imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255
axes = d2l.show_images(imgs, 2, 5).flatten()
for ax, label in zip(axes, batch.label[0][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors =['w'])

