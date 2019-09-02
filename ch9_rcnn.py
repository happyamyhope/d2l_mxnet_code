#9.8-区域卷积神经网络系列RCNN
#9.8.1-RCNN
#9.8.2-Fast RCNN
#兴趣区域池化层
from mxnet import nd
X = nd.arange(16).reshape((1, 1, 4, 4))
X

rois = nd.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
nd.ROIPooling(X, rois, pooled_size = (2, 2), spatial_scale = 0.1)
print('fastRCNN')
#9.8.3-Faster RCNN

#9.8.4-Mask RCNN

