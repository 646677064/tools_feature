#coding:utf-8
import numpy as np
import sys
caffe_root = '/home/liushuai/tiannuocaffe/py-rfcn-gpu/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# bot_data_root = 'F:/bot_data'


# net_file = bot_data_root + '/myVGG16/VGG_ILSVRC_16_layers_deploy.prototxt'

# caffe_model = bot_data_root + '/myVGG16/myvggmodel__iter_80000.caffemodel'

# mean_file = bot_data_root + '/myVGG16/mean.npy'

bot_data_root = '/storage2/for_gs4/compare/'


net_file = bot_data_root + 'res101_dropout_calssfy.prototxt'

caffe_model = bot_data_root + 'res101_dropout_calssfy.caffemodel'

#mean_file = bot_data_root + '/myVGG16/mean.npy'
gpu=6

if gpu >= 0:
  caffe.set_mode_gpu()
  caffe.set_device(gpu)
  print("GPU mode, device : {}".format(gpu))
else:
  caffe.set_mode_cpu()
  print("CPU mode")

# 构造一个Net
net = caffe.Net(net_file, caffe_model, caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB
# caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换

# channel 放到前面
transformer.set_transpose('data', (2, 0, 1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_mean('data', np.array([110.676,115.771,123.191]))
# 图片像素放大到[0-255]
transformer.set_raw_scale('data', 255)
# RGB-->BGR 转换
transformer.set_channel_swap('data', (2, 1, 0))
#设置输入的图片shape，1张，3通道，长宽都是224
net.blobs['data'].reshape(1, 3, 224, 224)
# 加载图片
im = caffe.io.load_image('/storage2/liushuai/data/similary_data/new_trainsimilary/a_f_k/alps4_back/33_alps4_back3313.jpg')

# 用上面的transformer.preprocess来处理刚刚加载图片
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#输出每层网络的name和shape
# for layer_name, blob in net.blobs.iteritems():
#     print layer_name + '\t' + str(blob.data.shape)

# 网络开始向前传播啦
output = net.forward()

# 找出最大的那个概率
#output_prob = output['out'][0]
output_prob = output['prob'][0]
print output_prob.shape
print 'max prob:', output_prob.argmax()

# 找出最可能的前俩名的类别和概率
top_inds = output_prob.argsort()[::-1][:2]
print "top2: ",top_inds
print "top2 probability: ", output_prob[top_inds[0]], output_prob[top_inds[1]]