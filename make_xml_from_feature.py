import numpy as np
import os
import sys
import argparse
import glob
import time
#import _init_paths
from units import SClassifier, AverageMeter, convert_secs2time
import caffe
import scipy.io as sio
import sys,os,subprocess,commands
from subprocess import Popen,PIPE
import random
import math
# from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect
# from fast_rcnn.nms_wrapper import nms
#from utils.timer import Timer
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
# import matplotlib.pyplot as plt
# import numpy as np
#import scipy.io as sio
import   cv2

  
def read_xml(in_path): 
    tree = ElementTree()
    tree.parse(in_path)
    return tree  
  
def write_xml(tree, out_path):  
    tree.write(out_path, encoding="utf-8",xml_declaration=True)  
  
def if_match(node, kv_map):  
    for key in kv_map:  
        if node.get(key) != kv_map.get(key):  
            return False  
    return True  
  
#---------------search -----  
  
def find_nodes(tree, path):  
    return tree.findall(path)  
  
  
def get_node_by_keyvalue(nodelist, kv_map):   
    result_nodes = []  
    for node in nodelist:  
        if if_match(node, kv_map):  
            result_nodes.append(node)  
    return result_nodes  
  
#---------------change -----  
  
def change_node_properties(nodelist, kv_map, is_delete=False):    
    for node in nodelist:  
        for key in kv_map:  
            if is_delete:   
                if key in node.attrib:  
                    del node.attrib[key]  
            else:  
                node.set(key, kv_map.get(key))  
              
def change_node_text(nodelist, text, is_add=False, is_delete=False):  
    for node in nodelist:  
        if is_add:  
            node.text += text  
        elif is_delete:  
            node.text = ""  
        else:  
            node.text = text  
              
def create_node(tag, property_map, content):  
    element = Element(tag, property_map)  
    element.text = content  
    return element  
          
def add_child_node(nodelist, element):  
    for node in nodelist:  
        node.append(element)  
          
def del_node_by_tagkeyvalue(nodelist, tag, kv_map):   
    for parent_node in nodelist:  
        children = parent_node.getchildren()  
        for child in children:  
            if child.tag == tag and if_match(child, kv_map):  
                parent_node.remove(child)  
                          

def parse_xml_WH(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    obj_size = tree.find('size')
    obj_struct = {}
    obj_struct['width'] = (obj_size.find('width').text)
    obj_struct['height'] = (obj_size.find('height').text)
    obj_struct['depth'] = (obj_size.find('depth').text)  

def parse_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    # tree=ElementTree()
    # tree.parse(filename)

    baseInfo={}
    #baseInfo['folder'] = tree.find('folder').text
    baseInfo['filename'] = tree.find('filename').text
    baseInfo['path'] = tree.find('path').text
    baseInfo['source/database'] = tree.find('source/database').text
    #tree.find('database')
    baseInfo['size/width'] = tree.find('size/width').text
    baseInfo['size/height'] = tree.find('size/height').text
    baseInfo['size/depth'] = tree.find('size/depth').text
    baseInfo['segmented'] = tree.find('segmented').text
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['score'] = obj.find('score').text
        obj_struct['region'] = obj.find('region').text
        obj_struct['imageptr'] = obj.find('imageptr').text
        if obj.find('label_des') is  None:
          obj_struct['label_des']=""
        else:
          obj_struct['label_des'] = obj.find('label_des').text
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = obj.find('truncated').text #remove int()
        obj_struct['difficult'] = obj.find('difficult').text #remove int()
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return baseInfo,objects

def parse_xml1(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    # tree=ElementTree()
    # tree.parse(filename)

    baseInfo={}
    baseInfo['foder'] = tree.find('foder').text
    baseInfo['filename'] = tree.find('filename').text
    baseInfo['path'] = tree.find('path').text
    baseInfo['source/database'] = tree.find('source/database').text
    #tree.find('database')
    baseInfo['size/width'] = tree.find('size/width').text
    baseInfo['size/height'] = tree.find('size/height').text
    baseInfo['size/depth'] = tree.find('size/depth').text
    baseInfo['segmented'] = tree.find('segmented').text
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['score'] = obj.find('score').text
        obj_struct['region'] = obj.find('region').text
        obj_struct['imageptr'] = obj.find('imageptr').text
        if obj.find('label_des') is  None:
          obj_struct['label_des']=""
        else:
          obj_struct['label_des'] = obj.find('label_des').text
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = obj.find('truncated').text #remove int()
        obj_struct['difficult'] = obj.find('difficult').text #remove int()
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return baseInfo,objects


def load_txt(xfile):
  img_files = []
  labels = []
  for line in open(xfile):
    line = line.strip('\n').split(' ')
    assert(len(line) == 2)
    img_files.append(line[0])
    labels.append(int(float(line[1])))
  return img_files, labels

def comp_feature(feature_1,feature_2):
  feature_1=feature_1.reshape(-1)
  feature_2=feature_2.reshape(-1)
  feature_1_mult = feature_1*feature_1
  feature_2_mult = feature_2*feature_2
  sum1=np.sqrt(sum(feature_1_mult))
  feature_1=feature_1/sum1
  sum2=np.sqrt(sum(feature_2_mult))
  feature_2=feature_2/sum2
  mult=feature_1*feature_2
  feature_1_mult = feature_1*feature_1
  feature_2_mult = feature_2*feature_2
  # print feature_1.shape
  # print feature_1_mult
  # print sum1
  # print feature_1
  ret = sum(feature_1_mult)+sum(feature_2_mult)-2*sum(mult)
  return ret

def Popen_do(pp_string,b_pip_stdout=True):
  #print pp_string
  if b_pip_stdout==True:
      p = Popen(pp_string, shell=True, stdout=PIPE, stderr=PIPE)#,close_fds=True)
  else:
      p = Popen(pp_string, shell=True, stderr=PIPE)#,close_fds=True)
  out, err = p.communicate()
  #p.wait()
  print pp_string
  if p.returncode != 0:
      print err
      #return 0
  return 1

def demo(net, image_name,num_class,save_ff):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file=image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #for zzz in range(100):
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.35
    NMS_THRESH = 0.3
    thresh=CONF_THRESH
    for cls_ind, cls in enumerate(range(num_class)):#CLASSES[1:]
        cls_ind += 1 # because we skipped background
        # cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        # cls_scores = scores[:, cls_ind]
        # dets = np.hstack((cls_boxes,
        #                   cls_scores[:, np.newaxis])).astype(np.float32)
        inds = np.where(scores[:, cls_ind] > thresh)[0]
        cls_scores = scores[inds, cls_ind]
        if cfg.TEST.AGNOSTIC:
            cls_boxes = boxes[inds, 4:8]
        else:
            cls_boxes = boxes[inds, cls_ind*4:(cls_ind+1)*4]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue

        im_tmp = im#im[:, :, (2, 1, 0)]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            print bbox,score,cls
            cv2.rectangle(im_tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255),2)
    #save_ff="/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641/test_result.jpg"
    im_tmp = im#im[:, :, (2, 1, 0)]
    cv2.imwrite(save_ff,im_tmp)
    #save_pic(im, cls, dets, thresh=CONF_THRESH,save_ff)

class Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, inputs, oversample=True):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.

        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        print inputs[0].shape
        print input_.shape
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        # if oversample:
        #     # Generate center, corner, and mirrored crops.
        #     input_ = caffe.io.oversample(input_, self.crop_dims)
        # else:
        #     # Take center crop.
        #     center = np.array(self.image_dims) / 2.0
        #     crop = np.tile(center, (1, 2))[0] + np.concatenate([
        #         -self.crop_dims / 2.0,
        #         self.crop_dims / 2.0
        #     ])
        #     crop = crop.astype(int)
        #     input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]

        # # For oversampling, average predictions across crops.
        # if oversample:
        #     predictions = predictions.reshape((len(predictions) / 10, 10, -1))
        #     predictions = predictions.mean(1)

        return predictions
    def get_blob_data(self, blob_name):
        return self.blobs[blob_name].data

def skloadimage(path_1, color=True):

  im_1 = skimage.img_as_float(skimage.io.imread(path_1, as_grey=not color)).astype(np.float32)
  print im_1.ndim
  print im_1.shape[2]
  if im_1.ndim == 2:
      im_1 = im_1[:, :, np.newaxis]
      if color:
          im_1 = np.tile(im_1, (1, 1, 3))
  elif im_1.shape[2] == 4:
      im_1 = im_1[:, :, :3]
  return im_1

def main(argv):

  parser = argparse.ArgumentParser()
  # Required arguments: input and output files.
  parser.add_argument(
    "input_file",
    help="Input image, directory"
  )
  parser.add_argument(
    "feature_file",
    help="Feature mat filename."
  )
  parser.add_argument(
    "score_file",
    help="Score Output mat filename."
  )
  # Optional arguments.
  parser.add_argument(
    "--model_def",
    default=os.path.join(
            "./models/market1501/caffenet/feature.proto"),
    help="Model definition file."
  )
  parser.add_argument(
    "--pretrained_model",
    default=os.path.join(
            "./models/market1501/caffenet/caffenet_iter_17000.caffemodel"),
    help="Trained model weights file."
  )
  parser.add_argument(
    "--gpu",
    type=int,
    default=-1,
    help="Switch for gpu computation."
  )
  parser.add_argument(
    "--center_only",
    action='store_true',
    help="Switch for prediction from center crop alone instead of " +
         "averaging predictions across crops (default)."
  )
  parser.add_argument(
    "--images_dim",
    default='256,256',
    help="Canonical 'height,width' dimensions of input images."
  )
  parser.add_argument(
    "--mean_value",
    default=os.path.join(
                         'examples/market1501/market1501_mean.binaryproto'),
    help="Data set image mean of [Channels x Height x Width] dimensions " +
         "(numpy array). Set to '' for no mean subtraction."
  )
  parser.add_argument(
    "--input_scale",
    type=float,
    help="Multiply input features by this scale to finish preprocessing."
  )
  parser.add_argument(
    "--raw_scale",
    type=float,
    default=255.0,
    help="Multiply raw input by this scale before preprocessing."
  )
  parser.add_argument(
    "--channel_swap",
    default='2,1,0',
    help="Order to permute input channels. The default converts " +
         "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
  )
  parser.add_argument(
    "--ext",
    default='jpg',
    help="Image file extension to take as input when a directory " +
         "is given as the input file."
  )
  parser.add_argument(
    "--feature_name",
    default="fc7",
    help="feature blob name."
  )
  parser.add_argument(
    "--score_name",
    default="prediction",
    help="prediction score blob name."
  )
  args = parser.parse_args()

  #======================================================================================================
  # cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  # cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
  # ## Number of top scoring boxes to keep after applying NMS to RPN proposals
  # cfg.TEST.RPN_POST_NMS_TOP_N = 2000 #lius 300
  # #cfg.TEST.RPN_POST_NMS_TOP_N = 2000 #lius 300
  # cfg.TEST.AGNOSTIC=True
  # #cfg.TEST.AGNOSTIC=False
  # cfg.TEST.RPN_MIN_SIZE=10
  # prototxt = "/home/liushuai/tiannuocaffe/py-rfcn-gpu/models/shape/ResNet-101_2/rfcn_end2end/s16_14/b14_test_16_s_4_8_16_32_agnostic.prototxt"
  # caffemodel = "/home/liushuai/tiannuocaffe/py-rfcn-gpu/output/goodsType/shapeproj2_trainval/ResNet-101_2_b14_16_s_4_8_16_32_shape_rfcn_ohem_iter_210000.caffemodel"

  # save_ff="/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641/test_resultvgg.jpg"
  # im_name="/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641//cat.jpg"
  # im_name="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj1/JPEGImages/budweiser08782.jpg"#budweiser15059.jpg"
  # num_class=2-1#1360-1 #341
  # if not os.path.isfile(caffemodel):
  #     raise IOError(('{:s} not found.\nDid you run ./data/script/'
  #                    'fetch_faster_rcnn_models.sh?').format(caffemodel))


  # caffe.set_mode_gpu()
  # caffe.set_device(5)
  # cfg.GPU_ID = 5
  # net = caffe.Net(prototxt, caffemodel, caffe.TEST)

  # print '\n\nLoaded network {:s}'.format(caffemodel)

  # # Warmup on a dummy image
  # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  # # for i in xrange(2):
  # #     _, _= im_detect(net, im)

  # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
  #             '001763.jpg', '004545.jpg']
  # # for im_name in im_names:
  # #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  # #     print 'Demo for data/demo/{}'.format(im_name)
  # demo(net, im_name,num_class,save_ff)
  #======================================================================================================
  #args.images_dim="224,224"
  image_dims = [int(s) for s in args.images_dim.split(',')]

  channel_swap = None
  if args.channel_swap:
    channel_swap = [int(s) for s in args.channel_swap.split(',')]

  mean_value = None
  if args.mean_value:
    mean_value = [float(s) for s in args.mean_value.split(',')]
    mean_value = np.array(mean_value)

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    print("GPU mode, device : {}".format(args.gpu))
  else:
    caffe.set_mode_cpu()
    print("CPU mode")

  # Make classifier
  classifier = SClassifier(args.model_def, args.pretrained_model,
        image_dims=image_dims, mean_value=mean_value,
        input_scale=args.input_scale, raw_scale=args.raw_scale,
        channel_swap=channel_swap)
  # classifier = Classifier(args.model_def, args.pretrained_model,
  #       image_dims=image_dims, mean=mean_value,
  #       input_scale=args.input_scale, raw_scale=args.raw_scale,
  #       channel_swap=channel_swap)
  dir_1="/storage2/liushuai/gs6_env/market1501_extract_freature/test/OK/"
  dir_2="/storage2/liushuai/gs6_env/market1501_extract_freature/test/NG/"
  dir_out="/storage2/liushuai/gs6_env/market1501_extract_freature/test/out/"
  save_feature_all=None
  labels_all=[]
  list_1 = os.listdir(dir_1)
  for file_1 in list_1:
    if os.path.splitext(file_1)[1] !=".xml":
      basename = os.path.splitext(file_1)[0]
      jpgname = dir_1+file_1
      xmlname= dir_1+basename+".xml"
      im = caffe.io.load_image(jpgname)#cv2.imread(jpgname)
      baseInfo,objects = parse_xml(xmlname)
      save_feature=None
      #labels=None
      for idx_f,oject_1 in enumerate(objects):
        cropImg = im[oject_1["bbox"][1]:oject_1["bbox"][3], oject_1["bbox"][0]:oject_1["bbox"][2],:]
        _ = classifier.predict([cropImg], not args.center_only)
        feature = classifier.get_blob_data(args.feature_name)
        assert (feature.shape[0] == 1 )
        feature_shape = feature.shape
        if save_feature is None:
            print('feature : {} : {}'.format(args.feature_name, feature_shape))
            save_feature = np.zeros((len(objects), feature.size),dtype=np.float32)
        feature = feature.reshape(1, feature.size)
        save_feature[idx_f, :] = feature.copy()
        labels_all.append(oject_1['name'])
        #tmp_file_name=os.path.basename(file_list[idx_f])
        #sio.savemat(dir_1+'/'+basename+".feature", {'feature':feature})
      if save_feature_all==None:
        save_feature_all=save_feature
      else:
        save_feature_all=np.concatenate((save_feature_all,save_feature),axis=0)
  print len(labels_all),len(save_feature)
  print labels_all

  list_2 = os.listdir(dir_2)
  for file_2 in list_2:
    if os.path.splitext(file_2)[1] !=".xml":
      basename_2 = os.path.splitext(file_2)[0]
      jpgname_2 = dir_2+file_2
      xmlname_2= dir_2+basename_2+".xml"
      print xmlname_2
      im = caffe.io.load_image(jpgname)#cv2.imread(jpgname_2)
      baseInfo_2,objects_2 = parse_xml(xmlname_2)
      save_feature=None
      labels=[]
      for idx_f_2,oject_2 in enumerate(objects_2):
        if oject_2['name']=="origin":
          labels.append("origin")
          continue
        if oject_2['name']=="miss":
          labels.append("miss")
          continue
        cropImg = im[oject_2["bbox"][1]:oject_2["bbox"][3], oject_2["bbox"][0]:oject_2["bbox"][2],:]
        _ = classifier.predict([cropImg], not args.center_only)
        feature = classifier.get_blob_data(args.feature_name)
        assert (feature.shape[0] == 1 )
        feature_shape = feature.shape
        # if save_feature is None:
        #     print('feature : {} : {}'.format(args.feature_name, feature_shape))
        #     save_feature = np.zeros((len(objects), feature.size),dtype=np.float32)
        feature_here = feature.reshape(1, feature.size)
       # save_feature[idx_f, :] = feature.copy()
        b_same_class=False
        for bb_fea in range(0,len(save_feature_all)):
          #print aa_fea," ",bb_fea," ",same_file_list[bb_fea][0]
          ret = comp_feature(save_feature_all[bb_fea],feature_here)
          print labels_all[bb_fea],ret,oject_2['name']
          if ret <0.2:
            print "            ",labels_all[bb_fea],ret,oject_2['name'],"     ok"
            b_same_class=True
            #print type(bb_fea)
            labels.append(labels_all[bb_fea])
            oject_2['name']=labels_all[bb_fea]
            break
        if b_same_class==False:
            print "                             ",oject_2['name'],"     background"
            labels.append("background")
            oject_2['name']="background"

      four_root = ElementTree()
      A1 = create_node('annotation',{},"")
      four_root._setroot(A1)
      B1 = create_node('foder',{},"2")
      B2 = create_node('filename',{},jpgname_2)
      B3 = create_node('path',{},"2")
      A1.append(B1)
      A1.append(B2)
      A1.append(B3)
      B4 = create_node('source',{},"")
      A1.append(B4)
      C1 = create_node('database',{},"Unknown")
      B4.append(C1)
      B5 = create_node('size',{},"")
      SubElement(B5,"width").text=str(im.shape[1])
      SubElement(B5,"height").text=str(im.shape[0])
      SubElement(B5,"depth").text="3"
      A1.append(B5)
      # SubElement(A1,"folder").text=str(width[i])
      # SubElement(A1,"filename").text=str(height[i])
      # SubElement(A1,"path").text="3"
      for idx_f_2,oject_2 in enumerate(objects_2):
        if oject_2['name']=="background":
          continue

        BBobj = create_node('object',{},"")
        SubElement(BBobj,"name").text=oject_2['name']
        SubElement(BBobj,"pose").text='Unspecified'
        SubElement(BBobj,"truncated").text='0'
        SubElement(BBobj,"difficult").text='0'
        SubElement(BBobj,"score").text=oject_2['score']
        SubElement(BBobj,"region").text=oject_2['region']
        SubElement(BBobj,"label_des").text=oject_2['label_des']
        SubElement(BBobj,"imageptr").text=oject_2['imageptr']
        child5 = SubElement(BBobj,"bndbox")
            # child1= create_node('name',{},obj['name'])
        SubElement(child5,"xmin").text=str(oject_2["bbox"][0])
        SubElement(child5,"ymin").text=str(oject_2["bbox"][1])
        SubElement(child5,"xmax").text=str(oject_2["bbox"][2])
        SubElement(child5,"ymax").text=str(oject_2["bbox"][3])
        A1.append(BBobj)
      print dir_out+"/"+basename_2+".xml"
      four_root.write(dir_out+"/"+basename_2+".xml", encoding="utf-8",xml_declaration=False)

  # args.input_file = os.path.expanduser(args.input_file)
  # if os.path.isdir(args.input_file):
  #   list_dir = os.listdir(args.input_file)
  # for idx_dir in list_dir:
  #   print idx_dir
  #   start_time = time.time()
  #   epoch_time = AverageMeter()
  #   if  os.path.isdir(args.input_file +"/"+idx_dir):
  #     #print idx_dir
  #     file_list=glob.glob(args.input_file +"/"+idx_dir+ '/*.' + args.ext)
  #     labels = [-1 for _ in xrange(len(file_list))]
  #     if not os.path.exists(args.feature_file+"/"+idx_dir+'/'):
  #       os.mkdir(args.feature_file+"/"+idx_dir+'/')
  #     with open(args.feature_file+"/"+idx_dir+"/list_file.txt","w") as z_f:
  #       tmp_file_list = [line+"\n" for line in file_list]
  #       z_f.writelines(tmp_file_list)

  #     save_feature = None
  #     size = len(file_list)
  #     for idx_f, _file_i in enumerate(file_list):
  #       _input=caffe.io.load_image(_file_i)
  #       _ = classifier.predict([_input], not args.center_only)
  #       feature = classifier.get_blob_data(args.feature_name)
  #       assert (feature.shape[0] == 1 )
  #       #assert (feature.shape[0] == 1 and score.shape[0] == 1)
  #       feature_shape = feature.shape
  #       #score   = classifier.get_blob_data(args.score_name)
  #      # score_shape = score.shape
  #       if save_feature is None:
  #           print('feature : {} : {}'.format(args.feature_name, feature_shape))
  #           save_feature = np.zeros((len(file_list), feature.size),dtype=np.float32)
  #       save_feature[idx_f, :] = feature.reshape(1, feature.size)
  #       tmp_file_name=os.path.basename(file_list[idx_f])
  #       #sio.savemat(args.feature_file+"/"+idx_dir+'/'+os.path.splitext(tmp_file_name)[0]+".feature", {'feature':feature})
      
  #     same_file_list=[]
  #     if len(same_file_list) == 0:
  #       tmp_list=[0]
  #       same_file_list.append(tmp_list)
  #     print size
  #     for aa_fea in range(1,size):
  #       #print len(same_file_list)
  #       b_same_class=False
  #       for bb_fea in range(0,len(same_file_list)):
  #         #print aa_fea," ",bb_fea," ",same_file_list[bb_fea][0]
  #         ret = comp_feature(save_feature[aa_fea],save_feature[same_file_list[bb_fea][0]])
  #         if ret <0.2:
  #           b_same_class=True
  #           same_file_list[bb_fea].append(aa_fea)
  #           break
  #       if b_same_class==False:
  #         tmp_list_in=[aa_fea]
  #         same_file_list.append(tmp_list_in)
  #     one_file_list=[ file_list[same_file_list[ss][0]] for ss in range(0,len(same_file_list))]
  #     with open(args.feature_file+"/"+idx_dir+"/everyclass_one_list_file.txt","w") as one_f:
  #       tmp_file_one = [line+"\n" for line in one_file_list]
  #       one_f.writelines(tmp_file_one)
  #     for cp_file in one_file_list:
  #       ppsring= "cp "+cp_file+" "+args.feature_file+"/"+idx_dir+"/"
  #       assert Popen_do(ppsring),ppsring+" error!"
  #     print idx_dir," different pic :",len(one_file_list)
  #     epoch_time.update(time.time() - start_time)
  #     start_time = time.time()
  #     need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (len(file_list)-1))
  #     need_time = '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
  #     print need_time


if __name__ == '__main__':
  main(sys.argv)
