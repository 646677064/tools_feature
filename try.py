#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

#import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement


  
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
       

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

#def save_pic(im, class_name, dets, thresh,out_file):

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
    CONF_THRESH = 0.45
    NMS_THRESH = 0.3
    thresh=CONF_THRESH

    baseInfo={}
    baseInfo['filename'] = ""
    baseInfo['path'] = ""
    baseInfo['source/database'] = ""
    #tree.find('database')
    baseInfo['size/width'] = str(im.shape[1])
    baseInfo['size/height'] = str(im.shape[0])
    baseInfo['size/depth'] = str(im.shape[2])
    baseInfo['segmented'] = ""
    objects = []
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
            bbox = dets[i, :4].copy()
            score = dets[i, -1].copy()
            # print bbox,score,cls
            # cv2.rectangle(im_tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255),2)
            obj_struct = {}
            obj_struct['name'] = "goods"
            obj_struct['pose'] = ""
            obj_struct['truncated'] = ""
            obj_struct['difficult'] = ""
            obj_struct['bbox'] = [int(bbox[0]),
                                  int(bbox[1]),
                                  int(bbox[2]),
                                  int(bbox[3])]
            objects.append(obj_struct)
    # im_tmp = im#im[:, :, (2, 1, 0)]
    # cv2.imwrite(save_ff,im_tmp)
    return baseInfo,objects

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # cfg.TEST.HAS_RPN = True  # Use RPN for proposals


    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    ## Number of top scoring boxes to keep after applying NMS to RPN proposals
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000 #lius 300
    #cfg.TEST.RPN_POST_NMS_TOP_N = 2000 #lius 300
    cfg.TEST.AGNOSTIC=True
    #cfg.TEST.AGNOSTIC=False
    cfg.TEST.RPN_MIN_SIZE=10

    args = parse_args()

    # prototxt = "/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641/test.t"
    # caffemodel = "/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641/VGG16_faster_rcnn_final.caffemodel"
    prototxt = "/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641/test.t"
    caffemodel = "/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641/VGG16_faster_rcnn_final.caffemodel"

    # prototxt = "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/models/baiwei/VGG16/faster_rcnn_end2end//test.prototxt";
    # caffemodel = "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/output/faster_rcnn_end2end/baiweiproj1_trainval/vgg16_faster_rcnn_iter_65000.caffemodel";
    
    # prototxt= "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/models/baiwei/ResNet-50/faster_rcnn_BN_SCALE_Merged/faster_rcnn_end2end//test.prototxt";
    # caffemodel = "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/output/faster_rcnn_end2end/baiweiproj1_trainval/resnet50_faster_rcnn_bn_scale_merged_end2end_iter_120000.caffemodel";
    
    prototxt= "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/models/baiwei/VGG16/faster_rcnn_end2end_agnostic//test.prototxt";
    caffemodel = "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/output/faster_rcnn_end2end/baiweiproj1_trainval/vgg16_faster_rcnn_agnostic_iter_220000.caffemodel"
    # prototxt= "/home/liushuai/tiannuocaffe/py-rfcn-gpu/models/baiwei/ResNet-50/rfcn_end2end/test_agnostic.prototxt";
    # caffemodel = "/home/liushuai/tiannuocaffe/py-rfcn-gpu/output/goodsType/baiweiproj1_trainval/ResNet-50_baiwei_rfcn_ohem_iter_120000.caffemodel"

    # prototxt= "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/models/baiwei/VGG16/faster_rcnn_end2end//test.prototxt";
    # caffemodel = "/storage2/liushuai/faster_rcnn/faster-rcnn-resnet-master/output/faster_rcnn_end2end/baiweiproj1_trainval/vgg16_faster_rcnn_iter_215000.caffemodel"

    # prototxt= "/storage2/liushuai/faster_rcnn/py-faster-rcnn-master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt";
    # caffemodel = "/storage2/liushuai/faster_rcnn/py-faster-rcnn-master/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_30000.caffemodel"

    save_ff="/storage2/liushuai/gs6_env/bak//test_resultvgg.jpg"
    im_name="/storage2/liushuai/faster_rcnn/FasterRCNN-Encapsulation-Cplusplus/faster_cxx_lib_ev2641//cat.jpg"
    im_name="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj1/JPEGImages/budweiser08782.jpg"#budweiser15059.jpg"


    cfg.GPU_ID = 6
    in_dir="/storage2/tiannuodata/work/projdata/baiwei/testdata/JPEGImages/"
    out_xmldir="/storage2/tiannuodata/work/projdata/baiwei/testdata/result_xml//"
    
    prototxt="/home/liushuai/tiannuocaffe/py-rfcn-gpu/models/shape/ResNet-101_2/rfcn_end2end/s16_14/b14_test_16_s_4_8_16_32_agnostic.prototxt"
    caffemodel="/home/liushuai/tiannuocaffe/py-rfcn-gpu/output/goodsType/shapeproj2_trainval/ResNet-101_2_b14_16_s_4_8_16_32_shape_rfcn_ohem_iter_200000.caffemodel"
    num_class=2-1#1360-1 #341
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))


    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i /storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/analysis/xml_shape/in xrange(2):
    #     _, _= im_detect(net, im)
    file_list = os.listdir(in_dir)
    for file_name in file_list:
        if os.path.splitext(file_name)[1] == '.xml':
            continue
        basename_2=os.path.splitext(file_name)[0]
        im_name = in_dir+"/"+file_name
        baseInfo,objects_2 = demo(net, im_name,num_class,save_ff)
        four_root = ElementTree()
        A1 = create_node('annotation',{},"")
        four_root._setroot(A1)
        B1 = create_node('foder',{},"2")
        B2 = create_node('filename',{},basename_2)
        B3 = create_node('path',{},"2")
        A1.append(B1)
        A1.append(B2)
        A1.append(B3)
        B4 = create_node('source',{},"")
        A1.append(B4)
        C1 = create_node('database',{},"Unknown")
        B4.append(C1)
        B5 = create_node('size',{},"")
        SubElement(B5,"width").text=baseInfo['size/width']#str(im.shape[1])
        SubElement(B5,"height").text=baseInfo['size/height']#str(im.shape[0])
        SubElement(B5,"depth").text=baseInfo['size/depth']#"3"
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
            # SubElement(BBobj,"score").text=oject_2['score']
            # SubElement(BBobj,"region").text=oject_2['region']
            # SubElement(BBobj,"label_des").text=oject_2['label_des']
            # SubElement(BBobj,"imageptr").text=oject_2['imageptr']
            child5 = SubElement(BBobj,"bndbox")
            # child1= create_node('name',{},obj['name'])
            SubElement(child5,"xmin").text=str(oject_2["bbox"][0])
            SubElement(child5,"ymin").text=str(oject_2["bbox"][1])
            SubElement(child5,"xmax").text=str(oject_2["bbox"][2])
            SubElement(child5,"ymax").text=str(oject_2["bbox"][3])
            A1.append(BBobj)
        print out_xmldir+"/"+basename_2+".xml"
        four_root.write(out_xmldir+"/"+basename_2+".xml", encoding="utf-8",xml_declaration=False)

    #plt.show()
