import numpy as np
import os
import sys
import argparse
import glob
import time
#import _init_paths
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
# import   cv2
# import skimage.io
# from scipy.ndimage import zoom
# from skimage.transform import resize

  
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

if __name__=="__main__":
    JPG_dir=""
    Anotation_dir="/"
    patchdir="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/analysis/patch/"
    result_patchlist=patchdir+"/result_patchlist/"
    out_xmlDir="/outdir/"
    # if not os.path.exists(patchdir+"/result_patchlist/"):
    #     os.mkdir(patchdir+"/result_patchlist/")
    subpatchs = os.listdir(result_patchlist)
    for subpatch in subpatchs:
        with open(result_patchlist+subpatch+".txt","r") as f_r:
            lines = f_r.readlines()
            last_lines = [labelname.strip().strip('\n').strip('\r') for labelname in lines]
        # with open(patchdir+"/result_patchlist/"+subpatch+".txt","w") as f_w:
        #     last_lines = [labelname.strip().strip('\n').strip('\r') for labelname in lines]
        #     file_lists=os.listdir(patchdir+subpatchs)
            for file in last_lines:
                file=os.path.splitext(file)[0]
                splitthins=file.split("_")
                name=splitthins[0]
                if len(splitthins)>5:
                    for ia in range(1,len(splitthins)-4):
                        name=name+"_"+splitthins[ia]
                ymax_target=splitthins[-1]
                xmax_target=splitthins[-2]
                ymin_target=splitthins[-3]
                xmin_target=splitthins[-4]
                file_tmp = Anotation_dir+name+".xml"
                # if file in last_lines:
                #     continue
                # f_w.write(file+"\n")
                treeA=ElementTree()
                treeA.parse(file_tmp)
                width = int(treeA.find('size/width').text)
                height = int(treeA.find('size/height').text)
                depth = int(treeA.find('size/depth').text)
                bfind_one_space = False;

                # JPEG_resetdir =JPG_dir#work_dir +name +"/JPEGImages_reset/"
                # print os.path.join(JPEG_resetdir, name+".jpg")
                # if os.path.exists(os.path.join(JPEG_resetdir, os.path.splitext(file_comp4)[0]+".jpg")):#(JPEG_resetdir+"/"+os.path.splitext(file_comp4)[0]+".jpg"):
                #     im = cv2.imread(os.path.join(JPEG_resetdir, os.path.splitext(file_comp4)[0]+".jpg"))
                #     #sp = im.shape
                #     imheight = im.shape[0]
                #     imwidth = im.shape[1]
                #     imdepth = im.shape[2]
                #     if width!=imwidth:
                #         bfind_one_space = True
                #         treeA.find('size/width').text=str(imwidth)
                #         width=imwidth
                #         print file_comp4,"error size/width"
                #     if height!=imheight:
                #         bfind_one_space = True
                #         treeA.find('size/height').text=str(imheight)
                #         height=imheight
                #         print file_comp4,"error size/height"
                #     if depth!=imdepth:
                #         bfind_one_space = True
                #         treeA.find('size/depth').text=str(imdepth)
                #         depth=imdepth
                #         print file_comp4,"error size/depth"
                # else:
                #     print file_comp4,'not exist'
                # if width==0 or height==0 or depth==0:
                #     print file_comp4,"width==0 or height==0 or depth==0,wrong and please check it!"
                #     return
                
                #bfind_one_space = False;
                for obj in treeA.findall('object'):
                    xmlname = obj.find('name').text
                    xmlname = xmlname.strip()
                    # for ifind,tmp_name in enumerate(src_list):
                    #     if xmlname==tmp_name:
                    #         i=i+1
                    #         print file_comp4
                    #         bfind_one_space = True
                    #         obj.find('name').text=des_list[ifind]
                    #         break
                    xmin = int(obj.find('bndbox').find('xmin').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    if xmin_target==xmin and ymin_target == ymin and xmax_target==xmax and ymax_target==ymax:
                        obj.find('name').text=subpatch
                        bfind_one_space==True
                    # if xmin>=xmax or ymin>= ymax or xmin<=0 or ymin <=0 or xmax>=width or ymax>height:
                    #     print file_comp4
                    # if xmin<=0 :
                    #     bfind_one_space = True
                    #     obj.find('bndbox').find('xmin').text = str(1)
                    # if ymin<=0 :
                    #     bfind_one_space = True
                    #     obj.find('bndbox').find('ymin').text = str(1)
                    # if   xmax>= width :
                    #     bfind_one_space = True
                    #     obj.find('bndbox').find('xmax').text = str(width-1)
                    # if   ymax>= height :
                    #     bfind_one_space = True
                    #     obj.find('bndbox').find('ymax').text = str(height-1)
                    # if xmin>=xmax or ymin>= ymax:
                    #     print file_comp4

                if bfind_one_space==True:
                    print name
                    treeA.write(out_xmlDir+name+".xml", encoding="utf-8",xml_declaration=False)  