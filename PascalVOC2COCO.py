# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
#import matplotlib.pyplot as plt
#import skimage.io as io
import cv2
#from labelme import utils
import numpy as np
import glob
#import PIL.Image
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
import os,sys

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

def image_check(annopath,
             imagesetfile,
             cachedir,
             jpegpath):
    print("begin")
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'check_image_annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_xml_WH(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        # with open(cachefile, 'w') as f:
        #     cPickle.dump(recs, f)
    else:
        print("load")
        # with open(cachefile, 'r') as f:
        #     recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    ballsame = True
    for imagename in imagenames:
        im = cv2.imread(jpegpath.format(imagename))
        #sp = im.shape
        height = im.shape[0]
        width = im.shape[1]
        depth = im.shape[2]
        if depth != 3 :
             print("depet wrong",imagename)
        # print type(recs[imagename]['width']) ,type(width)
        if ((int(recs[imagename]['width'])  != width)or (int(recs[imagename]['height']) != height) or (int(recs[imagename]['depth']) != depth)) :
            print(imagename,recs[imagename]['width'],width, recs[imagename]['height'],height,recs[imagename]['depth'],depth)
            ballsame=False
            # cv2.namedWindow(imagename)
            # cv2.imshow(imagename, im)
            # cv2.waitKey (0)
    print("==========")
    if ballsame:
        print("==========the anotationes and images all match ")
    else:
        print("==========please check the differences")
    print("end")

def parse_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    # tree=ElementTree()
    # tree.parse(filename)

    baseInfo={}
    if tree.find('folder'):
        baseInfo['folder'] = tree.find('folder').text
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


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json',base_dir="/data/liushuai/nestle4goods/nestle4goodsproj1/"):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.base_dir = base_dir

        self.save_json()

    def make_labelmap(self,sku_dir,sku_filename):
        imagesetfile = sku_dir+"/"+sku_filename
        i=1
        with open(imagesetfile, 'r') as f:
            lines_semi = f.readlines()

        lines = []
        sku_list = []
        sku_list.append("__background__")
        [lines.append(j) for j in lines_semi if not j in lines]
        for line in lines:
            x=line.strip()
            x=line.strip('\n')
            x=line.strip('\r')
            x=line.strip('\n')
            x=line.strip()
            if x=="":
                continue
            i += 1
            sku_list.append(x)
            self.supercategory = x
            if self.supercategory not in self.label:
                self.categories.append(self.categorie())
                self.label.append(self.supercategory)
        print(imagesetfile,"label = ",i)
        return tuple(sku_list)

    def data_transfer_3(self):
        # notconvert_list=[]
        # notconvert_list.append("budweiser47401")
        # notconvert_list.append("budweiser47379")
        txt_dir=self.base_dir+"/ImageSets/Main/"
        outdir=self.base_dir+"//odformat/"
        out_trainfile=outdir+"coco_trainvalmini.odgt"
        out_valfile=outdir+"coco_minival2014.odgt"
        train_orgpath=txt_dir+"/trainval.txt"
        with open(train_orgpath, 'r') as f:
            lines_semi = f.readlines()

        train_list = []
        for line in lines_semi:
            x=line.strip()
            x=line.strip('\n')
            x=line.strip('\r')
            x=line.strip('\n')
            x=line.strip()
            train_list.append(x)

        test_orgpath=txt_dir+"/test.txt"
        with open(test_orgpath, 'r') as f:
            lines_semi = f.readlines()

        test_list = []
        for line in lines_semi:
            x=line.strip()
            x=line.strip('\n')
            x=line.strip('\r')
            x=line.strip('\n')
            x=line.strip()
            test_list.append(x)

        trainstring_list=[]
        teststring_list=[]
        print("xml len:",len(self.xml))
        for num, json_file in enumerate(self.xml):

            basename=os.path.basename(json_file)
            basename=os.path.splitext(basename)[0]
            # if basename in notconvert_list:
            #     continue
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d %s ' % (
                num + 1, len(self.xml),basename))
            sys.stdout.flush()

            self.json_file = json_file
            self.num = num
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)
            # path=os.path.split(self.json_file)[0]
            # path=os.path.split(path)[0]
            #####################################################
            baseInfo,objects = parse_xml(json_file)
            writelin="{\"gtboxes\":["
            obj_path = glob.glob(os.path.join(path, 'SegmentationObject', '*.png'))
            bboxwrite=""
            bfirst=True
            for objbbox in objects:
                if bfirst==False:
                        bboxwrite=bboxwrite+","
                else:
                    bfirst=False
                bboxwrite=bboxwrite+"{\"bbox\": "
                bboxwrite=bboxwrite+str([objbbox['bbox'][0], objbbox['bbox'][1], objbbox['bbox'][2] - objbbox['bbox'][0], objbbox['bbox'][3] - objbbox['bbox'][1]])
                bboxwrite=bboxwrite+", \"occ\": 0,"
                bboxwrite=bboxwrite+"\"tag\": \""+objbbox['name']
                bboxwrite=bboxwrite+"\",\"extra\": {\"ignore\": 0}"
                bboxwrite=bboxwrite+"}"
            writelin=writelin+bboxwrite+"],"
            fpath="\"fpath\": \"/val2014/"+basename+".jpg\", \"dbName\": \"COCO\", \"dbInfo\": {\"vID\": \"COCO_trainval2014_womini\", \"frameID\": -1}, \"width\": "
            fpath=fpath+baseInfo['size/width']+", \"height\": "+baseInfo['size/height']+",\"ID\": \""+basename+".jpg\""
            writelin=writelin+fpath+"}\n"
            #####################################################
            # writelin="{\"gtboxes\":["
            # obj_path = glob.glob(os.path.join(path, 'SegmentationObject', '*.png'))
            # bboxwrite=""
            # bfirst=True
            # with open(json_file, 'r') as fp:
            #     bobject_in=False
            #     for p in fp:
            #         # if 'folder' in p:
            #         #     folder =p.split('>')[1].split('<')[0]
            #         if 'filename' in p:
            #             path=os.path.split(self.json_file)[0]
            #             self.filen_ame = basename#p.split('>')[1].split('<')[0]

            #             self.path = os.path.join(path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')


            #         if 'width' in p:
            #             sssssssa=p.split('>')[1].split('<')[0]
            #             # if basename=="budweiser47401":
            #             #     print("===  ",sssssssa)
            #             self.width = int(sssssssa)
            #         if 'height' in p:
            #             self.height = int(p.split('>')[1].split('<')[0])

            #             self.images.append(self.image())

            #         if '</object>' in p:
            #             bobject_in=False
            #         if '<object>' in p:
            #             bobject_in=True
            #         if bobject_in==True and 'name' in p:
            #             self.supercategory = p.split('>')[1].split('<')[0]
            #             if self.supercategory not in self.label:
            #                 print("======================================error")
            #         if bobject_in==True and '<bndbox>' in p:
            #             d = [next(fp).split('>')[1].split('<')[0] for _ in range(4)]
            #             # 边界框
            #             x1 = int(d[-4]);
            #             y1 = int(d[-3]);
            #             x2 = int(d[-2]);
            #             y2 = int(d[-1])
            #             self.rectangle = [x1, y1, x2, y2]
            #             self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]
            #             self.annotations.append(self.annotation())
            #             self.annID += 1
            #             if bfirst==False:
            #                     bboxwrite=bboxwrite+","
            #             else:
            #                 bfirst=False
            #             bboxwrite=bboxwrite+"{\"bbox\": "
            #             bboxwrite=bboxwrite+str(self.bbox)
            #             bboxwrite=bboxwrite+", \"occ\": 0,"
            #             bboxwrite=bboxwrite+"\"tag\": \""+self.supercategory
            #             bboxwrite=bboxwrite+"\",\"extra\": {\"ignore\": 0}"
            #             bboxwrite=bboxwrite+"}"

            # writelin=writelin+bboxwrite+"],"
            # fpath="\"fpath\": \"/val2014/"+basename+".jpg\", \"dbName\": \"COCO\", \"dbInfo\": {\"vID\": \"COCO_trainval2014_womini\", \"frameID\": -1}, \"width\": "
            # fpath=fpath+str(self.width)+", \"height\": "+str(self.height)+",\"ID\": \""+basename+".jpg\""
            # writelin=writelin+fpath+"}\n"
            if basename in train_list:
                for zzz in train_list:
                    if basename == zzz:
                        trainstring_list.append(writelin)
            elif basename in test_list:
                for zzz in test_list:
                    if basename == zzz:
                        teststring_list.append(writelin)
        with open(out_trainfile, 'w') as fout_trainfile:
            fout_trainfile.writelines(trainstring_list)
        with open(out_valfile, 'w') as fout_valfile:
            fout_valfile.writelines(teststring_list)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def data_transfer_2(self):
        notconvert_list=[]
        notconvert_list.append("budweiser47401")
        notconvert_list.append("budweiser47379")
        txt_dir=self.base_dir+"//ImageSets/Main/"
        outdir=self.base_dir+"//odformat/"
        out_trainfile=outdir+"coco_trainvalmini.odgt"
        out_valfile=outdir+"coco_minival2014.odgt"
        train_orgpath=txt_dir+"/trainval.txt"
        with open(train_orgpath, 'r') as f:
            lines_semi = f.readlines()

        train_list = []
        for line in lines_semi:
            x=line.strip()
            x=line.strip('\n')
            x=line.strip('\r')
            x=line.strip('\n')
            x=line.strip()
            train_list.append(x)

        test_orgpath=txt_dir+"/test.txt"
        with open(test_orgpath, 'r') as f:
            lines_semi = f.readlines()

        test_list = []
        for line in lines_semi:
            x=line.strip()
            x=line.strip('\n')
            x=line.strip('\r')
            x=line.strip('\n')
            x=line.strip()
            test_list.append(x)

        trainstring_list=[]
        teststring_list=[]
        print("xml len:",len(self.xml))
        for num, json_file in enumerate(self.xml):

            basename=os.path.basename(json_file)
            basename=os.path.splitext(basename)[0]
            if basename in notconvert_list:
                continue
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d %s ' % (
                num + 1, len(self.xml),basename))
            sys.stdout.flush()

            self.json_file = json_file
            self.num = num
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)
            # path=os.path.split(self.json_file)[0]
            # path=os.path.split(path)[0]
            writelin="{\"gtboxes\":["
            obj_path = glob.glob(os.path.join(path, 'SegmentationObject', '*.png'))
            bboxwrite=""
            bfirst=True
            with open(json_file, 'r') as fp:
                bobject_in=False
                for p in fp:
                    # if 'folder' in p:
                    #     folder =p.split('>')[1].split('<')[0]
                    if 'filename' in p:
                        path=os.path.split(self.json_file)[0]
                        self.filen_ame = basename#p.split('>')[1].split('<')[0]

                        self.path = os.path.join(path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')


                    if 'width' in p:
                        sssssssa=p.split('>')[1].split('<')[0]
                        # if basename=="budweiser47401":
                        #     print("===  ",sssssssa)
                        self.width = int(sssssssa)
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])

                        self.images.append(self.image())

                    if '</object>' in p:
                        bobject_in=False
                    if '<object>' in p:
                        bobject_in=True
                    if bobject_in==True and 'name' in p:
                        self.supercategory = p.split('>')[1].split('<')[0]
                        if self.supercategory not in self.label:
                            print("======================================error")
                    if bobject_in==True and '<bndbox>' in p:
                        d = [next(fp).split('>')[1].split('<')[0] for _ in range(4)]
                        # 边界框
                        x1 = int(d[-4]);
                        y1 = int(d[-3]);
                        x2 = int(d[-2]);
                        y2 = int(d[-1])
                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]
                        self.annotations.append(self.annotation())
                        self.annID += 1
                        if bfirst==False:
                                bboxwrite=bboxwrite+","
                        else:
                            bfirst=False
                        bboxwrite=bboxwrite+"{\"bbox\": "
                        bboxwrite=bboxwrite+str(self.bbox)
                        bboxwrite=bboxwrite+", \"occ\": 0,"
                        bboxwrite=bboxwrite+"\"tag\": \""+self.supercategory
                        bboxwrite=bboxwrite+"\",\"extra\": {\"ignore\": 0}"
                        bboxwrite=bboxwrite+"}"

            writelin=writelin+bboxwrite+"],"
            fpath="\"fpath\": \"/val2014/"+basename+".jpg\", \"dbName\": \"COCO\", \"dbInfo\": {\"vID\": \"COCO_trainval2014_womini\", \"frameID\": -1}, \"width\": "
            fpath=fpath+str(self.width)+", \"height\": "+str(self.height)+",\"ID\": \""+basename+".jpg\""
            writelin=writelin+fpath+"}\n"
            if basename in train_list:
                for zzz in train_list:
                    if basename == zzz:
                        trainstring_list.append(writelin)
            elif basename in test_list:
                for zzz in test_list:
                    if basename == zzz:
                        teststring_list.append(writelin)
        with open(out_trainfile, 'w') as fout_trainfile:
            fout_trainfile.writelines(trainstring_list)
        with open(out_valfile, 'w') as fout_valfile:
            fout_valfile.writelines(teststring_list)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def data_transfer(self):
        for num, json_file in enumerate(self.xml):

            basename=os.path.basename(json_file)
            basename=os.path.splitext(basename)[0]
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d %s' % (
                num + 1, len(self.xml),basename))
            sys.stdout.flush()

            self.json_file = json_file
            self.num = num
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)
            # path=os.path.split(self.json_file)[0]
            # path=os.path.split(path)[0]
            obj_path = glob.glob(os.path.join(path, 'SegmentationObject', '*.png'))
            with open(json_file, 'r') as fp:
                bobject_in=False
                for p in fp:
                    # if 'folder' in p:
                    #     folder =p.split('>')[1].split('<')[0]
                    if 'filename' in p:
                        path=os.path.split(self.json_file)[0]
                        self.filen_ame = basename#p.split('>')[1].split('<')[0]

                        self.path = os.path.join(path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')
                        # if self.path not in obj_path:
                        #     print "path not in obj_path"
                        #     break


                    if 'width' in p:
                        self.width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])

                        self.images.append(self.image())

                    if '</object>' in p:
                        bobject_in=False
                    if '<object>' in p:
                        bobject_in=True
                        # # 类别
                        # d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                        # self.supercategory = d[0]
                        # if self.supercategory not in self.label:
                        #     # self.categories.append(self.categorie())
                        #     # self.label.append(self.supercategory)
                        #     print "======================================error"

                        # # 边界框
                        # x1 = int(d[-4]);
                        # y1 = int(d[-3]);
                        # x2 = int(d[-2]);
                        # y2 = int(d[-1])
                        # self.rectangle = [x1, y1, x2, y2]
                        # self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

                        # self.annotations.append(self.annotation())
                        # self.annID += 1
                    if bobject_in==True and 'name' in p:
                        self.supercategory = p.split('>')[1].split('<')[0]
                        if self.supercategory not in self.label:
                            # self.categories.append(self.categorie())
                            # self.label.append(self.supercategory)
                            print("======================================error")
                    if bobject_in==True and '<bndbox>' in p:
                        d = [next(fp).split('>')[1].split('<')[0] for _ in range(4)]
                        # 边界框
                        x1 = int(d[-4]);
                        y1 = int(d[-3]);
                        x2 = int(d[-2]);
                        y2 = int(d[-1])
                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]
                        self.annotations.append(self.annotation())
                        self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie

    def annotation(self):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        # annotation['bbox'] = list(map(float, self.bbox))
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i;
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i;
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox=[]
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.make_labelmap(self.base_dir,"skufile.txt")
        self.data_transfer_3()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示

def txt_2_odformattrain_val(txt_dir,json_file,outdir):
    out_trainfile=outdir+"coco_trainvalmini.odgt"
    out_valfile=outdir+"coco_minival2014.odgt"

    train_orgpath=txt_dir+"/trainval.txt"
    with open(train_orgpath, 'r') as f:
        lines_semi = f.readlines()

    train_list = []
    for line in lines_semi:
        x=line.strip()
        x=line.strip('\n')
        x=line.strip('\r')
        x=line.strip('\n')
        x=line.strip()
        train_list.append(x)

    test_orgpath=txt_dir+"/test.txt"
    with open(test_orgpath, 'r') as f:
        lines_semi = f.readlines()

    test_list = []
    for line in lines_semi:
        x=line.strip()
        x=line.strip('\n')
        x=line.strip('\r')
        x=line.strip('\n')
        x=line.strip()
        test_list.append(x)

    datas = json.load(open(json_file,'r'))
    with open(out_trainfile, 'w') as fout_trainfile:
        for trian in train_list:
            for line in datas['images']:
                if trian==line['file_name']:
                    print(trian)
                    writelin="{\"gtboxes\":["
                    # print line['id']
                    # print line['file_name']
                    # print line['height']
                    # print line['width']
                    bfirst=True
                    bfindanntaotion=False
                    bboxwrite=""
                    for annotations_ in datas['annotations']:
                        if annotations_['image_id'] != line['id'] and bfindanntaotion==True:
                            break;
                        if annotations_['image_id'] == line['id']:
                            bfindanntaotion=True
                            if bfirst==False:
                                bboxwrite=bboxwrite+","
                            else:
                                bfirst=False
                            bboxwrite=bboxwrite+"{\"bbox\": "
                            bboxwrite=bboxwrite+str(annotations_['bbox'])
                            bboxwrite=bboxwrite+", \"occ\": 0,"
                            # print annotations_['bbox']
                            # print annotations_['category_id']
                            for id_ in datas['categories']:
                                if id_['id']==annotations_['category_id']:
                                    bboxwrite=bboxwrite+"\"tag\": \""+id_['name']
                                    bboxwrite=bboxwrite+"\",\"extra\": {\"ignore\": 0}"
                                    #print id_['name'],"\n"
                            bboxwrite=bboxwrite+"}"
                    writelin=writelin+bboxwrite+"],"
                    fpath="\"fpath\": \"/val2014/"+trian+".jpg\", \"dbName\": \"COCO\", \"dbInfo\": {\"vID\": \"COCO_trainval2014_womini\", \"frameID\": -1}, \"width\": "
                    fpath=fpath+str(line['width'])+", \"height\": "+str(line['height'])+",\"ID\": \""+trian+".jpg\""
                    writelin=writelin+fpath+"}\n"
                    # print writelin
                    fout_trainfile.write(writelin)

    with open(out_valfile, 'w') as fout_valfile:
        for test in test_list:
            for line in datas['images']:
                if test==line['file_name']:
                    print(test)
                    writelin="{\"gtboxes\":["
                    # print line['id']
                    # print line['file_name']
                    # print line['height']
                    # print line['width']
                    bfirst=True
                    bfindanntaotion=False
                    bboxwrite=""
                    for annotations_ in datas['annotations']:
                        if annotations_['image_id'] != line['id'] and bfindanntaotion==True:
                            break;
                        if annotations_['image_id'] == line['id']:
                            bfindanntaotion=True
                            if bfirst==False:
                                bboxwrite=bboxwrite+","
                            else:
                                bfirst=False
                            bboxwrite=bboxwrite+"{\"bbox\": "
                            bboxwrite=bboxwrite+str(annotations_['bbox'])
                            bboxwrite=bboxwrite+", \"occ\": 0,"
                            # print annotations_['bbox']
                            # print annotations_['category_id']
                            for id_ in datas['categories']:
                                if id_['id']==annotations_['category_id']:
                                    bboxwrite=bboxwrite+"\"tag\": \""+id_['name']
                                    bboxwrite=bboxwrite+"\",\"extra\": {\"ignore\": 0}"
                                    #print id_['name'],"\n"
                            bboxwrite=bboxwrite+"}"
                    writelin=writelin+bboxwrite+"],"
                    fpath="\"fpath\": \"/val2014/"+trian+".jpg\", \"dbName\": \"COCO\", \"dbInfo\": {\"vID\": \"COCO_trainval2014_womini\", \"frameID\": -1}, \"width\": "
                    fpath=fpath+str(line['width'])+", \"height\": "+str(line['height'])+",\"ID\": \""+trian+".jpg\""
                    writelin=writelin+fpath+"}\n"
                    # print writelin
                    # print writelin
                    fout_valfile.write(writelin)

if __name__ == "__main__":
    base_dir="/data/liushuai/nestle4goods/nestle4goodsproj1/"
    base_dir="/storage3/tiannuodata/work/projdata/nestle4goods/nestle4goodsproj1/"
    txt_list=base_dir+"/ImageSets/Main/"
    json_file=base_dir+"/new.json"
    #json_file="/home/liushuai/work/light_head_rcnn/data/MSCOCO/instances_minival2014.json"
    #json_file='/data/tiannuodata/nestle4goods/nestle4goodsproj1/2.json'
    outdir=base_dir+"//odformat/" #coco_minival2014.odgt coco_trainvalmini.odgt
    #txt_2_odformattrain_val(txt_list,json_file,outdir)
    xml_file = glob.glob(base_dir+'/Annotations/*.xml')
    PascalVOC2coco(xml_file, base_dir+'/new.json',base_dir)

    # #=========================================================================test
    # json_file='/data/tiannuodata/nestle4goods/nestle4goodsproj1/new.json'
    # # person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
    # # captions_val2017.json  # Image Caption的标注格式

    # data=json.load(open(json_file,'r'))

    # data_2={}
    # # data_2['info']=data['info']
    # # data_2['licenses']=data['licenses']
    # data_2['images']=[data['images'][0]] # 只提取第一张图片
    # data_2['categories']=data['categories']
    # annotation=[]

    # # 通过imgID 找到其所有对象
    # imgID=data_2['images'][0]['id']
    # for ann in data['annotations']:
    #     if ann['image_id']==imgID:
    #         annotation.append(ann)

    # data_2['annotations']=annotation

    # # 保存到新的JSON文件，便于查看数据特点
    # json.dump(data_2,open('/data/tiannuodata/nestle4goods/nestle4goodsproj1/2.json','w'),indent=4) # indent=4 更加美观显示


    # data_3={}
    # # data_3['info']=data['info']
    # # data_3['licenses']=data['licenses']
    # data_3['images']=[data['images'][1]] # 只提取第一张图片
    # data_3['categories']=data['categories']
    # annotation=[]

    # # 通过imgID 找到其所有对象
    # imgID=data_3['images'][0]['id']
    # for ann in data['annotations']:
    #     if ann['image_id']==imgID:
    #         annotation.append(ann)

    # data_3['annotations']=annotation

    # # 保存到新的JSON文件，便于查看数据特点
    # json.dump(data_3,open('/data/tiannuodata/nestle4goods/nestle4goodsproj1/3.json','w'),indent=4) # indent=4 更加美观显示
    # #=========================================================================test