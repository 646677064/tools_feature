
import argparse
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import cv2
import numpy as np
from math import *
import os
import sys
import cPickle
import random
import math
import shutil
import numpy as np
import os
import skimage.io as io#,transform,data
import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
import cPickle
import random
import math
import csv

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
    print "begin"
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
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        # with open(cachefile, 'w') as f:
        #     cPickle.dump(recs, f)
    else:
        print "load"
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
             print "depet wrong",imagename
        # print type(recs[imagename]['width']) ,type(width)
        if ((int(recs[imagename]['width'])  != width)or (int(recs[imagename]['height']) != height) or (int(recs[imagename]['depth']) != depth)) :
            print imagename,recs[imagename]['width'],width, recs[imagename]['height'],height,recs[imagename]['depth'],depth
            ballsame=False
            # cv2.namedWindow(imagename)
            # cv2.imshow(imagename, im)
            # cv2.waitKey (0)
    print "=========="
    print "=========="
    print "=========="
    if ballsame:
        print "==========the anotationes and images all match "
    else:
        print "==========please check the differences"
    print "end"

def parse_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    # tree=ElementTree()
    # tree.parse(filename)

    baseInfo={}
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

    return baseInfo,objec

def cut_rect_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations):
    print "==========="
    f_list = os.listdir(JPEGImagesDir)
    i=0
    for file_comp4 in f_list:
        im = cv2.imread(JPEGImagesDir+"/"+file_comp4)#jpegpath.format(imagename)
        #sp = im.shape
        img_height = im.shape[0]
        img_width = im.shape[1]
        img_depth = im.shape[2]
        if img_depth != 3 or img_height<=0 or img_depth<=0 :
             print "depet wrong",file_comp4
        if os.path.splitext(file_comp4)[1] == '.png':
            file_comp4 = os.path.splitext(file_comp4)[0] + ".jpg"

        cv2.imwrite(outDir_JPEGImages+"/w1_"+file_comp4, im)
        print "w1_"+file_comp4

        cropImg = im[0:img_height, 0:img_width/2]
        cv2.imwrite(outDir_JPEGImages+"/w2_1_"+file_comp4, cropImg)
        cropImg = im[0:img_height, img_width/2:img_width]
        cv2.imwrite(outDir_JPEGImages+"/w2_2_"+file_comp4, cropImg)
        print "w2_1_"+file_comp4
        print "w2_2_"+file_comp4

        cropImg = im[0:img_height, 0:img_width/3]
        cv2.imwrite(outDir_JPEGImages+"/w3_1_"+file_comp4, cropImg)
        cropImg = im[0:img_height, img_width/3:2*img_width/3]
        cv2.imwrite(outDir_JPEGImages+"/w3_2_"+file_comp4, cropImg)
        cropImg = im[0:img_height, 2*img_width/3:img_width]
        cv2.imwrite(outDir_JPEGImages+"/w3_3_"+file_comp4, cropImg)
        print "w3_1_"+file_comp4
        print "w3_2_"+file_comp4
        print "w3_3_"+file_comp4

        cropImg = im[0:img_height, 0:img_width/4]
        cv2.imwrite(outDir_JPEGImages+"/w4_1_"+file_comp4, cropImg)
        cropImg = im[0:img_height, img_width/4:img_width/2]
        cv2.imwrite(outDir_JPEGImages+"/w4_2_"+file_comp4, cropImg)
        cropImg = im[0:img_height, img_width/2:3*img_width/4]
        cv2.imwrite(outDir_JPEGImages+"/w4_3_"+file_comp4, cropImg)
        cropImg = im[0:img_height, 3*img_width/4:img_width]
        cv2.imwrite(outDir_JPEGImages+"/w4_4_"+file_comp4, cropImg)
        print "w4_1_"+file_comp4
        print "w4_2_"+file_comp4
        print "w4_3_"+file_comp4
        print "w4_4_"+file_comp4

 
def static_class_obj_num(SKUfile,cachefile):
    with open(SKUfile,"r") as SKUfile_f:
        SKUfile_lines = SKUfile_f.readlines()
    labelmap={}
    labellist=[]
    label_recall_all=[]
    label_recall_count=[]
    for ilable,labelname in enumerate(SKUfile_lines):
        labelname = labelname.strip().strip('\n').strip('\r')
        labelmap[labelname] = ilable+1
        labellist.append(labelname)
        label_recall_all.append(0)
        label_recall_count.append(0)

    with open(cachefile, 'r') as f:
        gt_obj = cPickle.load(f)
    for pindexname in gt_obj:
        for j_p,objcts_p in enumerate(gt_obj):
            #print objcts_p
            for idx,iclass in enumerate(objcts_p["gt_classes"]):
                label_recall_all[iclass-1]  += 1
            # if labelmap[objcts_p["name"]] == labelIDtxt[i_pNametxt]:
            #     areaRate_p = IOU(x1txt[i_pNametxt],y1txt[i_pNametxt],x2txt[i_pNametxt],y2txt[i_pNametxt],
            #                 objcts_p['bbox'][0],objcts_p['bbox'][1],objcts_p['bbox'][2],objcts_p['bbox'][3])

    for ii,ss in enumerate(label_recall_all):
        print labellist[ii]," = ",ss


def rotateimg2(src,degree=270):
    img=src.copy(); 
    rows,cols,ch = img.shape
    height = rows
    width = cols
    #degree = 270
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    matRotation[0,2] +=(widthNew-width)/2
    matRotation[1,2] +=(heightNew-height)/2
    dst = cv2.warpAffine(img,matRotation,(widthNew,heightNew))
    return dst

def rotate_270(jpg_path,xml_path,out_jpg_path,out_xml_path):
    # jpg_path="/home/liushuai/medical/baiwei/baiweiproj329/JPEGImages/"
    # xml_path="/home/liushuai/medical/baiwei/baiweiproj329/Annotations/"
    # out_jpg_path="/home/liushuai/medical/baiwei/baiweiproj329/JPEGImages_rotate90/"
    # out_xml_path="/home/liushuai/medical/baiwei/baiweiproj329/Annotations_rotate90/"
    jpg_list=os.listdir(jpg_path)
    for file_comp4 in jpg_list:
        basename = os.path.splitext(file_comp4)[0]
        im = cv2.imread(jpg_path+file_comp4)
        h=im.shape[0]
        w=im.shape[1]
       #print w,h
        #im=im[:,::-1]
        im = rotateimg2(im,270)
        cv2.imwrite(out_jpg_path+basename+"_90.jpg", im)
        file_xml = xml_path+"/"+basename+".xml"
        if not os.path.exists(file_xml):
            continue
        treeA=ElementTree()
        treeA.parse(file_xml)
        width = int(treeA.find('size/width').text)
        height = int(treeA.find('size/height').text)
        depth = int(treeA.find('size/depth').text)
        # if width==0 or height==0 or depth==0:
        #     print file_comp4,"width==0 or height==0 or depth==0"
        # treeA.find('size/width').text = str(scaled_width)
        # treeA.find('size/height').text = str(scaled_height)
        #treeA.find('size/depth').text = str(img_depth)
        
        bfind_one_space = False;
        treeA.find('size/width').text=str(h)
        treeA.find('size/height').text=str(w)
        for obj in treeA.findall('object'):
            # xmlname = obj.find('name').text
            # xmlname = xmlname.strip()
            # if xmlname=="Mini Oreo SDW 55g*24 Strawberry" :
            #     i=i+1
            #     print file_comp4
            #     bfind_one_space = True
            #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
            #     obj.find('name').text="Mini Oreo SDW 55g*24  Strawberry"
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            xmin_1 = int(h-ymax )
            ymin_1 = int(xmin )
            xmax_1 = int(h-ymin )
            ymax_1 = int(xmax )
            obj.find('bndbox').find('xmin').text = str(xmin_1)
            obj.find('bndbox').find('ymin').text = str(ymin_1)
            obj.find('bndbox').find('xmax').text = str(xmax_1)
            obj.find('bndbox').find('ymax').text = str(ymax_1)
            # if xmin>=xmax or ymin>= ymax or xmin<=0 or ymin <=0 or xmax>=width or ymax>height:
            #     print file_comp4
            if xmin_1<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('xmin').text = str(1)
            if ymin_1<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('ymin').text = str(1)
            if   xmax_1>= h :
                bfind_one_space = True
                obj.find('bndbox').find('xmax').text = str(h-1)
            if   ymax_1>= w :
                bfind_one_space = True
                obj.find('bndbox').find('ymax').text = str(w-1)
            if xmin>=xmax or ymin>= ymax:
                print out_xml_path

        #i+=1 
        treeA.write(out_xml_path+"/"+basename+"_90.xml", encoding="utf-8",xml_declaration=False)

def rotate_180(jpg_path,xml_path,out_jpg_path,out_xml_path):
    jpg_list=os.listdir(jpg_path)
    for file_comp4 in jpg_list:
        basename = os.path.splitext(file_comp4)[0]
        im = cv2.imread(jpg_path+file_comp4)
        h=im.shape[0]
        w=im.shape[1]
       #print w,h
        #im=im[:,::-1]
        im = rotateimg2(im,180)
        cv2.imwrite(out_jpg_path+basename+"_180.jpg", im)
        file_xml = xml_path+"/"+basename+".xml"
        if not os.path.exists(file_xml):
            continue
        treeA=ElementTree()
        treeA.parse(file_xml)
        width = int(treeA.find('size/width').text)
        height = int(treeA.find('size/height').text)
        depth = int(treeA.find('size/depth').text)
        # if width==0 or height==0 or depth==0:
        #     print file_comp4,"width==0 or height==0 or depth==0"
        # treeA.find('size/width').text = str(scaled_width)
        # treeA.find('size/height').text = str(scaled_height)
        #treeA.find('size/depth').text = str(img_depth)
        
        bfind_one_space = False;
        treeA.find('size/width').text=str(h)
        treeA.find('size/height').text=str(w)
        for obj in treeA.findall('object'):
            # xmlname = obj.find('name').text
            # xmlname = xmlname.strip()
            # if xmlname=="Mini Oreo SDW 55g*24 Strawberry" :
            #     i=i+1
            #     print file_comp4
            #     bfind_one_space = True
            #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
            #     obj.find('name').text="Mini Oreo SDW 55g*24  Strawberry"
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            xmin_1 = int(w-xmax )
            ymin_1 = int(h-ymax )
            xmax_1 = int(w-xmin )
            ymax_1 = int(h-ymin )
            obj.find('bndbox').find('xmin').text = str(xmin_1)
            obj.find('bndbox').find('ymin').text = str(ymin_1)
            obj.find('bndbox').find('xmax').text = str(xmax_1)
            obj.find('bndbox').find('ymax').text = str(ymax_1)
            # if xmin>=xmax or ymin>= ymax or xmin<=0 or ymin <=0 or xmax>=width or ymax>height:
            #     print file_comp4
            if xmin_1<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('xmin').text = str(1)
            if ymin_1<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('ymin').text = str(1)
            if   xmax_1>= w :
                bfind_one_space = True
                obj.find('bndbox').find('xmax').text = str(w-1)
            if   ymax_1>= h :
                bfind_one_space = True
                obj.find('bndbox').find('ymax').text = str(h-1)
            if xmin>=xmax or ymin>= ymax:
                print out_xml_path

        #i+=1 
        treeA.write(out_xml_path+"/"+basename+"_180.xml", encoding="utf-8",xml_declaration=False)

def tie_patch_to_pic(jpg_path,xml_path,out_jpg_path,out_xml_path):
    shape=(1024,1024,3)
    l_size=512#256 # 1024/4
    wwhh=2#4
    black_pic = np.zeros(shape, dtype=float, order='C')
    jpg_list=os.listdir(jpg_path)
    for file_comp4 in jpg_list:
        basename = os.path.splitext(file_comp4)[0]
        im = cv2.imread(jpg_path+file_comp4)
        h=im.shape[0]
        w=im.shape[1]
        w_roi=int(l_size if w>h else w*l_size/h)
        h_roi=int(h*l_size/w if w>h else l_size)
        print w,h,w_roi,h_roi
        im_roi = cv2.resize(im, (w_roi, h_roi), interpolation=cv2.INTER_CUBIC)
        radnum=np.random.randint(0,wwhh*wwhh)
        w_new = l_size*(radnum%wwhh)
        h_new = l_size*(radnum/wwhh)
        black_pic[:,:,:]=0
        black_pic[h_new:(h_new+h_roi),w_new:(w_new+w_roi),:]=im_roi

        cv2.imwrite(out_jpg_path+basename+"_rand1.jpg", black_pic)
        file_xml = xml_path+"/"+basename+".xml"
        if not os.path.exists(file_xml):
            continue
        treeA=ElementTree()
        treeA.parse(file_xml)
        width = int(treeA.find('size/width').text)
        height = int(treeA.find('size/height').text)
        depth = int(treeA.find('size/depth').text)
        # if width==0 or height==0 or depth==0:
        #     print file_comp4,"width==0 or height==0 or depth==0"
        # treeA.find('size/width').text = str(scaled_width)
        # treeA.find('size/height').text = str(scaled_height)
        #treeA.find('size/depth').text = str(img_depth)
        
        bfind_one_space = False;
        treeA.find('size/width').text=str(h)
        treeA.find('size/height').text=str(w)
        for obj in treeA.findall('object'):
            # xmlname = obj.find('name').text
            # xmlname = xmlname.strip()
            # if xmlname=="Mini Oreo SDW 55g*24 Strawberry" :
            #     i=i+1
            #     print file_comp4
            #     bfind_one_space = True
            #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
            #     obj.find('name').text="Mini Oreo SDW 55g*24  Strawberry"
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            xmin_1 = int(w_new+xmin*w_roi/w)
            ymin_1 = int(h_new+ymin*h_roi/h )
            xmax_1 = int(w_new+xmax*w_roi/w)
            ymax_1 = int(h_new+ymax*h_roi/h)
            obj.find('bndbox').find('xmin').text = str(xmin_1)
            obj.find('bndbox').find('ymin').text = str(ymin_1)
            obj.find('bndbox').find('xmax').text = str(xmax_1)
            obj.find('bndbox').find('ymax').text = str(ymax_1)
            # if xmin>=xmax or ymin>= ymax or xmin<=0 or ymin <=0 or xmax>=width or ymax>height:
            #     print file_comp4
            if xmin_1<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('xmin').text = str(1)
            if ymin_1<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('ymin').text = str(1)
            if   xmax_1>= 1024 :
                bfind_one_space = True
                obj.find('bndbox').find('xmax').text = str(1024-1)
            if   ymax_1>= 1024 :
                bfind_one_space = True
                obj.find('bndbox').find('ymax').text = str(1024-1)
            if xmin>=xmax or ymin>= ymax:
                print out_xml_path

        #i+=1 
        treeA.write(out_xml_path+"/"+basename+"_rand1.xml", encoding="utf-8",xml_declaration=False)

if __name__ == "__main__":
    # basedir="/storage2/liushuai/RFCN/make_data/"
    # JPEGImagesDir=basedir+"JPEG"
    # AnnotationsDir=""
    # outDir_JPEGImages=basedir+"JPEGOUT"
    # outDir_Annotations=""
    # cut_rect_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations)
    # cachefile = "/home/zhuxiaoli/py-R-FCN/data/cache/cookie_trainval_gt_roidb.pkl"
    # #SKUfile = "/nas/public/zhuxiaoli/cookie2/cookie324.txt"
    # static_class_obj_num(SKUfile,cachefile)
    # jpg_path='C:\\Users\\ysc\\Desktop\\nco\\jiabao000952.jpg'

    # xml_path='C:\\Users\\ysc\\Desktop\\nco\\jiabao000952.xml'
    # out_jpg_path='C:\\Users\\ysc\\Desktop\\nco\\rotate90\\jiabao000952_900.jpg'
    # out_xml_path='C:\\Users\\ysc\\Desktop\\nco\\rotate90\\jiabao000952_900.xml'
    jpg_path="/storage2/tiannuodata/ftpUpload/texiezhao0703-2327/JPEGImages/"
    xml_path="/storage2/tiannuodata/ftpUpload/texiezhao0703-2327/Annotations/"
    out_jpg_path="/storage2/tiannuodata/ftpUpload/texiezhao0703-2327/JPEGImages_new1/"
    out_xml_path="/storage2/tiannuodata/ftpUpload/texiezhao0703-2327/Annotations_new1/"
    tie_patch_to_pic(jpg_path,xml_path,out_jpg_path,out_xml_path)
