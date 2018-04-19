
import argparse
from collections import OrderedDict
from google.protobuf import text_format
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
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

    return baseInfo,objects

def  remove_Anotations(AnnotationDir,outDir):
    print "==========="
    # removelist_key = ["Maxwellhouse","G7"]
    # removelist_key.append("Oldtown")
    # removelist_key.append("Hougu")
    # removelist_key.append("Kapalapi")
    # removelist_key.append("Moka")
    # removelist_key.append("coffee")
    # removelist_key.append("kopiko")
    # removelist_key.append("package1")
    # removelist_key.append("package3")
    # removelist_key.append("package4")
    # removelist_key.append("package5")
    # removelist_key.append("package10")
    # removelist_key.append("package11")
    # removelist_key.append("package17")
    removelist_key = ["package1"]
    removelist_key.append("package2")
    removelist_key.append("package3")
    removelist_key.append("package4")
    removelist_key.append("package5")
    removelist_key.append("package6")
    removelist_key.append("package7")
    removelist_key.append("package8")
    removelist_key.append("package9")
    removelist_key.append("package10")
    removelist_key.append("package11")
    removelist_key.append("package12")
    removelist_key.append("package13")
    removelist_key.append("package14")
    removelist_key.append("package15")
    removelist_key.append("package16")
    removelist_key.append("package17")
    f_list = os.listdir(AnnotationDir)
    i=0
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == '.xml':
            file_tmp = AnnotationDir+"/"+file_comp4
            treeA=ElementTree()
            treeA.parse(file_tmp)
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            if width==0 or height==0 or depth==0:
                print file_comp4,"width==0 or height==0 or depth==0"
                break
            bfind_one_space = False;
            
            # if JPEG_Dir!="":
            #     im = cv2.imread(JPEG_Dir+os.path.splitext(file_comp4)[0]+".jpg")
            #     #sp = im.shape
            #     imheight = im.shape[0]
            #     imwidth = im.shape[1]
            #     imdepth = im.shape[2]
            #     if imwidth!=width or imheight!=height or imdepth!=depth :
            #         bfind_one_space = True
            #         print file_comp4,"width,height,depth error"
            #         treeA.find('size/width').text = str(imwidth)
            #         treeA.find('size/height').text =str(imheight)
            #         treeA.find('size/depth').text =str(imdepth)
            # anno = treeA.find("annotation")
            # children = anno.getchildren()
            # for child in children:
            #     if child.tag=="object":
            #         if child.find('name').text in removelist_key:
            #             bfind_one_space = True
            #             children.remove(child)

            rootA=treeA.getroot()
            print rootA.tag
            children = rootA.findall('object')
            for obj in children:
                xmlname = obj.find('name').text
                xmlname = xmlname.strip()
                # if xmlname=="Others1" :
                #     i=i+1
                #     print file_comp4
                #     bfind_one_space = True
                #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
                #     obj.find('name').text="others1"
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
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

                if  xmlname in removelist_key:
                    print xmlname
                    bfind_one_space = True
                    name_s = obj.findall('name')
                    pose_s = obj.findall('pose')
                    truncated_s = obj.findall('truncated')
                    difficult_s = obj.findall('difficult')
                    bndbox_s = obj.findall('bndbox')
                    for oobj in name_s:
                    	obj.remove(oobj)
                    for oobj in pose_s:
                    	obj.remove(oobj)
                    for oobj in truncated_s:
                    	obj.remove(oobj)
                    for oobj in difficult_s:
                    	obj.remove(oobj)
                    for oobj in bndbox_s:
	                    # xmin_s = oobj.findall('xmin')
	                    # ymin_s = oobj.findall('ymin')
	                    # xmax_s = oobj.findall('xmax')
	                    # ymax_s = oobj.findall('ymax')
	                    # for ooobj in xmin_s:
	                    # 	oobj.remove(ooobj)
	                    # for ooobj in ymin_s:
	                    # 	oobj.remove(ooobj)
	                    # for ooobj in xmax_s:
	                    # 	oobj.remove(ooobj)
	                    # for ooobj in ymax_s:
	                    # 	oobj.remove(ooobj)
                        obj.remove(oobj)
                    rootA.remove(obj)
            if bfind_one_space==True:
                #print file_comp4
                #print treeA
                treeA.write(outDir+file_comp4, encoding="utf-8",xml_declaration=False)
    print i

def check_Anotation_name_error(AnnotationDir,JPEG_Dir,outDir):
    print "==========="
    f_list = os.listdir(AnnotationDir)
    i=0
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == '.xml':
            file_tmp = AnnotationDir+"/"+file_comp4
            treeA=ElementTree()
            treeA.parse(file_tmp)
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            if width==0 or height==0 or depth==0:
                print file_comp4,"width==0 or height==0 or depth==0"
                break
            bfind_one_space = False;
            
            if JPEG_Dir!="":
                im = cv2.imread(JPEG_Dir+os.path.splitext(file_comp4)[0]+".jpg")
                #sp = im.shape
                imheight = im.shape[0]
                imwidth = im.shape[1]
                imdepth = im.shape[2]
                if imwidth!=width or imheight!=height or imdepth!=depth :
                    bfind_one_space = True
                    print file_comp4,"width,height,depth error"
                    treeA.find('size/width').text = str(imwidth)
                    treeA.find('size/height').text =str(imheight)
                    treeA.find('size/depth').text =str(imdepth)
                    width=imwidth
                    height=imheight

            for obj in treeA.findall('object'):
                xmlname = obj.find('name').text
                xmlname = xmlname.strip()
                # if xmlname=="Others1" :
                #     i=i+1
                #     print file_comp4
                #     bfind_one_space = True
                #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
                #     obj.find('name').text="others1"
                # if xmlname=="Others2" :
                #     i=i+1
                #     print file_comp4
                #     bfind_one_space = True
                #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
                #     obj.find('name').text="others2"
                # if xmlname=="Others3" :
                #     i=i+1
                #     print file_comp4
                #     bfind_one_space = True
                #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
                #     obj.find('name').text="others3"
                # if xmlname=="Others4" :
                #     i=i+1
                #     print file_comp4
                #     bfind_one_space = True
                #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
                #     obj.find('name').text="others4"
                # if xmlname=="Budweiser Beer 330ML Can" :
                #     i=i+1
                #     print file_comp4
                #     bfind_one_space = True
                #     #obj.set("name","Mini Oreo SDW 55g*24  Strawberry")
                #     obj.find('name').text="budweiser30"
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                # if xmin>=xmax or ymin>= ymax or xmin<=0 or ymin <=0 or xmax>=width or ymax>height:
                #     print file_comp4
                if xmin<=0 :
                    bfind_one_space = True
                    obj.find('bndbox').find('xmin').text = str(1)
                if ymin<=0 :
                    bfind_one_space = True
                    obj.find('bndbox').find('ymin').text = str(1)
                if   xmax>= width :
                    bfind_one_space = True
                    obj.find('bndbox').find('xmax').text = str(width-1)
                if   ymax>= height :
                    bfind_one_space = True
                    obj.find('bndbox').find('ymax').text = str(height-1)
                if xmin>=xmax or ymin>= ymax:
                    print file_comp4

            if bfind_one_space==True:
                print file_comp4
                treeA.write(outDir+file_comp4, encoding="utf-8",xml_declaration=False)

    # for node in nodelist:  
    #     for key in kv_map:  
    #         if is_delete:   
    #             if key in node.attrib:  
    #                 del node.attrib[key]  
    #         else:  
    #             node.set(key, kv_map.get(key))  
    print i


def scale_JPEGImages(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations):
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
        
        max_input_side=1000
        min_input_side=600
        # (cols---img_width)   (rows---img_height ) 
        max_side = max(img_height, img_width)
        min_side = min(img_height, img_width)

        max_side_scale = float(max_side) / float(max_input_side)
        min_side_scale = float(min_side) /float( min_input_side)
        max_scale=max(max_side_scale, min_side_scale)

        img_scale = float(1)

        if max_scale > 1:
            img_scale = float(1) / max_scale
        
        scaled_height = int(img_height * img_scale)
        scaled_width = int(img_width * img_scale)

        scaled_img=cv2.resize(im,(scaled_width,scaled_height),interpolation=cv2.INTER_CUBIC) #CV_INTER_NN CV_INTER_LINEAR INTER_AREA INTER_CUBIC
        cv2.imwrite(outDir_JPEGImages+"/"+file_comp4, scaled_img)

        basename = os.path.splitext(file_comp4)[0]
        file_xml = AnnotationsDir+"/"+basename+".xml"
        treeA=ElementTree()
        treeA.parse(file_xml)
        width = int(treeA.find('size/width').text)
        height = int(treeA.find('size/height').text)
        depth = int(treeA.find('size/depth').text)
        if width==0 or height==0 or depth==0:
            print file_comp4,"width==0 or height==0 or depth==0"
        treeA.find('size/width').text = str(scaled_width)
        treeA.find('size/height').text = str(scaled_height)
        #treeA.find('size/depth').text = str(img_depth)
        
        bfind_one_space = False;
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
            xmin = int(xmin * img_scale)
            ymin = int(ymin * img_scale)
            xmax = int(xmax * img_scale)
            ymax = int(ymax * img_scale)
            obj.find('bndbox').find('xmin').text = str(xmin)
            obj.find('bndbox').find('ymin').text = str(ymin)
            obj.find('bndbox').find('xmax').text = str(xmax)
            obj.find('bndbox').find('ymax').text = str(ymax)
            # if xmin>=xmax or ymin>= ymax or xmin<=0 or ymin <=0 or xmax>=width or ymax>height:
            #     print file_comp4
            if xmin<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('xmin').text = str(1)
            if ymin<=0 :
                bfind_one_space = True
                obj.find('bndbox').find('ymin').text = str(1)
            if   xmax>= width :
                bfind_one_space = True
                obj.find('bndbox').find('xmax').text = str(width-1)
            if   ymax>= height :
                bfind_one_space = True
                obj.find('bndbox').find('ymax').text = str(height-1)
            if xmin>=xmax or ymin>= ymax:
                print file_comp4

        i+=1 
        print file_comp4
        treeA.write(outDir_Annotations+"/"+basename+".xml", encoding="utf-8",xml_declaration=False)
    print "num of image:",str(i)

def cut_4_xml(txtDir,AnnotationDir,ovthresh,outDir_Annotations):
    f_list = os.listdir(txtDir)
    # print f_list
    for file_comp4 in f_list:
        print file_comp4
        if os.path.splitext(file_comp4)[1] == '.txt':
            basename = os.path.splitext(file_comp4)[0]
            #spname = basename.split("comp4_92df78d2-ec23-4267-9e11-6a15a184d89d_det_test_")[1]
            #print spname
            file_tmp = txtDir+"/"+file_comp4
            with open(file_tmp,"r") as f:
                lines = f.readlines()
                #lines = lines.strip("\n")
                #lines = lines.strip("\r")
                splitlines = [x.strip().split(' ') for x in lines]
                #print splitlines
                Nametxt =  [x[0] for x in splitlines]
                width =    [int(float(x[1])) for x in splitlines]
                height =   [int(float(x[2])) for x in splitlines]
                x1txt =    [int(float(x[3])) for x in splitlines]
                y1txt =    [int(float(x[4])) for x in splitlines]
                # x2txt = x1txt + width
                # y2txt = y1txt + height
                # print width
                # print height
                # print x1txt
                # print y1txt
                # print x2txt
                # print y2txt
                # x2txt =    [x[5] for x in splitlines]
                # y2txt =    [x[6] for x in splitlines]
                bbfromtxt = np.array([[float(z) for z in x[3:]] for x in splitlines])
                #recs = {}
                #recs[basename] 
                baseInfo,objects = parse_xml(AnnotationDir+"/"+basename+".xml")
                for i,x1 in enumerate(x1txt):
                    #xml====================================
                    four_root = ElementTree()
                    A1 = create_node('annotation',{},"")
                    four_root._setroot(A1)
                    B1 = create_node('folder',{},"2")
                    B2 = create_node('filename',{},Nametxt[i])
                    B3 = create_node('path',{},"2")
                    A1.append(B1)
                    A1.append(B2)
                    A1.append(B3)
                    # SubElement(A1,"folder").text=str(width[i])
                    # SubElement(A1,"filename").text=str(height[i])
                    # SubElement(A1,"path").text="3"
                    B4 = create_node('source',{},"")
                    A1.append(B4)
                    C1 = create_node('database',{},"Unknown")
                    B4.append(C1)
                    B5 = create_node('size',{},"")
                    SubElement(B5,"width").text=str(width[i])
                    SubElement(B5,"height").text=str(height[i])
                    SubElement(B5,"depth").text="3"
                    A1.append(B5)
                    # D1 = create_node('width',{},str(width[i]))
                    # B5.append(D1)
                    # D2 = create_node('height',{},str(height[i]))
                    # B5.append(D2)
                    # D3 = create_node('depth',{},str(3))
                    # B5.append(D3)
                    B6 = create_node('segmented',{},"0")
                    A1.append(B6)
                    #xml====================================

                    for obj in objects:
                        bbox = obj['bbox']
                        ixmin = np.maximum(bbox[ 0], x1txt[i])
                        iymin = np.maximum(bbox[ 1], y1txt[i])
                        ixmax = np.minimum(bbox[ 2], (x1txt[i]+width[i]))
                        iymax = np.minimum(bbox[ 3], (y1txt[i]+height[i]))

                        if ixmin<=x1txt[i] :
                            ixmin = x1txt[i]+1
                        if iymin<=y1txt[i] :
                            iymin = y1txt[i]+1
                        if   ixmax>= (x1txt[i]+width[i]) :
                            ixmax = (x1txt[i]+width[i])-1
                        if   iymax>= (y1txt[i]+height[i]) :
                            iymax = (y1txt[i]+height[i])-1
                        # if ixmin>=ixmax or iymin>= iymax:
                        #      print ixmin,iymin,ixmax,iymax,"==",bbox[ 0],bbox[ 1],bbox[ 2],bbox[ 3],"=====",x1txt[i],y1txt[i],(x1txt[i]+width[i]),(y1txt[i]+height[i])
                        overArea = 0
                        if ixmax>ixmin and iymax>iymin:
                            #print bbox[ 0],bbox[ 1],bbox[ 2],bbox[ 3],"==================================="
                            iw = np.maximum(ixmax - ixmin + 1., 0.)
                            ih = np.maximum(iymax - iymin + 1., 0.)
                            inters = iw * ih

                            # union
                            uni = (bbox[ 2] - bbox[ 0] + 1.) *(bbox[3] - bbox[ 1] + 1.) 

                            overlaps = inters / uni
                            #ovmax = np.max(overlaps)
                            #jmax = np.argmax(overlaps)
                            if overlaps > ovthresh :
                                #xml====================================
                                BBobj = create_node('object',{},"")
                                SubElement(BBobj,"name").text=obj['name']
                                SubElement(BBobj,"pose").text=obj['pose']
                                SubElement(BBobj,"truncated").text=obj['truncated']
                                SubElement(BBobj,"difficult").text=obj['difficult']
                                child5 = SubElement(BBobj,"bndbox")
                                # child1= create_node('name',{},obj['name'])
                                # child2= create_node('pose',{},obj['pose'])
                                # child3= create_node('truncated',{},obj['truncated'])
                                # child4= create_node('difficult',{},obj['difficult'])
                                # child5= create_node('bndbox',{},"")
                                # BBobj.append(child1)
                                # BBobj.append(child2)
                                # BBobj.append(child3)
                                # BBobj.append(child4)
                                # BBobj.append(child5)
                                SubElement(child5,"xmin").text=str(ixmin-x1txt[i])
                                SubElement(child5,"ymin").text=str(iymin-y1txt[i])
                                SubElement(child5,"xmax").text=str(ixmax-x1txt[i])
                                SubElement(child5,"ymax").text=str(iymax-y1txt[i])
                                A1.append(BBobj)
                                # childbbox1= create_node('xmin',{},str(ixmin))
                                # child5.append(childbbox1)
                                # childbbox2= create_node('ymin',{},str(iymin))
                                # child5.append(childbbox2)
                                # childbbox3= create_node('xmax',{},str(ixmax))
                                # child5.append(childbbox3)
                                # childbbox4= create_node('ymax',{},str(iymax))
                                # child5.append(childbbox4)
                                #xml====================================
                    four_root.write(outDir_Annotations+"/"+Nametxt[i]+".xml", encoding="utf-8",xml_declaration=False)

def IOU(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    ixmin = np.maximum(x1, xmin)
    iymin = np.maximum(y1, ymin)
    ixmax = np.minimum(x2, xmax)
    iymax = np.minimum(y2, ymax)
    recArea1 = (x2 - x1) * (y2 - y1)
    recArea2 = (xmax - xmin) * (ymax - ymin)
    overlapArea = 0
    areaRate = 0
    if ixmax>ixmin and iymax>iymin:
        #print x1, y1, x2, y2, xmin, ymin, xmax, ymax, ixmin, iymin, ixmax, iymax
        overlapArea = (ixmax - ixmin)*(iymax - iymin)
        areaRate = float(overlapArea) / float(recArea1 + recArea2 - overlapArea)
    else:  
        areaRate = 0
    #print "areaRate= ",areaRate
    return areaRate

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def GoodEval(resultfile,Anotations_Dir,SKUfile,test_filelist,cachedir):
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
    print labelmap
    with open(test_filelist,"r") as test_filelist_f:
        test_filelist_lines = test_filelist_f.readlines()
    # gt_obj = []
    # for i_file,test_file in enumerate(test_filelist_lines):
    #     obj_struct = {}
    #     obj_struct["objs"] = parse_rec(Anotations_Dir+test_file+".xml")
    #     obj_struct["filename"] = test_file
    #     gt_obj.append(obj_struct)
    # gt_obj = {}
    # for i_file,test_file in enumerate(test_filelist_lines):
    #     gt_obj[test_file]=parse_rec(Anotations_Dir+test_file+".xml")
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'GoodEval_annots.pkl')
    if not os.path.isfile(cachefile):
        # load annots
        gt_obj = {}
        for i_file,test_file in enumerate(test_filelist_lines):
            print test_file
            temp_file = test_file.strip().strip('\n').strip('\r')
            if temp_file=="" or temp_file=='':
                continue
            gt_obj[temp_file]=parse_rec(Anotations_Dir+temp_file+".xml")
            if i_file % 50 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i_file + 1, len(test_filelist_lines))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(gt_obj, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            gt_obj = cPickle.load(f)

    with open(resultfile,"r") as f:
        lines = f.readlines()
        #lines = lines.strip("\n")
        #lines = lines.strip("\r")
    splitlines = [x.strip().split(' ') for x in lines]
    #Nametxt labelIDtxt width height x1txt y1txt x2txt y2txt
    Nametxt =  [x[0] for x in splitlines]
    labelIDtxt =    [int(x[1]) for x in splitlines]
    width =   [int(float(x[2])) for x in splitlines]
    height =    [int(float(x[3])) for x in splitlines]
    x1txt =    [int(float(x[4])) for x in splitlines]
    y1txt =    [int(float(x[5])) for x in splitlines]
    x2txt =    [int(float(x[6])) for x in splitlines]
    y2txt =    [int(float(x[7])) for x in splitlines]   
    r = 0
    rsum = 0;
    for i_file,test_file in enumerate(test_filelist_lines):
        temp_file = test_file.strip().strip('\n').strip('\r')
        if temp_file=="" or temp_file=='':
            continue
        obj_struc=gt_obj[temp_file]
    #for i,obj_struc in enumerate(gt_obj):
        for j,objcts in enumerate(obj_struc):
            tmp_name = objcts['name']
            tmp_name=tmp_name.lower() #tmp_name.find('other')<0 and
            if  objcts['name'] != 'Others' :
                #print objcts['name']
                rsum = rsum + 1
                label_recall_all[labelmap[objcts['name']]-1] +=1
                for i_Nametxt, indexname in enumerate(Nametxt):
                    if temp_file == indexname  and labelIDtxt[i_Nametxt] == labelmap[objcts['name']]:
                        #print temp_file,indexname,str(labelIDtxt[i_Nametxt] ),str(labelmap[objcts['name']])
                        areaRate = IOU(x1txt[i_Nametxt],y1txt[i_Nametxt],x2txt[i_Nametxt],y2txt[i_Nametxt],
                                       objcts['bbox'][0],objcts['bbox'][1],objcts['bbox'][2],objcts['bbox'][3])
                        #print "areaRate= ",areaRate
                        if areaRate >= 0.5:
                            r = r +1
                            label_recall_count[labelmap[objcts['name']]-1] +=1
    p = 0
    psum = 0
    for i_pNametxt, pindexname in enumerate(Nametxt):
        tmp_name = labellist[labelIDtxt[i_pNametxt]-1]
        tmp_name=tmp_name.lower() #tmp_name.find('other')<0 and
        if   labellist[labelIDtxt[i_pNametxt]-1] !='Others':
            psum = psum + 1
            for j_p,objcts_p in enumerate(gt_obj[pindexname]):
                if labelmap[objcts_p["name"]] == labelIDtxt[i_pNametxt]:
                    areaRate_p = IOU(x1txt[i_pNametxt],y1txt[i_pNametxt],x2txt[i_pNametxt],y2txt[i_pNametxt],
                                objcts_p['bbox'][0],objcts_p['bbox'][1],objcts_p['bbox'][2],objcts_p['bbox'][3])
                    if areaRate_p >= 0.5:
                        p = p+1
    P = float(p)/float(psum)
    R = float(r)/float(rsum)
    for ii,ss in enumerate(label_recall_all):
        class_rec = float(label_recall_count[ii])/float(label_recall_all[ii]) if label_recall_all[ii]!=0 else 0
        print labellist[ii],"class_rec =",class_rec,"   {}/{} ".format(label_recall_count[ii],label_recall_all[ii])
    print " psum=",psum," p=",p," rsum=",rsum," r=",r
    F1 = float(2*P*R)/float(P + R)
    print "              Precision=",P," Recall=",R," F1-measure=",F1
    #f_list = os.listdir(txtDir)
    # print f_list
    # for file_comp4 in f_list:
    #     print file_comp4
    #     if os.path.splitext(file_comp4)[1] == '.txt':
    #         basename = os.path.splitext(file_comp4)[0]

def tianruo_GoodEval(resultfile,Anotations_Dir,SKUfile,test_filelist,cachedir,cachefilename,threhold=293):
    with open(SKUfile,"r") as SKUfile_f:
        SKUfile_lines = SKUfile_f.readlines()
    labelmap={}
    labellist=[]
    label_recall_all=[]
    label_recall_count=[]
    label_detect_wrong=[]


    label_precision_all=[]
    label_precision_count=[]
    for ilable,labelname in enumerate(SKUfile_lines):
        labelname = labelname.strip().strip('\n').strip('\r')
        #print labelname
        labelmap[labelname] = ilable+1
        labellist.append(labelname)
        label_recall_all.append(0)
        label_recall_count.append(0)
        label_detect_wrong.append(0)
        label_precision_all.append(0)
        label_precision_count.append(0)
    print labelmap
    with open(test_filelist,"r") as test_filelist_f:
        test_filelist_lines = test_filelist_f.readlines()
    # gt_obj = []
    # for i_file,test_file in enumerate(test_filelist_lines):
    #     obj_struct = {}
    #     obj_struct["objs"] = parse_rec(Anotations_Dir+test_file+".xml")
    #     obj_struct["filename"] = test_file
    #     gt_obj.append(obj_struct)
    # gt_obj = {}
    # for i_file,test_file in enumerate(test_filelist_lines):
    #     gt_obj[test_file]=parse_rec(Anotations_Dir+test_file+".xml")
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, cachefilename)
    if not os.path.isfile(cachefile):
        # load annots
        gt_obj = {}
        for i_file,test_file in enumerate(test_filelist_lines):
            print test_file
            temp_file = test_file.strip().strip('\n').strip('\r')
            if temp_file=="" or temp_file=='':
                print temp_file
                continue
            gt_obj[temp_file]=parse_rec(Anotations_Dir+temp_file+".xml")
            if i_file % 50 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i_file + 1, len(test_filelist_lines))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(gt_obj, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            gt_obj = cPickle.load(f)
    f_badpic = open(os.path.join(cachedir, 'badpic_namelist.txt'),"w")#2018_1_11
    f_badclass = open(os.path.join(cachedir, 'bad_class.txt'),"w")#2018_1_11
    f_badenoughtclass = open(os.path.join(cachedir, 'f_bad_enought_class.txt'),"w")#2018_1_11
    with open(resultfile,"r") as f:
        lines = f.readlines()
        #lines = lines.strip("\n")
        #lines = lines.strip("\r")
    splitlines = [x.strip().split(' ') for x in lines]
    #Nametxt labelIDtxt width height x1txt y1txt x2txt y2txt
    Nametxt =  [x[0] for x in splitlines]
    labelIDtxt =    [int(x[1]) for x in splitlines]
    width =   [int(float(x[2])) for x in splitlines]
    height =    [int(float(x[3])) for x in splitlines]
    x1txt =    [int(float(x[4])) for x in splitlines]
    y1txt =    [int(float(x[5])) for x in splitlines]
    x2txt =    [int(float(x[6])) for x in splitlines]
    y2txt =    [int(float(x[7])) for x in splitlines]
    r = 0
    rsum = 0;
    detect_wrong_whole = 0
    tianluo_dic={}
    for i_file,test_file in enumerate(test_filelist_lines):
        temp_file = test_file.strip().strip('\n').strip('\r')
        tianluo_dic[temp_file] = 0
        tianluo_dic[temp_file+'_sum_'] = 0
        if temp_file=="" or temp_file=='':
            continue
        obj_struc=gt_obj[temp_file]
    #for i,obj_struc in enumerate(gt_obj):
        for j,objcts in enumerate(obj_struc):
            tmp_name = objcts['name']
            tmp_name=tmp_name.lower() #tmp_name.find('other')<0 and
            if tmp_name.find('other')>=0:
                #print temp_file,"===== ",tmp_name
                continue
            if labelmap[objcts['name']]>=threhold:
                #print temp_file,"===== ",tmp_name
                continue
            # if tmp_name.find('colgate21')>=0 or tmp_name.find('colgate22')>=0 or tmp_name.find('colgate23')>=0 or tmp_name.find('colgate24')>=0 or tmp_name.find('colgate25')>=0:
            #     continue
            if  objcts['name'] != 'Others' :
                #print objcts['name']
                rsum = rsum + 1
                tianluo_dic[temp_file+'_sum_']+=1
                #print temp_file,objcts['name']
                label_recall_all[labelmap[objcts['name']]-1] +=1
                for i_Nametxt, indexname in enumerate(Nametxt):
                    # if temp_file == indexname:
                    #     #print temp_file,indexname,str(labelIDtxt[i_Nametxt] ),str(labelmap[objcts['name']])
                    #     areaRate = IOU(x1txt[i_Nametxt],y1txt[i_Nametxt],x2txt[i_Nametxt],y2txt[i_Nametxt],
                    #                    objcts['bbox'][0],objcts['bbox'][1],objcts['bbox'][2],objcts['bbox'][3])
                    #     #print "areaRate= ",areaRate
                    #     if areaRate >= 0.5:
                    #         if  labelIDtxt[i_Nametxt] == labelmap[objcts['name']]:
                    #             r = r +1
                    #             label_recall_count[labelmap[objcts['name']]-1] +=1
                    #         else:
                    #             label_detect_wrong[labelmap[objcts['name']]-1] +=1
                    #             detect_wrong_whole+=1
                    if temp_file == indexname  and labelIDtxt[i_Nametxt] == labelmap[objcts['name']]:
                        #print temp_file,indexname,str(labelIDtxt[i_Nametxt] ),str(labelmap[objcts['name']])
                        areaRate = IOU(x1txt[i_Nametxt],y1txt[i_Nametxt],x2txt[i_Nametxt],y2txt[i_Nametxt],
                                       objcts['bbox'][0],objcts['bbox'][1],objcts['bbox'][2],objcts['bbox'][3])
                        #print "areaRate= ",areaRate
                        if areaRate >= 0.5:
                            r = r +1
                            label_recall_count[labelmap[objcts['name']]-1] +=1
                            tianluo_dic[temp_file]+=1
    p = 0
    psum = 0
    tianluo_p = 0
    tianluo_psum = 0
    name_w = ''
    every_pic_tianluo = 0
    tianluo_psum = 0
    #pic_tianluo = {}
    for i_pNametxt, pindexname in enumerate(Nametxt):
        tmp_name = labellist[labelIDtxt[i_pNametxt]-1]
        tmp_name=tmp_name.lower() #tmp_name.find('other')<0 and
        if name_w!=pindexname or i_pNametxt==len(Nametxt)-1:
            if name_w!='' :
                tmp_rec = float(tianluo_dic[name_w]-tianluo_psum+every_pic_tianluo)/float(tianluo_dic[name_w+'_sum_']) if tianluo_dic[name_w+'_sum_']!=0 else 0
                print name_w,"    tianruo_prec= ",tmp_rec," ({}-{}+{}/{})".format(tianluo_dic[name_w],tianluo_psum,every_pic_tianluo,tianluo_dic[name_w+'_sum_'])
                if tmp_rec<0.75:##2018_1_11
                    f_badpic.writelines(name_w+'\n')
            name_w = pindexname
            every_pic_tianluo = 0
            tianluo_psum = 0
        if tmp_name.find('other')>=0:
            #print Nametxt[i_pNametxt],"===== ",tmp_name
            continue
        if labelIDtxt[i_pNametxt]>=threhold:
            #print temp_file,"===== ",tmp_name
            continue
        if   labellist[labelIDtxt[i_pNametxt]-1] !='Others':
            psum = psum + 1
            tianluo_psum = tianluo_psum+1
            label_precision_all[labelIDtxt[i_pNametxt]-1] +=1
            for j_p,objcts_p in enumerate(gt_obj[pindexname]):
                #if labelmap[objcts_p["name"]] == labelIDtxt[i_pNametxt]:
                areaRate_p = IOU(x1txt[i_pNametxt],y1txt[i_pNametxt],x2txt[i_pNametxt],y2txt[i_pNametxt],
                            objcts_p['bbox'][0],objcts_p['bbox'][1],objcts_p['bbox'][2],objcts_p['bbox'][3])
                if areaRate_p >= 0.5:
                    #every_pic_tianluo =every_pic_tianluo+1
                    tianluo_p=tianluo_p+1
                    every_pic_tianluo =every_pic_tianluo+1
                    tmp_name1 =objcts_p['name']
                    # if tmp_name1.find('colgate21')>=0 or tmp_name1.find('colgate22')>=0 or tmp_name1.find('colgate23')>=0 or tmp_name1.find('colgate24')>=0 or tmp_name1.find('colgate25')>=0:
                    #     continue
                    if labelmap[objcts_p["name"]] == labelIDtxt[i_pNametxt]:
                        p = p+1
                        label_precision_count[labelIDtxt[i_pNametxt]-1] +=1
    P = float(p)/float(psum)
    R = float(r)/float(rsum)
    tianruo_prec = float(r-psum+tianluo_p)/float(rsum)
    for ii,ss in enumerate(label_recall_all):
        class_rec = float(label_recall_count[ii])/float(label_recall_all[ii]) if label_recall_all[ii]!=0 else 0
        class_precision = float(label_precision_count[ii])/float(label_precision_all[ii]) if label_precision_all[ii]!=0 else 0
        print labellist[ii],"class_rec =",class_rec,"   {}/{} ".format(label_recall_count[ii],label_recall_all[ii])," class_precision =",class_precision,"   {}/{} ".format(label_precision_count[ii],label_precision_all[ii])
        if class_rec<0.8 or class_precision<0.8:##2018_1_11
            bad_class_string=labellist[ii]+"class_rec ="+str(class_rec)+"   {}/{} ".format(label_recall_count[ii],label_recall_all[ii])+" class_precision ="+str(class_precision)+"   {}/{} ".format(label_precision_count[ii],label_precision_all[ii])
            f_badclass.writelines(labellist[ii]+'\n')
            if label_recall_all[ii]>40:
                f_badenoughtclass.writelines(labellist[ii]+'\n')
    print " psum=",psum," p=",p,"  tianluo_p=",tianluo_p," rsum=",rsum," r=",r
    #print " psum=",psum," p=",p,"  tianluo_p=",tianluo_p," rsum=",rsum," r=",r,"  detect_wrong_whole=",detect_wrong_whole
    F1 = float(2*P*R)/float(P + R)
    print "              Precision=",P," Recall=",R," F1-measure=",F1,"  r-psum+tianluo_p/rsum tianruo_prec=",tianruo_prec," ({}/{})".format(r-psum+tianluo_p,rsum)
    f_badpic.close()
    f_badclass.close()
    f_badenoughtclass.close()


# def parse_labelmap(imagesetfile,imagesetfile1,imagesetfile2):

#     i=0
#     labelsum="\'__background__\'"
#     labelproto='item{\n\tname: "%s"\n\tlabel: %d\n\tdisplay_name: "%s"\n}\n'%("none_of_the_above",i,"background")
#     with open(imagesetfile, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#               i+=1
#               x=line.strip('\n')
#               #x=x.strip('\t')
#               #x=x.strip('\t')
#               #x=x.strip('\t')
#               x=x.replace("\r", "")
#               #x=x.replace("\t", "")
#               #x=x.replace("\n\t", "")
#               #x=x.replace(" ", "\ ")
#               tmp = ',\"'+x+'\"'
#               labelsum+=tmp
#               labelproto += 'item{\n\tname: "%s"\n\tlabel: %d\n\tdisplay_name: "%s"\n}\n'%(x,i,x)
#     with open(imagesetfile1, 'wb') as ff:
#         ff.write(labelsum)
#     print labelsum
#     with open(imagesetfile2, 'wb') as ff:
#         ff.write(labelproto)
#     return labelsum
def parse_labelmap(imagesetfile,basedir,imagesetfile1,imagesetfile2):

    i=1
    labelsum="\"__background__\""
    labelproto='item{\n\tname: "%s"\n\tlabel: %d\n\tdisplay_name: "%s"\n}\n'%("none_of_the_above",i,"background")
    with open(imagesetfile, 'r') as f:
        lines_semi = f.readlines()
    
    lines = []
    [lines.append(j) for j in lines_semi if not j in lines]    
    for line in lines:
        i += 1
        x=line.strip('\n')
              #x=x.strip('\t')
              #x=x.strip('\t')
              #x=x.strip('\t')
        x=x.replace("\r", "")
              #x=x.replace("\t", "")
              #x=x.replace("\n\t", "")
              #x=x.replace(" ", "\ ")
        tmp = ',\"'+x+'\"'
        labelsum+=tmp
        labelproto += 'item{\n\tname: "%s"\n\tlabel: %d\n\tdisplay_name: "%s"\n}\n'%(x,i,x)
    with open(basedir+str(i)+"_"+str((i)*49)+"_"+str((i)*196)+"_"+imagesetfile1, 'wb') as ff:
        ff.write(labelsum)
    print labelsum
    print i
    with open(basedir+str(i+1)+"_"+imagesetfile2, 'wb') as ff:
        ff.write(labelproto)
    return i

def randomAlloc(Annotations_Dir,out_ImageSets_Dir):
    f_list = os.listdir(Annotations_Dir)
    file_list = [os.path.splitext(x)[0] for x in f_list]
    #print file_list
    # for file_comp4 in f_list:
    # 	filename = file_comp4.splitext()[0]
    #     print file_comp4
    i_whole = len(f_list)
    file_index = range(0,i_whole)
    trainval_tmp = random.sample(file_index,int(math.floor(float(10)*float(i_whole)/float(11))))
    test_tmp = list(set(file_index).difference(set(trainval_tmp)))
    trainval=[file_list[x] for x in trainval_tmp]
    test=[file_list[x] for x in test_tmp]


    i_trainval = len(trainval)
    trainval_index = range(0,i_trainval)
    train_tmp = random.sample(trainval_index,int(math.floor(float(2)*float(i_trainval)/float(3))))
    val_tmp = list(set(trainval_index).difference(set(train_tmp)))
    train=[trainval[x] for x in train_tmp]
    val=[trainval[x] for x in val_tmp]

    trainval = [(x.lower(),x) for x in trainval]
    #trainval.sort()
    trainval = [x[1]+"\n" for x in trainval]
    test = [(x.lower(),x) for x in test]
    #test.sort()
    test = [x[1]+"\n" for x in test]
    train = [(x.lower(),x) for x in train]
    #train.sort()
    train = [x[1]+"\n" for x in train]
    val = [(x.lower(),x) for x in val]
    #val.sort()
    val = [x[1]+"\n" for x in val]
    print test
    with open(out_ImageSets_Dir+"trainval.txt","w") as f_trainval:
    	f_trainval.writelines(trainval)
    with open(out_ImageSets_Dir+"test.txt","w") as f_test:
    	f_test.writelines(test)
    with open(out_ImageSets_Dir+"train.txt","w") as f_train:
    	f_train.writelines(train)
    with open(out_ImageSets_Dir+"val.txt","w") as f_val:
    	f_val.writelines(val)

def copy_cut_4_traivaltxt(orignal_Anotations,output_Anotations):
    # testfile = orignal_Anotations + "test.txt"#/mnt/storage/liushuai/data/cookie/cookieproj1/ImageSets/Main/
    # trainfile = orignal_Anotations + "train.txt"
    # trainvalfile = orignal_Anotations + "trainval.txt"
    # valfile = orignal_Anotations + "val.txt"
    # output_testfile = output_Anotations + "test.txt"#/mnt/storage/liushuai/data/cookie/cookieproj1/ImageSets/Main/
    # output_trainfile = output_Anotations + "train.txt"
    # output_trainvalfile = output_Anotations + "trainval.txt"
    # output_valfile = output_Anotations + "val.txt"
    # output_test_cut_file = output_Anotations + "test_cut_merge.txt"
    # with open(testfile,'r') as f:
    #     lines = f.readlines()
    #     outputlist = []
    #     outputlist_nomerge = []
    #     for line in lines:
    #         filename = line.strip().strip('\n').strip('\r')
    #         outputlist.append(filename+"\n")
    #         outputlist.append(filename+"_1\n")
    #         outputlist.append(filename+"_2\n")
    #         outputlist.append(filename+"_3\n")
    #         outputlist.append(filename+"_4\n")

    #         outputlist_nomerge.append(filename+"_1\n")
    #         outputlist_nomerge.append(filename+"_2\n")
    #         outputlist_nomerge.append(filename+"_3\n")
    #         outputlist_nomerge.append(filename+"_4\n")
    #     with open(output_testfile,'w') as out_f:
    #         out_f.writelines(outputlist_nomerge)
    #     with open(output_test_cut_file,'w') as out_f:
    #         out_f.writelines(outputlist)

    # with open(trainvalfile,'r') as f:
    #     lines = f.readlines()
    #     outputlist = []
    #     for line in lines:
    #         filename = line.strip().strip('\n').strip('\r')
    #         #outputlist.append(filename+"\n")
    #         outputlist.append(filename+"_1\n")
    #         outputlist.append(filename+"_2\n")
    #         outputlist.append(filename+"_3\n")
    #         outputlist.append(filename+"_4\n")
    #     with open(output_trainvalfile,'w') as out_f:
    #         out_f.writelines(outputlist) 

    file_262 = "/mnt/storage/liushuai/RFCN/R_fcn_bin/m10/m10.txt";
    output_262_cut_file = "/mnt/storage/liushuai/RFCN/R_fcn_bin/m60_cut/m60_cut.txt";
    with open(file_262,'r') as f:
        lines = f.readlines()
        outputlist = []
        for line in lines:
            filename = line.strip().strip('\n').strip('\r')
            #outputlist.append(filename)
            outputlist.append(filename+"_1\n")
            outputlist.append(filename+"_2\n")
            outputlist.append(filename+"_3\n")
            outputlist.append(filename+"_4\n")
        with open(output_262_cut_file,'w') as out_f:
            out_f.writelines(outputlist)

def celect_Dir_2_filelist(AnnotationsDir,outputfile):
    f_list = os.listdir(AnnotationsDir)
    file_list = [os.path.splitext(x)[0]+'\n' for x in f_list if os.path.splitext(x)[1]==".xml"]
    with open(outputfile,'w') as f:
        f.writelines(file_list)

def made_rect_JPEGImages_from_SKUfile(SKUfile,test_filelist,JPEGImagesDir,AnnotationsDir,outDir_JPEGImages):
    
    with open(SKUfile,"r") as SKUfile_f:
        SKUfile_lines = SKUfile_f.readlines()
    labelmap={}
    # labellist=[]
    # label_recall_all=[]
    # label_recall_count=[]
    # label_detect_wrong=[]
    for ilable,labelname in enumerate(SKUfile_lines):
        labelname = labelname.strip().strip('\n').strip('\r')
        labelmap[labelname] = ilable+1
        # labellist.append(labelname)
        # label_recall_all.append(0)
        # label_recall_count.append(0)
        # label_detect_wrong.append(0)
    k= 0
    with open(test_filelist,"r") as test_filelist_f:
        test_filelist_lines = test_filelist_f.readlines()
        for i_file,test_file in enumerate(test_filelist_lines):
            #print test_file
            temp_file = test_file.strip().strip('\n').strip('\r')
            im = cv2.imread(JPEGImagesDir+"/"+temp_file+".jpg")#jpegpath.format(imagename)
            #sp = im.shape
            img_height = im.shape[0]
            img_width = im.shape[1]
            img_depth = im.shape[2]
            #file_xml = 
            treeA=ElementTree()
            treeA.parse(AnnotationsDir+"/"+temp_file+".xml")
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            if width==0 or height==0 or depth==0:
                print temp_file,"width==0 or height==0 or depth==0"
            # treeA.find('size/width').text = str(scaled_width)
            # treeA.find('size/height').text = str(scaled_height)
            #treeA.find('size/depth').text = str(img_depth)
            
            bfind_one_space = False;
            for obj in treeA.findall('object'):
                if labelmap[obj.find('name').text] == 100:
                    continue
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)

                cv2.rectangle(im, (xmin,ymin), (xmax,ymax), (0,0,233),3)
                font = 1
                if width<600:
                    font = 1
                elif 600<=width<1200:
                    font = 2
                elif 1200<=width<1800:
                    font = 3
                elif 1800<=width<2400:
                    font = 4
                else:
                    font = 5

                cv2.putText(im,str(labelmap[obj.find('name').text]), (xmin,ymin),0,font, (0, 0, 255),3)

            k+=1
            print str(k),"  ",temp_file
            cv2.imwrite(outDir_JPEGImages+"/"+temp_file+".jpg", im)

def made_rect_JPEGImages(SKUfile,JPEGImagesDir,AnnotationsDir,outDir_JPEGImages):
    print "==========="

    with open(SKUfile,"r") as SKUfile_f:
        SKUfile_lines = SKUfile_f.readlines()
    labelmap={}
    # labellist=[]
    # label_recall_all=[]
    # label_recall_count=[]
    # label_detect_wrong=[]
    for ilable,labelname in enumerate(SKUfile_lines):
        labelname = labelname.strip().strip('\n').strip('\r')
        labelmap[labelname] = ilable+1

    print "suntoryv4  ",labelmap["suntoryv4"]
    print "suntoryv5  ",labelmap["suntoryv5"]

    f_list = os.listdir(JPEGImagesDir)
    i=0
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == ".xml":
            continue
        im = cv2.imread(JPEGImagesDir+"/"+file_comp4)#jpegpath.format(imagename)
        #sp = im.shape
        img_height = im.shape[0]
        img_width = im.shape[1]
        img_depth = im.shape[2]
        if img_depth != 3 or img_height<=0 or img_depth<=0 :
             print "depet wrong",file_comp4
        
        basename = os.path.splitext(file_comp4)[0]
        file_xml = AnnotationsDir+"/"+basename+".xml"
        treeA=ElementTree()
        treeA.parse(file_xml)
        width = int(treeA.find('size/width').text)
        height = int(treeA.find('size/height').text)
        depth = int(treeA.find('size/depth').text)
        if width==0 or height==0 or depth==0:
            print file_comp4,"width==0 or height==0 or depth==0"
        # treeA.find('size/width').text = str(scaled_width)
        # treeA.find('size/height').text = str(scaled_height)
        #treeA.find('size/depth').text = str(img_depth)
        
        bfind_one_space = False;
        for obj in treeA.findall('object'):
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            cv2.rectangle(im, (xmin,ymin), (xmax,ymax), (255,0,0),3)
            font = 1
            if width<600:
                font = 1
            elif 600<=width<1200:
                font = 2
            elif 1200<=width<1800:
                font = 3
            elif 1800<=width<2400:
                font = 4
            else:
                font = 5
            if obj.find('name').text == "miss" or obj.find('name').text=="origin" or obj.find('name').text=="others":
                print_name=obj.find('name').text 
            else:
                print_name=str(labelmap[obj.find('name').text])
            #print_name=obj.find('name').text 
            cv2.putText(im,print_name, (xmin,ymin),0,font, (0, 0, 255),3)

        i+=1 
        print file_comp4
        cv2.imwrite(outDir_JPEGImages+"/"+file_comp4, im)
        #treeA.write(outDir_Annotations+"/"+basename+".xml", encoding="utf-8",xml_declaration=False)
    print "num of image:",str(i)
 

def cut_Annotation_roi_from_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages):
    print "==========="
    f_list = os.listdir(JPEGImagesDir)
    i=0
    for file_comp4 in f_list:
        tmpname = file_comp4.strip().strip('\n').strip('\r')
        numstr = os.path.splitext(tmpname)[0].split("colgate")[1]
        inum = int(numstr)
        if inum<1000:
            continue
        im = cv2.imread(JPEGImagesDir+"/"+file_comp4)#jpegpath.format(imagename)
        #sp = im.shape
        img_height = im.shape[0]
        img_width = im.shape[1]
        img_depth = im.shape[2]
        if img_depth != 3 or img_height<=0 or img_depth<=0 :
             print "depet wrong",file_comp4
        
        basename = os.path.splitext(file_comp4)[0]
        file_xml = AnnotationsDir+"/"+basename+".xml"
        treeA=ElementTree()
        treeA.parse(file_xml)
        width = int(treeA.find('size/width').text)
        height = int(treeA.find('size/height').text)
        depth = int(treeA.find('size/depth').text)
        if width==0 or height==0 or depth==0:
            print file_comp4,"width==0 or height==0 or depth==0"
        # treeA.find('size/width').text = str(scaled_width)
        # treeA.find('size/height').text = str(scaled_height)
        #treeA.find('size/depth').text = str(img_depth)
        mkdir_tmp = treeA.find('size/width').text+"_"+treeA.find('size/height').text+"/"
        bexist = os.path.exists(outDir_JPEGImages+mkdir_tmp)
        if False==bexist:
            os.makedirs(outDir_JPEGImages+mkdir_tmp)

        bfind_one_space = False;
        icrop=0
        for obj in treeA.findall('object'):
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            icrop = icrop+1
            cropImg = im[ymin:ymax, xmin:xmax]
            name = obj.find('name').text+"_"+str(icrop)+"_"+str(xmax-xmin)+"_"+str(ymax-ymin)+".jpg"
            cv2.imwrite(outDir_JPEGImages+mkdir_tmp+"/"+name, cropImg)
            #cv2.rectangle(im, (xmin,ymin), (xmax,ymax), (255,0,0),3)

        i+=1 
        # if i==100:
        #     break
        print file_comp4
        #cv2.imwrite(outDir_JPEGImages+"/"+file_comp4, im)
        #treeA.write(outDir_Annotations+"/"+basename+".xml", encoding="utf-8",xml_declaration=False)
    print "num of image:",str(i)

def make_new_image_from_roi(JPEGImagesDir,AnnotationsDir,roi_dir,roi_width,roi_height,outDir_JPEGImages,outDir_Annotations):
    f_list_roi = os.listdir(roi_dir)
    ilength_roi = len(f_list_roi)
    roi_width=960
    roi_height=1280
    f_list = os.listdir(AnnotationsDir)
    for file_comp4 in f_list:
        print file_comp4
        if os.path.splitext(file_comp4)[1]!=".xml":
            continue
        basename = os.path.splitext(file_comp4)[0]
        im = cv2.imread(JPEGImagesDir+"/"+basename+".jpg")
        img_height = im.shape[0]
        img_width = im.shape[1]
        img_depth = im.shape[2]

        n1 = random.randint(0 , ilength_roi/2-1)
        n2 = random.randint(ilength_roi/2 , ilength_roi-1)
        im1 = cv2.imread(roi_dir+"/"+f_list_roi[n1])
        im2 = cv2.imread(roi_dir+"/"+f_list_roi[n2])
        scale1_ = float(img_width)/float(roi_width)
        print scale1_
        #print im1.shape
        scaled_width1 = int(float(scale1_)*float(im1.shape[1]))
        scaled_height1 = int(float(scale1_)*float(im1.shape[0]))
        scaled_width2 = int(float(scale1_)*float(im2.shape[1]))
        scaled_height2 = int(float(scale1_)*float(im2.shape[0]))
        scaled_img1=cv2.resize(im1,(scaled_width1,scaled_height1),interpolation=cv2.INTER_CUBIC)
        scaled_img2=cv2.resize(im2,(scaled_width2,scaled_height2),interpolation=cv2.INTER_CUBIC)

        if (img_width-scaled_width1-1)<2:
            print "("+str(img_width)+","+str(img_height)+")"+"    ("+str(scaled_width1)+","+str(scaled_height1)+")"
            continue
        if (img_height/2-scaled_height1-1)<2:
            print "("+str(img_width)+","+str(img_height)+")"+"    ("+str(scaled_width1)+","+str(scaled_height1)+")"
            continue
        if (img_width-scaled_width2-1)<2:
            print "("+str(img_width)+","+str(img_height)+")"+"    ("+str(scaled_width2)+","+str(scaled_height2)+")"
            continue
        if (img_height/2-scaled_height2-1)<2:
            print "("+str(img_width)+","+str(img_height)+")"+"    ("+str(scaled_width2)+","+str(scaled_height2)+")"
            continue
        start1_x = random.randint(1,(img_width-scaled_width1-1))
        start1_y = random.randint(1,(img_height/2-scaled_height1-1))
        print start1_x," ",start1_y," ",scaled_width1," ",scaled_height1
        im[start1_y:start1_y+scaled_height1,start1_x:start1_x+scaled_width1,:] = scaled_img1

        start2_x = random.randint(1,(img_width-scaled_width2-1))
        start2_y = random.randint(img_height/2,(img_height-scaled_height2-1))
        print start2_x," ",start2_y," ",scaled_width2," ",scaled_height2
        im[start2_y:start2_y+scaled_height2,start2_x:start2_x+scaled_width2,:] = scaled_img2
        basename = basename + "_1"
        cv2.imwrite(outDir_JPEGImages+"/"+basename+".jpg",im)

        file_xml = AnnotationsDir+"/"+file_comp4
        treeA=ElementTree()
        treeA.parse(file_xml)
        treeA.find('size/width').text=str(img_width)
        treeA.find('size/height').text=str(img_height)
        treeA.find('size/depth').text=str(img_depth)
        # width = int(treeA.find('size/width').text)
        # height = int(treeA.find('size/height').text)
        # depth = int(treeA.find('size/depth').text)
        #A1=treeA.find('annotation')
        A1=treeA.getroot()
        BBobj = create_node('object',{},"")
        SubElement(BBobj,"name").text=os.path.splitext(f_list_roi[n1])[0].split("_")[0]#f_list_roi[n1].split("_")[3]
        SubElement(BBobj,"pose").text="Unspecified"
        SubElement(BBobj,"truncated").text="0"
        SubElement(BBobj,"difficult").text="0"
        child5 = SubElement(BBobj,"bndbox")
        SubElement(child5,"xmin").text=str(start1_x)
        SubElement(child5,"ymin").text=str(start1_y)
        SubElement(child5,"xmax").text=str(start1_x+scaled_width1)
        SubElement(child5,"ymax").text=str(start1_y+scaled_height1)

        BBobj1 = create_node('object',{},"")
        SubElement(BBobj1,"name").text=os.path.splitext(f_list_roi[n2])[0].split("_")[0]
        SubElement(BBobj1,"pose").text="Unspecified"
        SubElement(BBobj1,"truncated").text="0"
        SubElement(BBobj1,"difficult").text="0"
        child51 = SubElement(BBobj1,"bndbox")
        SubElement(child51,"xmin").text=str(start2_x)
        SubElement(child51,"ymin").text=str(start2_y)
        SubElement(child51,"xmax").text=str(start2_x+scaled_width2)
        SubElement(child51,"ymax").text=str(start2_y+scaled_height2)

        A1.append(BBobj)
        A1.append(BBobj1)
        treeA.write(outDir_Annotations+basename+".xml", encoding="utf-8",xml_declaration=False)

def cut_roi_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations):
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



def test_make_xml(SKUfile,resultfile,testDir,outDir_Annotations):
    with open(SKUfile,"r") as SKUfile_f:
        SKUfile_lines = SKUfile_f.readlines()
    labelmap={}
    labellist=[] 
    for ilable,labelname in enumerate(SKUfile_lines):
        labelname = labelname.strip().strip('\n').strip('\r')
        #print labelname
        labelmap[labelname] = ilable+1
        labellist.append(labelname)

    with open(resultfile,"r") as f:
        lines = f.readlines()
        #lines = lines.strip("\n")
        #lines = lines.strip("\r")
    splitlines = [x.strip().split(' ') for x in lines]
    #Nametxt labelIDtxt width height x1txt y1txt x2txt y2txt
    Nametxt =  [x[0] for x in splitlines]
    labelIDtxt =    [int(x[1]) for x in splitlines]
    width =   [int(float(x[2])) for x in splitlines]
    height =    [int(float(x[3])) for x in splitlines]
    x1txt =    [int(float(x[4])) for x in splitlines]
    y1txt =    [int(float(x[5])) for x in splitlines]
    x2txt =    [int(float(x[6])) for x in splitlines]
    y2txt =    [int(float(x[7])) for x in splitlines]

    ilen = len(Nametxt)
    f_list = os.listdir(testDir)
    # print f_list
    for file_comp4 in f_list:
        print file_comp4
        if os.path.splitext(file_comp4)[1] == '.jpg':
            basename = os.path.splitext(file_comp4)[0]

            bfind = False
            bmakehead = False
            for i, pindexname in enumerate(Nametxt):
                if bfind== True:
                    if basename!=pindexname:
                        four_root.write(outDir_Annotations+"/"+basename+".xml", encoding="utf-8",xml_declaration=False)
                        break;
                if basename ==pindexname:
                    bfind = True
                #or i,x1 in enumerate(x1txt):
                    #xml====================================
                    if bmakehead==False:
                        bmakehead = True
                        four_root = ElementTree()
                        A1 = create_node('annotation',{},"")
                        four_root._setroot(A1)
                        B1 = create_node('folder',{},"2")
                        B2 = create_node('filename',{},Nametxt[i])
                        B3 = create_node('path',{},"2")
                        A1.append(B1)
                        A1.append(B2)
                        A1.append(B3)
                        # SubElement(A1,"folder").text=str(width[i])
                        # SubElement(A1,"filename").text=str(height[i])
                        # SubElement(A1,"path").text="3"
                        B4 = create_node('source',{},"")
                        A1.append(B4)
                        C1 = create_node('database',{},"Unknown")
                        B4.append(C1)
                        B5 = create_node('size',{},"")
                        SubElement(B5,"width").text=str(width[i])
                        SubElement(B5,"height").text=str(height[i])
                        SubElement(B5,"depth").text="3"
                        A1.append(B5)
                        # D1 = create_node('width',{},str(width[i]))
                        # B5.append(D1)
                        # D2 = create_node('height',{},str(height[i]))
                        # B5.append(D2)
                        # D3 = create_node('depth',{},str(3))
                        # B5.append(D3)
                        B6 = create_node('segmented',{},"0")
                        A1.append(B6)
                    #xml====================================

                    #or obj in objects:
                        #box = obj['bbox']
                    ixmin = x1txt[i] #p.maximum(bbox[ 0], x1txt[i])
                    iymin = y1txt[i]#p.maximum(bbox[ 1], y1txt[i])
                    ixmax = x2txt[i]#p.minimum(bbox[ 2], (x1txt[i]+width[i]))
                    iymax = y2txt[i]#np.minimum(bbox[ 3], (y1txt[i]+height[i]))

                    if ixmin<=0 :
                        ixmin = 1
                    if iymin<=0 :
                        iymin = 1
                    if   ixmax>= width[i] :
                        ixmax = width[i]-1
                    if   iymax>= height[i] :
                        iymax = height[i]-1
                    
                    BBobj = create_node('object',{},"")
                    SubElement(BBobj,"name").text=labellist[labelIDtxt[i]-1]
                    SubElement(BBobj,"pose").text='Unspecified'
                    SubElement(BBobj,"truncated").text='0'
                    SubElement(BBobj,"difficult").text='0'
                    child5 = SubElement(BBobj,"bndbox")
                        # child1= create_node('name',{},obj['name'])
                    SubElement(child5,"xmin").text=str(ixmin)
                    SubElement(child5,"ymin").text=str(iymin)
                    SubElement(child5,"xmax").text=str(ixmax)
                    SubElement(child5,"ymax").text=str(iymax)
                    A1.append(BBobj)
                    if i==ilen-1:
                        four_root.write(outDir_Annotations+"/"+basename+".xml", encoding="utf-8",xml_declaration=False)


def make_101_prototxt(proto_src_dir,proto_des_dir,name,aa):
    ResNet_type = "ResNet-101"
    train_prototxt = proto_src_dir+"/"+ResNet_type+"/b14_train_agnostic_16_s_4_8_16_32_ohem.prototxt"
    test_prototxt = proto_src_dir+"/"+ResNet_type+"/b14_test_16_s_4_8_16_32_agnostic.prototxt"
    solver_prototxt = proto_src_dir+"/"+ResNet_type+"/b14_solver_ohem_16_s_4_8_16_32.prototxt"
    out_train_prototxt = proto_des_dir+"/rfcn_end2end/s16_14/"+"b14_train_agnostic_16_s_4_8_16_32_ohem.prototxt"
    out_test_prototxt = proto_des_dir+"/rfcn_end2end/s16_14/"+"b14_test_16_s_4_8_16_32_agnostic.prototxt"
    out_solver_prototxt = proto_des_dir+"/rfcn_end2end/s16_14/"+"b14_solver_ohem_16_s_4_8_16_32.prototxt"
    
    with open(train_prototxt,"r") as file_f:
        lines_semi = file_f.readlines()
        lines_semi[10] = "    param_str: \"'num_classes': "+str(aa)+"\" #lius\n"
        lines_semi[7131] = "        num_output: "+str(aa*4*49)+" #21*(7^2) cls_num*(score_maps_size^2) #lius\n"
        lines_semi[7185] = "        output_dim: "+str(aa)+" #lius\n"
        with open(out_train_prototxt,"w") as file_f1:
            #lines_semi = [line+'\n' for line in lines_semi]
            file_f1.writelines(lines_semi)

    with open(test_prototxt,"r") as file_f:
        lines_semi = file_f.readlines()
        lines_semi[7053] = "        num_output: "+str(aa*4*49)+" #21*(7^2) cls_num*(score_maps_size^2) #lius\n"
        lines_semi[7107] = "        output_dim: "+str(aa)+" #lius\n"
        lines_semi[7167] = "            dim: "+str(aa)+" #lius\n"
        with open(out_test_prototxt,"w") as file_f1:
            #lines_semi = [line+'\n' for line in lines_semi]
            file_f1.writelines(lines_semi)
    with open(solver_prototxt,"r") as file_f:
        lines_semi = file_f.readlines()
        lines_semi[0] = "train_net: \"models/"+name+"/"+ResNet_type+"/rfcn_end2end/s16_14/b14_train_agnostic_16_s_4_8_16_32_ohem.prototxt\"\n"
        lines_semi[13] = "snapshot_prefix: \""+ResNet_type+"_b14_16_s_4_8_16_32_"+name+"_rfcn_ohem\""  # "101_b14_16_s_4_8_16_32_"+tmp_work[4]+"_rfcn_ohem"
        with open(out_solver_prototxt,"w") as file_f1:
            #lines_semi = [line+'\n' for line in lines_semi]
            file_f1.writelines(lines_semi)

    train_prototxt = proto_src_dir+"/"+ResNet_type+"/train_agnostic_ohem.prototxt"
    test_prototxt = proto_src_dir+"/"+ResNet_type+"/test_agnostic.prototxt"
    solver_prototxt = proto_src_dir+"/"+ResNet_type+"/solver_ohem.prototxt"
    out_train_prototxt = proto_des_dir+"/rfcn_end2end/"+"train_agnostic_ohem.prototxt"
    out_test_prototxt = proto_des_dir+"/rfcn_end2end/"+"test_agnostic.prototxt"
    out_solver_prototxt = proto_des_dir+"/rfcn_end2end/"+"solver_ohem.prototxt"
    
    with open(train_prototxt,"r") as file_f:
        lines_semi = file_f.readlines()
        lines_semi[10] = "    param_str: \"'num_classes': "+str(aa)+"\" #lius\n"
        lines_semi[7131] = "        num_output: "+str(aa*4*49)+" #21*(7^2) cls_num*(score_maps_size^2) #lius\n"
        lines_semi[7185] = "        output_dim: "+str(aa)+" #lius\n"
        with open(out_train_prototxt,"w") as file_f1:
            #lines_semi = [line+'\n' for line in lines_semi]
            file_f1.writelines(lines_semi)

    with open(test_prototxt,"r") as file_f:
        lines_semi = file_f.readlines()
        lines_semi[7053] = "        num_output: "+str(aa*4*49)+" #21*(7^2) cls_num*(score_maps_size^2) #lius\n"
        lines_semi[7107] = "        output_dim: "+str(aa)+" #lius\n"
        lines_semi[7167] = "            dim: "+str(aa)+" #lius\n"
        with open(out_test_prototxt,"w") as file_f1:
            #lines_semi = [line+'\n' for line in lines_semi]
            file_f1.writelines(lines_semi)
    with open(solver_prototxt,"r") as file_f:
        lines_semi = file_f.readlines()
        lines_semi[0] = "train_net: \"models/"+name+"/"+ResNet_type+"/rfcn_end2end/solver_ohem.prototxt\"\n"
        lines_semi[13] = "snapshot_prefix: \""+ResNet_type+"_"+name+"_rfcn_ohem\""  # "101_"+tmp_work[4]+"_rfcn_ohem"
        with open(out_solver_prototxt,"w") as file_f1:
            #lines_semi = [line+'\n' for line in lines_semi]
            file_f1.writelines(lines_semi)

def stastic_all_SKU_NUM(SKUfile,xmldir,outfile):
    ft = open(SKUfile,"r")
    line = ft.readline()
    classnum = {}
    xmllist_top = []
    xmllist = []
    xmlmap=[]
    classlen = 0
    index = 0 
    while line:
        line = line.strip('\r\n')
        classnum[line] = 0 
        xmllist_top.append(line)
        xmllist.append(classlen)
        xmlmap.append(classlen)
        xmlmap[classlen] = {}
        xmllist[classlen] = []
        xmllist[classlen].append(line)
        line = ft.readline()
        classlen = classlen + 1 
    ft.close()

    #xmldir = "/nas/public/liushuai/beer/baiwei_2143/xml_reset/"
    files = os.listdir(xmldir)
    for filename in files:
        if os.path.splitext(filename)[1] == '.xml':
            print filename + "\n"
            xmldata = ElementTree()
            xmldata.parse(xmldir+filename)
            nodelist = xmldata.findall("object")
            for node in nodelist: 
                num = classnum.get(node.find("name").text)
                if num is None:
                    classnum.setdefault(node.find("name").text,1)
                    xmllist_top.append(node.find("name").text)
                    xmllist.append(0)
                    xmllist[len(xmllist)-1] = []
                    xmllist[len(xmllist)-1].append(node.find("name").text)
                    xmllist[len(xmllist)-1].append(filename)
                else:
                    classnum[node.find("name").text] = num + 1
                    index = xmllist_top.index(node.find("name").text)
                    xmllist[index].append(filename)                    
                    xmlmap[index][filename]=0


                
    fl = open(outfile,"w")
    index = 0
    for i in range(len(xmllist_top)):
        if  xmllist_top[i]=="binggan":
            fl.write("\n\n\nWARN  WARN  WARN\n"+xmllist_top[i]+"\t"+bytes(classnum[xmllist_top[i]])+"\nWarning!!!this SKU is not exist!! appear files :\n")
            for key in xmlmap[i]:#enumrate(xmlmap[i])
                fl.write(key+"\n")
            fl.write("\n\n\n")

        if index < classlen  :
            fl.write(xmllist_top[i]+"\t"+bytes(classnum[xmllist_top[i]])+"\n")
        else:
            fl.write("\n\n\nWARN  WARN  WARN\n"+xmllist_top[i]+"\t"+bytes(classnum[xmllist_top[i]])+"\nWarning!!!this SKU is not exist!! appear files :\n")
            for j in range(len(xmllist[index])):
                fl.write(xmllist[index][j]+"\n")
            fl.write("\n\n\n")
        index =index + 1
    fl.write("\n\n\n\n\n\n")
    index = 0
    for i in range(len(xmllist_top)):
        if index < classlen:
            fl.write(xmllist_top[i]+"\t"+bytes(classnum[xmllist_top[i]])+"\n")
        else:
            fl.write(xmllist_top[i]+"\n")
            
        index =index + 1
    fl.close()

def confusionMatrix(skufile,testlistfile,Annotation_dir,resultfile,out_dir,cachedir,cachefilename):
    with open(skufile,"r") as SKUfile_f:
        SKUfile_lines = SKUfile_f.readlines()
    labellist=[]
    labelmap={}
    labelnamemap={}
    label_sum={}
    for ilable,labelname in enumerate(SKUfile_lines):
        labelname = labelname.strip().strip('\n').strip('\r')
        labellist.append(labelname)
        labelmap[ilable]=labelname
        labelnamemap[labelname]=ilable
        label_sum[ilable]=0
    label_dic=[]
    for ilable,labelname in enumerate(labellist):
        labelmap_tmp={}
        for i,z in  enumerate(labellist):
            labelmap_tmp[i]=0
        label_dic.append(labelmap_tmp)

    with open(testlistfile,"r") as test_filelist_f:
        test_filelist_lines = test_filelist_f.readlines()

    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, cachefilename)
    if not os.path.isfile(cachefile):
        # load annots
        gt_obj = {}
        for i_file,test_file in enumerate(test_filelist_lines):
            print test_file
            temp_file = test_file.strip().strip('\n').strip('\r')
            if temp_file=="" or temp_file=='':
                print temp_file
                continue
            gt_obj[temp_file]=parse_rec(Annotation_dir+temp_file+".xml")
            if i_file % 50 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i_file + 1, len(test_filelist_lines))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(gt_obj, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            gt_obj = cPickle.load(f)
    with open(resultfile,"r") as f:
        lines = f.readlines()
        #lines = lines.strip("\n")
        #lines = lines.strip("\r")
    splitlines = [x.strip().split(' ') for x in lines]
    #Nametxt labelIDtxt width height x1txt y1txt x2txt y2txt
    Nametxt =  [x[0] for x in splitlines]
    labelIDtxt =    [int(x[1]) for x in splitlines]
    width =   [int(float(x[2])) for x in splitlines]
    height =    [int(float(x[3])) for x in splitlines]
    x1txt =    [int(float(x[4])) for x in splitlines]
    y1txt =    [int(float(x[5])) for x in splitlines]
    x2txt =    [int(float(x[6])) for x in splitlines]
    y2txt =    [int(float(x[7])) for x in splitlines]
    for i_file,test_file in enumerate(test_filelist_lines):
        temp_file = test_file.strip().strip('\n').strip('\r')
        #tianluo_dic[temp_file] = 0
        #tianluo_dic[temp_file+'_sum_'] = 0
        if temp_file=="" or temp_file=='':
            continue
        obj_struc=gt_obj[temp_file]
        #for i,obj_struc in enumerate(gt_obj):
        for j,objcts in enumerate(obj_struc):
            tmp_name = objcts['name']
            label_sum[labelnamemap[tmp_name]]+=1
            bfind=False
            for i_Nametxt, indexname in enumerate(Nametxt):
                if temp_file == indexname:# and labelIDtxt[i_Nametxt] == labelmap[objcts['name']]:
                    bfind=True
                    #print temp_file,indexname,str(labelIDtxt[i_Nametxt] ),str(labelmap[objcts['name']])
                    areaRate = IOU(x1txt[i_Nametxt],y1txt[i_Nametxt],x2txt[i_Nametxt],y2txt[i_Nametxt],
                                   objcts['bbox'][0],objcts['bbox'][1],objcts['bbox'][2],objcts['bbox'][3])
                    #print "areaRate= ",areaRate
                    if areaRate >= 0.5:
                        label_dic[labelnamemap[tmp_name]][labelIDtxt[i_Nametxt]-1]+=1
                else:
                    if bfind==True:
                        break
    conf_file=out_dir+'conf_file.txt'
    #horizon_file=out_dir+'horizon_file.txt'
    csv_file = out_dir+"csvData.csv"
    csv_f = open(csv_file, "w")
    writer = csv.writer(csv_f)
    cc=[]
    cc.append(" ")
    cc=cc+labellist
    writer.writerow(cc)
    #with open("csvData.csv", "r") as csv_f:
    #conf_file=out_dir+labelmap[i]+".txt"
    txt_f = open(conf_file, "w")
    for i,iname in enumerate(label_dic):
        sum=[]
        sum.append(labelmap[i])
        txt_f.write(labelmap[i]+"-"+str(label_sum[i])+" \t ")
        for sz,szname in enumerate(labellist):
            sum.append(str(label_dic[i][sz]) if label_dic[i][sz]!=0 else "")
            if label_dic[i][sz]!=0:
                txt_f.write(szname+"-"+str(label_dic[i][sz])+"  ")
        txt_f.write("\n")
        writer.writerow(sum)
    csv_f.close()
    txt_f.close()
          

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #         description = "Plot the detection results output by ssd_detect.")
    # parser.add_argument("resultfile",
    #         help = "A dir which contains all the detection results.")
    # parser.add_argument("imgdir",
    #         help = "A dir which contains all the detection results.")
    # args = parser.parse_args()

    # result_file_dir = args.resultfile
    # if not os.path.exists(result_file_dir):
    #     print "{} does not exist".format(result_file_dir)
    #     sys.exit()

    # img_dir = args.imgdir
    # if not os.path.exists(img_dir):
    #     print "{} does not exist".format(img_dir)
    #     sys.exit()

#     annopath = "/mnt/storage/liushuai/data/cookie/cookieproj1/Annotations/{}.xml"
# jpegpath = "/mnt/storage/liushuai/data/cookie/cookieproj1/JPEGImages/{}.jpg"
# imagesetfile = "/mnt/storage/liushuai/data/cookie/cookieproj1/ImageSets/Main/trainval.txt"
# cachedir = "/home/liushuai/storage/data/beer/check_iamgeHW_cache"
# image_check(annopath,
#              imagesetfile,
#              cachedir,
#              jpegpath)

    #check_anotation_name("/mnt/storage/liushuai/data/cookiecut/cookiecutproj1//Annotations/","/mnt/storage/liushuai/data/cookiecut/cookiecutproj1//Annotations/")
    thedatadir = "/mnt/storage/liushuai/data/cookie/cookieproj1/"
    #scale_JPEGImages(thedatadir+"JPEGImages",thedatadir+"Annotations",thedatadir+"scale_JPEGImages",thedatadir+"scale_Annotations")
    cookiecutdir = "/mnt/storage/liushuai/data/cookiecut/cookiecutproj1/"
    #aa=parse_labelmap("/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/predefined_classes0809.txt","/mnt/storage/liushuai/data/colgatecut/colgatecutproj1//105a111111111111.txt","/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/105b111111111.prototxt")

    basedir = "/mnt/storage/liushuai/RFCN/R_fcn_bin/"
    resultfile= basedir+"detect_new/build/colgate1000_17_gpu_0.45conf_0.2gpunms_others0.75conf_0.7RPN_nms.txt" #"/nas/public/pengshengfeng/annotation.txt"#
    Anotations_Dir="/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/Annotations/"#basedir+"m10/"#
    SKUfile=basedir+"colgate_newSKU.txt"
    test_filelist="/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/ImageSets/Main/test1000.txt" #"m10/m10.txt"
    nas_dir = "/nas/public/liushuai/"

    # basedir = "/mnt/storage/liushuai/RFCN/R_fcn_bin/"
    # resultfile= basedir+"detect_new/build/colgate_gpu_0.45conf_0.2gpunms_others0.75conf_0.7RPN_nms.txt" #"/nas/public/pengshengfeng/annotation.txt"#
    # Anotations_Dir="/mnt/storage/liushuai/data/colgate/colgateproj1/Annotations/"#basedir+"m10/"#
    # SKUfile=basedir+"colgate_classes.txt"
    # test_filelist="/mnt/storage/liushuai/data/colgate/colgateproj1/ImageSets/Main/test.txt" #"m10/m10.txt"
    # nas_dir = "/nas/public/liushuai/"

    #test_filelist=maindir+"ImageSets/Main/test.txt"

    # resultfile= basedir+"detect_new/build/cut_60_gpu_0.45conf_0.2cpunms_others0.6conf_0.7RPN_nms.txt" 
    # Anotations_Dir= basedir+"m60_cut/"#basedir+"m10/"#
    # SKUfile=basedir+"cookieSKU.txt"
    # test_filelist=basedir+"m60_cut/m60_cut.txt" #"m10/m10.txt"
    # nas_dir = "/nas/public/liushuai/"
    cachedir=basedir
    #cut_4_xml(basedir+"TXT/",basedir+"m10/",0.4,basedir+"m60_cut/")
    beerdir="/mnt/storage/liushuai/data/beer/beerproj1/"
    colgatecut = "/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/"
    baiwei_basedir="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj1/"
    base="/storage2/tiannuodata/work/projdata/nestle4goods/nestle4goodsproj2/"
    base="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj1/"
    SKUfile=base+"/skufile_3_14.txt"
    xmldir=base+"/Annotations/"
    outfile=base+"baiwei_sum_statistic.txt"
    #stastic_all_SKU_NUM(SKUfile,xmldir,outfile)
    #check_Anotation_name_error(baiwei_basedir+"/Annotations_3_14/",baiwei_basedir+"/JPEGImages/",baiwei_basedir+"/Annotations/")
    coffeedir ="/mnt/storage/liushuai/data/nestlecoffee/nestlecoffeeproj1/"
    nestle4goods_dir="/storage2/tiannuodata/work/projdata/nestle4goods/nestle4goodsproj2/"
    #remove_Anotations(nestle4goods_dir+"/Annotations/",nestle4goods_dir+"/Annotations_3/")

    basedir = "/mnt/storage/liushuai/RFCN/R_fcn_bin/"
    resultfile= basedir+"detect_new/build/colgate0810_148_106_gpu_0.45conf_0.2gpunms_others0.75conf_0.7RPN_nms.txt" #"/nas/public/pengshengfeng/annotation.txt"#
    Anotations_Dir=colgatecut+"/test_colgate/xml/"#basedir+"m10/"#
    SKUfile=colgatecut+"predefined_classes0809.txt"
    test_filelist=colgatecut+"/test_colgate/setfile.txt" #"m10/m10.txt"
    nas_dir = "/nas/public/liushuai/"
    
    resultfile= basedir+"detect_new/build/nocutbeer_134_gpu_0.45conf_0.2gpunms_others0.75conf_0.7RPN_nms.txt"
    test_beer_dir = "//storage/dataset/tiannuo_data/seconde_data/baiwei082224_2152/"
    Anotations_Dir=test_beer_dir+"/test_beer/"#basedir+"m10/"#
    test_filelist=test_beer_dir+"test.txt" #"m10/m10.txt"
    SKUfile="/storage/dataset/tiannuo_data/seconde_data/134_baiweiSKU.txt"

    # basedir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj1/"
    # resultfile= basedir+"_result.txt"
    # #test_beer_dir = "//storage/dataset/tiannuo_data/seconde_data/baiwei082224_2152/"
    # Anotations_Dir=basedir+"/Annotations/"#basedir+"m10/"#
    # test_filelist=basedir+"ImageSets/Main/test.txt" #"m10/m10.txt"
    # SKUfile=basedir+"skufile.txt"
    # cachedir=basedir
    #tianruo_GoodEval(resultfile,Anotations_Dir,SKUfile,test_filelist,cachedir,'nestle4goodsproj2.pkl',threhold=231)#293
    base_dir='/storage2/tiannuodata/work/projdata/aofei/aofeiproj1/'
    basedir = "/storage2/tiannuodata/work/projdata/aofei/aofeiproj1//"
    skufile=basedir+"skufile.txt"
    testlistfile = basedir+'/ImageSets/Main/test.txt'
    Annotation_dir = basedir+'/Annotations/'
    resultfile = basedir+'/_result.txt'
    out_dir = basedir+'/analysis/'
    cachedir = out_dir
    cachefilename = "aofei_analysis.pkl"
    #confusionMatrix(skufile,testlistfile,Annotation_dir,resultfile,out_dir,cachedir,cachefilename)
    resultfile= basedir+"/_result.txt"
    #test_beer_dir = "//storage/dataset/tiannuo_data/seconde_data/baiwei082224_2152/"
    Anotations_Dir=basedir+"//Annotations/"#basedir+"m10/"#
    test_filelist=basedir+"/ImageSets/Main/test.txt" #"m10/m10.txt"
    SKUfile=basedir+"skufile.txt"
    cachedir=basedir+"/"
    #remove_Anotations(basedir+"/Annotations_package/",basedir+"/Annotations/")
    #tianruo_GoodEval(resultfile,Anotations_Dir,SKUfile,test_filelist,cachedir,'aofei.pkl',threhold=1352)#293

    proto_src_dir='/home/liushuai/tiannuocaffe/prototxtdir/'
    rfcn_dir_model='/home/liushuai/tiannuocaffe/py-rfcn-gpu/models/'
    tmp_work='nestle4goods'
    # #make model 101
    # if False == os.path.exists(rfcn_dir_model+"/"+tmp_work+"/"):
    #     os.mkdir(rfcn_dir_model+"/"+tmp_work+"/")
    # model_path101 = rfcn_dir_model+"/"+tmp_work+"/ResNet-101/"
    # if False == os.path.exists(model_path101):
    #     os.mkdir(model_path101)
    # if False == os.path.exists(model_path101+"/rfcn_end2end/"):
    #     os.mkdir(model_path101+"/rfcn_end2end/")
    # if False == os.path.exists(model_path101+"/rfcn_end2end/s16_14/"):
    #     os.mkdir(model_path101+"/rfcn_end2end/s16_14/")
    # proto_des_dir= model_path101
    # bg_plus_fg=parse_labelmap(basedir+'skufile.txt',basedir,"aa.txt","bb.prototxt")
    # make_101_prototxt(proto_src_dir,proto_des_dir,tmp_work,bg_plus_fg)
    # resultfile= basedir+"detect_new/build/colgate0803_262_21_gpu_0.45conf_0.2gpunms_others0.75conf_0.7RPN_nms.txt" #"/nas/public/pengshengfeng/annotation.txt"#
    # Anotations_Dir="/nas/public/liushuai/colgate/colgate0803_262//xml/"#basedir+"m10/"#
    # SKUfile=basedir+"colgate_newSKU.txt"
    # test_filelist="/nas/public/liushuai/colgate/colgate0803_262/setfile.txt" #"m10/m10.txt"
    # nas_dir = "/nas/public/liushuai/"

    # extra2basedir = "/mnt/storage/liushuai/data/extra2/extra2proj1/"
    # resultfile= basedir+"detect_new/build/cut_4extra_gpu_0.45conf_0.2cpunms_others0.6conf_0.7RPN_nms.txt" 
    # Anotations_Dir= extra2basedir+"TESTJPG_127/"#basedir+"m10/"#
    # SKUfile=extra2basedir+"ExtraSKU_new.txt"
    # test_filelist=extra2basedir+"TESTJPG_127/test.txt" #"m10/m10.txt"
    # nas_dir = "/nas/public/liushuai/"
    # cachedir=extra2basedir

    # # #cut_4_xml(chuttybasedir+"TXT/",chuttybasedir+"yida0607-127",0.4,chuttybasedir+"JPEGImages_test_cut/")
    # # tianruo_GoodEval(resultfile,Anotations_Dir,SKUfile,test_filelist,cachedir,'extra2_GoodEval_annots.pkl')
    testDir = basedir+"/JPEGImages/"#"/mnt/storage/liushuai/tiannuo_data/xin/JPEG/"
    outDir_Annotations = basedir+"/outDir_Annotations/"#"/mnt/storage/liushuai/tiannuo_data/xin/xml/"
    #test_make_xml(SKUfile,resultfile,testDir,outDir_Annotations)
    #cut_roi_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations)
    cachefile = "/home/zhuxiaoli/py-R-FCN/data/cache/cookie_trainval_gt_roidb.pkl"
    #SKUfile = "/nas/public/zhuxiaoli/cookie2/cookie324.txt"
    #static_class_obj_num(SKUfile,cachefile)
    colgatecutproj1_dir = "/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/"
    JPEGImagesDir = colgatecutproj1_dir + "JPEGImages/"
    AnnotationsDir = colgatecutproj1_dir + "Annotations/"
    outDir_JPEGImages = colgatecutproj1_dir + "out_roi_image/"
    colgateproj1_dir = "/mnt/storage/liushuai/data/colgate/colgateproj1/"
    #cut_Annotation_roi_from_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages)
    JPEGImagesDir = colgateproj1_dir + "/cookie_extra/"
    AnnotationsDir = colgateproj1_dir + "/testAnno/"
    roi_dir = colgatecutproj1_dir + "out_roi_image/"
    outDir_JPEGImages = colgatecutproj1_dir + "new_make_image/"
    outDir_Annotations =colgatecutproj1_dir +  "new_make_anno/"
    #make_new_image_from_roi(JPEGImagesDir,AnnotationsDir,roi_dir+"960_1280/",960,1280,outDir_JPEGImages,outDir_Annotations)
    #celect_Dir_2_filelist(basedir+"m10/",basedir+"m10/m10.txt")
    #copy_cut_4_traivaltxt(thedatadir+"ImageSets/Main/",cookiecutdir+"ImageSets/Main/")
    # JPEGImagesDir= "/mnt/storage/dataset/baipai/beer/JPEG/"
    # AnnotationsDir= "/mnt/storage/dataset/baipai/beer/xml/"
    # outDir_JPEGImages= "/mnt/storage/dataset/baipai/check/"
    # made_rect_JPEGImages(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages)
    base="/storage2/liushuai/gs6_env/market1501_extract_freature/test/"
    SKUfile = base+"skufile.txt"
    JPEGImagesDir= base+"/NG/"
    AnnotationsDir= base+"/out/"
    outDir_JPEGImages= base+"/out_rect/"
    made_rect_JPEGImages(SKUfile,JPEGImagesDir,AnnotationsDir,outDir_JPEGImages)
    base="/storage2/liushuai/gs6_env/market1501_extract_freature/test/"
    SKUfile = base+"skufile.txt"
    JPEGImagesDir= base+"/OK/"
    AnnotationsDir= base+"/OK/"
    outDir_JPEGImages= base+"/OK_rect/"
    made_rect_JPEGImages(SKUfile,JPEGImagesDir,AnnotationsDir,outDir_JPEGImages)
    JPEGImagesDir= base+"/NG/"
    AnnotationsDir= base+"/NG/"
    outDir_JPEGImages= base+"/NG_rect/"
    made_rect_JPEGImages(SKUfile,JPEGImagesDir,AnnotationsDir,outDir_JPEGImages)

    #@made_rect_JPEGImages_from_SKUfile(SKUfile,nas_dir+"60_pic_testset/m10.txt",nas_dir+"60_pic_testset",nas_dir+"60_pic_testset",nas_dir+"/60_gt_box")
    colgatecutdir = "/mnt/storage/liushuai/data/colgatecut/colgatecutproj1/"

    beer134dir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj66/"
    # randomAlloc(beer134dir+"Annotations/",beer134dir+"ImageSets/Main/")
    # beer134dir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/"
    # randomAlloc(beer134dir+"Annotations/",beer134dir+"ImageSets/Main/")
