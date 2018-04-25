
import argparse
from collections import OrderedDict
#from google.protobuf import text_format
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import skimage.io as io
import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
import cPickle
import random
import math
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

def cut_4_xml(basename,crop_width,crop_height, objects ,x1txt, y1txt,ovthresh=0.4):
    for i,x1 in enumerate(x1txt):
        #xml====================================
        four_root = ElementTree()
        A1 = create_node('annotation',{},"")
        four_root._setroot(A1)
        B1 = create_node('folder',{},"2")
        B2 = create_node('filename',{},basename)
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
        SubElement(B5,"width").text=str(crop_width)
        SubElement(B5,"height").text=str(crop_height)
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
            ixmax = np.minimum(bbox[ 2], (x1txt[i]+crop_width))
            iymax = np.minimum(bbox[ 3], (y1txt[i]+crop_height))

            if ixmin<=x1txt[i] :
                ixmin = x1txt[i]+1
            if iymin<=y1txt[i] :
                iymin = y1txt[i]+1
            if   ixmax>= (x1txt[i]+crop_width) :
                ixmax = (x1txt[i]+crop_width)-1
            if   iymax>= (y1txt[i]+crop_height) :
                iymax = (y1txt[i]+crop_height)-1
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
    return four_root

def cut_4_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations):
    f_list = os.listdir(JPEGImagesDir)
    i=0
    for file_comp4 in f_list:
        im = cv2.imread(os.path.join(JPEGImagesDir,file_comp4))
        # if os.path.splitext(file_comp4)[1] == '.png':
        #     file_comp4 = os.path.splitext(file_comp4)[0] + ".jpg"
        basename= os.path.splitext(file_comp4)[0]
        if  not im is None:
            img_height=im.shape[0]
            img_width = im.shape[1]

            baseInfo,objects = parse_xml(os.path.join(AnnotationsDir,basename+".xml"))
            #for z in list(['1','2','3','4']):


            filename_cut=basename+"_1.jpg"
            filename_cut_xml=basename+"_1.xml"
            cropImg = im[0 : int(img_height*0.6) , 0 : int(img_width*0.6)]
            cv2.imwrite(os.path.join(outDir_JPEGImages,filename_cut), cropImg)
            crop_width=int(img_width*0.6)
            crop_height=int(img_height*0.6)
            x1txt=[0] 
            y1txt=[0]
            four_root=cut_4_xml(basename,crop_width,crop_height, objects ,x1txt, y1txt)
            # basename crop_width crop_height objects x1txt y1txt
            four_root.write(os.path.join(outDir_Annotations,filename_cut_xml), encoding="utf-8",xml_declaration=False)

            filename_cut=basename+"_2.jpg"
            filename_cut_xml=basename+"_2.xml"
            cropImg = im[0 : int(img_height*0.6) , int(img_width*0.4) : img_width]
            cv2.imwrite(os.path.join(outDir_JPEGImages,filename_cut), cropImg)
            y1txt=[0] 
            x1txt=[int(img_width*0.4)]
            four_root=cut_4_xml(basename,crop_width,crop_height, objects ,x1txt, y1txt)
            # basename crop_width crop_height objects x1txt y1txt
            four_root.write(os.path.join(outDir_Annotations,filename_cut_xml), encoding="utf-8",xml_declaration=False)


            filename_cut=basename+"_3.jpg"
            filename_cut_xml=basename+"_3.xml"
            cropImg = im[int(img_height*0.4) : img_height , 0 : int(img_width*0.6)]
            cv2.imwrite(os.path.join(outDir_JPEGImages,filename_cut), cropImg)
            y1txt=[int(img_height*0.4)] 
            x1txt=[0]
            four_root=cut_4_xml(basename,crop_width,crop_height, objects ,x1txt, y1txt)
            # basename crop_width crop_height objects x1txt y1txt
            four_root.write(os.path.join(outDir_Annotations,filename_cut_xml), encoding="utf-8",xml_declaration=False)


            filename_cut=basename+"_4.jpg"
            filename_cut_xml=basename+"_4.xml"
            cropImg = im[int(img_height*0.4) : img_height , int(img_width*0.4) : img_width]
            cv2.imwrite(os.path.join(outDir_JPEGImages,filename_cut), cropImg)
            y1txt=[int(img_height*0.4)] 
            x1txt=[int(img_width*0.4)]
            four_root=cut_4_xml(basename,crop_width,crop_height, objects ,x1txt, y1txt)
            # basename crop_width crop_height objects x1txt y1txt
            four_root.write(os.path.join(outDir_Annotations,filename_cut_xml), encoding="utf-8",xml_declaration=False)

    

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

if __name__ == "__main__":
    JPEGImagesDir="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\jpegnest\\"
    AnnotationsDir="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\xml\\"
    outDir_JPEGImages="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\out_jpg\\"
    outDir_Annotations="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\out_xml\\"
    cut_4_image(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages,outDir_Annotations)