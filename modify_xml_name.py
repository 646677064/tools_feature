
import argparse
# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# #matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
# import skimage.io as io
import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
# import cPickle
import random
# import math
# import sys,os,subprocess,commands
# from subprocess import Popen,PIPE

def modify_xml_name(src_xml,out_xmlDir,src_list,des_list,JPG_dir):
    print "===========start"
    if len(src_list)!=len(des_list):
        print "src_list and des_list is wrong!!!"
        return
    if False==os.path.exists(src_xml):
        print "wrong src_xml dir!!!"
        return
    if False==os.path.exists(out_xmlDir):
        os.mkdir(out_xmlDir)
    command = "cp "+src_xml+"*.xml "+out_xmlDir
    print command
    os.system(command)
    f_list = os.listdir(src_xml)
    i=0
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == '.xml':
            file_tmp = src_xml+"\\"+file_comp4
            treeA=ElementTree()
            treeA.parse(file_tmp)
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            bfind_one_space = False;

            JPEG_resetdir =JPG_dir#work_dir +name +"/JPEGImages_reset/"
            print os.path.join(JPEG_resetdir, os.path.splitext(file_comp4)[0]+".jpg")
            if os.path.exists(os.path.join(JPEG_resetdir, os.path.splitext(file_comp4)[0]+".jpg")):#(JPEG_resetdir+"/"+os.path.splitext(file_comp4)[0]+".jpg"):
                im = cv2.imread(os.path.join(JPEG_resetdir, os.path.splitext(file_comp4)[0]+".jpg"))
                #sp = im.shape
                imheight = im.shape[0]
                imwidth = im.shape[1]
                imdepth = im.shape[2]
                if width!=imwidth:
                    bfind_one_space = True
                    treeA.find('size/width').text=str(imwidth)
                    width=imwidth
                    print file_comp4,"error size/width"
                if height!=imheight:
                    bfind_one_space = True
                    treeA.find('size/height').text=str(imheight)
                    height=imheight
                    print file_comp4,"error size/height"
                if depth!=imdepth:
                    bfind_one_space = True
                    treeA.find('size/depth').text=str(imdepth)
                    depth=imdepth
                    print file_comp4,"error size/depth"
            else:
                print file_comp4,'not exist'
            if width==0 or height==0 or depth==0:
                print file_comp4,"width==0 or height==0 or depth==0,wrong and please check it!"
                return
            
            #bfind_one_space = False;
            for obj in treeA.findall('object'):
                xmlname = obj.find('name').text
                xmlname = xmlname.strip()
                for ifind,tmp_name in enumerate(src_list):
                    if xmlname==tmp_name:
                        i=i+1
                        print file_comp4
                        bfind_one_space = True
                        obj.find('name').text=des_list[ifind]
                        break
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
                treeA.write(out_xmlDir+file_comp4, encoding="utf-8",xml_declaration=False)
    print i,"===========over"



if __name__ == "__main__":
    # src_xml="C:\\Users\\ysc\\Desktop\\test\\wrong\\"
    # out_xmlDir="C:\\Users\\ysc\\Desktop\\test\\wrong\\11\\"
    # JPG_dir="C:\\Users\\ysc\\Desktop\\test\\wrong\\"
    src_xml="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/Annotations/"
    out_xmlDir="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/Annotations_combine/"
    JPG_dir="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/JPEGImages/"
    src_list=["nco7","w"] #change it , it can have any element
    des_list=["nco14","s"]
    src_list.append("Nestle 50")
    des_list.append("Nestle50")
    modify_xml_name(src_xml,out_xmlDir,src_list,des_list,JPG_dir)
