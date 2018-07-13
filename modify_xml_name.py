
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
    command = "rsync -vp "+src_xml+"*.xml "+out_xmlDir
    print command
    os.system(command)
    f_list = os.listdir(src_xml)
    i=0
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == '.xml':
            file_tmp = src_xml+"/"+file_comp4
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
    # src_list=["nco7","w"] #change it , it can have any element
    # des_list=["nco14","s"]
    # src_list.append("Nestle 50")
    # des_list.append("Nestle50")    
    src_list=[] #change it , it can have any element
    des_list=[]
    src_list.append("budweiser4")
    des_list.append("budweiser33")#1

    src_list.append("budweiser19")
    des_list.append("budweiser18")
    src_list.append("budweiser21")
    des_list.append("budweiser18")
    src_list.append("budweiser94")
    des_list.append("budweiser18")#2

    src_list.append("budweiser26")
    des_list.append("budweiser18")
    src_list.append("budweiser28")
    des_list.append("budweiser18")#1

    src_list.append("budweiser12")
    des_list.append("budweiser11")
    src_list.append("budweiser53")
    des_list.append("budweiser11")
    src_list.append("budweiser89")
    des_list.append("budweiser11")#2

    src_list.append("budweiser38")
    des_list.append("budweiser37")#1

    src_list.append("budweiser20")
    des_list.append("budweiser60")#1

    src_list.append("budweiser5")
    des_list.append("budweiser35")#1

    src_list.append("budweiser15")
    des_list.append("budweiser14")#1

    src_list.append("budweiser31")
    des_list.append("budweiser30")#1

    src_list.append("budweiser75")
    des_list.append("budweiser67")#1

    src_list.append("budweiser48")
    des_list.append("budweiser47")#1

    src_list.append("hoegaarden3")
    des_list.append("hoegaarden12")#1

    src_list.append("snow13")
    des_list.append("snow1")#1

    src_list.append("snow31")
    des_list.append("snow16")#1

    src_list.append("snow129")
    des_list.append("snow128")#1

    src_list.append("snow49")
    des_list.append("snow10")#1
    src_list.append("snow95")
    des_list.append("snow32")#1
    src_list.append("snow113")
    des_list.append("snow112")#1
    src_list.append("snow130")
    des_list.append("snow39")#1
    src_list.append("snow179")
    des_list.append("snow178")#1
    src_list.append("snow218")
    des_list.append("snow152")#1
    src_list.append("snow198")
    des_list.append("snow144")#1
    src_list.append("snow19")
    des_list.append("snow59")#1
    src_list.append("snow70")
    des_list.append("snow14")#1
    src_list.append("snow223")
    des_list.append("snow222")#1
    src_list.append("snow174")
    des_list.append("snow28")#1
    src_list.append("suntory16")
    des_list.append("suntory1")#1
    src_list.append("suntory24")
    des_list.append("suntory19")#1
    src_list.append("suntory29")
    des_list.append("suntory25")#1

    src_list.append("suntory42")
    des_list.append("suntory32")#1
    src_list.append("tiger8")
    des_list.append("tiger4")#1
    src_list.append("tsingtao16")
    des_list.append("tsingtao14")#1
    src_list.append("tsingtao122")
    des_list.append("tsingtao19")#1
    src_list.append("tsingtao31")
    des_list.append("tsingtao24")#1
    src_list.append("tsingtao43")
    des_list.append("tsingtao42")#1
    src_list.append("tsingtao126")
    des_list.append("tsingtao83")#1
    src_list.append("tsingtao86")
    des_list.append("tsingtao85")#1
    src_list.append("tsingtao127")
    des_list.append("tsingtao119")#1
    src_list.append("tsingtao159")
    des_list.append("tsingtao134")#1
    src_list.append("tsingtao165")
    des_list.append("tsingtao156")#1
    src_list.append("tsingtao213")
    des_list.append("tsingtao50")#1
    src_list.append("tsingtao190")
    des_list.append("tsingtao101")#1

    src_list.append("tsingtao62")
    des_list.append("tsingtao58")#1
    src_list.append("tsingtao208")
    des_list.append("tsingtao66")#1
    src_list.append("tsingtao240")
    des_list.append("tsingtao155")#1
    src_list.append("tsingtao240")
    des_list.append("tsingtao155")#1

    src_list.append("tsingtao27")
    des_list.append("tsingtao26")#1
    src_list.append("tsingtao56")
    des_list.append("tsingtao26")#1
    src_list.append("tsingtao57")
    des_list.append("tsingtao26")#1

    src_list.append("harbin103")
    des_list.append("harbin30")#1
    src_list.append("harbin58")
    des_list.append("harbin30")#1
    src_list.append("harbin60")
    des_list.append("harbin30")#1

    src_list.append("harbin131")
    des_list.append("harbin89")#1
    src_list.append("harbin29")
    des_list.append("harbin89")#1
    src_list.append("harbin31")
    des_list.append("harbin89")#1

    src_list.append("harbin107")
    des_list.append("harbin104")#1

    src_list.append("harbin34")
    des_list.append("harbin33")#1

    src_list.append("harbin5")
    des_list.append("harbin2")#1

    src_list.append("harbin12")
    des_list.append("harbin65")#1

    src_list.append("harbin32")
    des_list.append("harbin73")#1

    src_list.append("landai6")
    des_list.append("landai4")#1
    src_list.append("landai19")
    des_list.append("landai2")#1
    src_list.append("landai13")
    des_list.append("landai12")#1

    src_list.append("corona11")
    des_list.append("corona10")#1
    src_list.append("becks6")
    des_list.append("becks1")#1
    src_list.append("sedrin17")
    des_list.append("sedrin15")#1
    src_list.append("sedrin8")
    des_list.append("sedrin6")#1

    src_list.append("sedrin22")
    des_list.append("sedrin19")#1
    src_list.append("sedrin25")
    des_list.append("sedrin2")#1
    src_list.append("sedrin51")
    des_list.append("sedrin45")#1
    src_list.append("sedrin41")
    des_list.append("sedrin40")#1
    src_list.append("sedrin3")
    des_list.append("sedrin1")#1
    src_list.append("sedrin9")
    des_list.append("sedrin7")#1
    src_list.append("sedrin16")
    des_list.append("sedrin14")#1

    src_list.append("carlsberg14")
    des_list.append("carlsberg7")#1
    src_list.append("carlsberg20")
    des_list.append("carlsberg11")#1
    src_list.append("carlsberg2")
    des_list.append("carlsberg30")#1
    src_list.append("carlsberg13")
    des_list.append("carlsberg3")#1

    src_list.append("heineken6")
    des_list.append("heineken5")#1
    src_list.append("heineken10")
    des_list.append("heineken19")#1
    src_list.append("heineken17")
    des_list.append("heineken1")#1
    src_list.append("heineken18")
    des_list.append("heineken16")#1


    src_list.append("laoshan2")
    des_list.append("laoshan1")#1

    src_list.append("laoshan19")
    des_list.append("laoshan5")#1
    src_list.append("laoshan41")
    des_list.append("laoshan33")#1
    src_list.append("laoshan11")
    des_list.append("laoshan8")#1
    src_list.append("laoshan46")
    des_list.append("laoshan9")#1
    src_list.append("laoshan10")
    des_list.append("laoshan23")#1

    src_list.append("yanjing97")
    des_list.append("yanjing96")#1
    src_list.append("yanjing15")
    des_list.append("yanjing13")#1
    src_list.append("yanjing32")
    des_list.append("yanjing7")#1
    src_list.append("yanjing19")
    des_list.append("yanjing6")#1
    src_list.append("yanjing9")
    des_list.append("yanjing68")#1

    src_list.append("yanjing16")
    des_list.append("yanjing3")#1
    src_list.append("yanjing21")
    des_list.append("yanjing4")#1

    src_list.append("tsingtao70")
    des_list.append("tsingtao6")#1
    src_list.append("tsingtao242")
    des_list.append("tsingtao203")#1
    modify_xml_name(src_xml,out_xmlDir,src_list,des_list,JPG_dir)
