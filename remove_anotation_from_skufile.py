
import argparse
from collections import OrderedDict
#from google.protobuf import text_format
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
#import skimage.io as io
import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
import cPickle
import random
import math
import sys,os,subprocess,commands
from subprocess import Popen,PIPE

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
                        #   oobj.remove(ooobj)
                        # for ooobj in ymin_s:
                        #   oobj.remove(ooobj)
                        # for ooobj in xmax_s:
                        #   oobj.remove(ooobj)
                        # for ooobj in ymax_s:
                        #   oobj.remove(ooobj)
                        obj.remove(oobj)
                    rootA.remove(obj)
            if bfind_one_space==True:
                #print file_comp4
                #print treeA
                treeA.write(outDir+file_comp4, encoding="utf-8",xml_declaration=False)
    print i

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

def find_and_remove_cp_file(skufile,dir_1,outDir):
    with open(skufile,"r") as sku_f:
        sku_lines_1 = sku_f.readlines();
    test_sku=[]
    for i_sku,test_sku1 in enumerate(sku_lines_1):
        #for i_file,test_file in enumerate(test_filelist_lines):
        #print test_sku
        test_sku1 = test_sku1.strip().strip('\n').strip('\r')
        test_sku.append(test_sku1)

    f_list = os.listdir(dir_1+"/Annotations/")
    for file_comp4 in f_list:
        #print fname
        basename=os.path.splitext(file_comp4)[0]
        orig_xml=dir_1+"/Annotations/"+basename+".xml"
        orig_pic=dir_1+"/JPEGImages/"+basename+".jpg"
        # ppsring= "cp "+orig_xml+" "+des_dir+"/Annotations/"
        # assert Popen_do(ppsring),ppsring+" error!"
        # ppsring= "cp "+orig_pic+" "+des_dir+"/JPEGImages/"
        # assert Popen_do(ppsring),ppsring+" error!"
        if os.path.splitext(file_comp4)[1] == '.xml':
            file_tmp = dir_1+"/Annotations/"+file_comp4
            treeA=ElementTree()
            treeA.parse(file_tmp)
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            if width==0 or height==0 or depth==0:
                print file_comp4,"width==0 or height==0 or depth==0"
                break
            bfind_one_space = False;
            rootA=treeA.getroot()
            #print rootA.tag
            children = rootA.findall('object')
            for obj in children:
                xmlname = obj.find('name').text
                if xmlname in test_sku:
                    bfind_one_space=True
                    break
            if bfind_one_space==True:
                for obj in children:
                    xmlname = obj.find('name').text
                    if xmlname not in test_sku:
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
                            #   oobj.remove(ooobj)
                            # for ooobj in ymin_s:
                            #   oobj.remove(ooobj)
                            # for ooobj in xmax_s:
                            #   oobj.remove(ooobj)
                            # for ooobj in ymax_s:
                            #   oobj.remove(ooobj)
                            obj.remove(oobj)
                        rootA.remove(obj)
            if bfind_one_space==True:
                #print file_comp4
                #print treeA
                treeA.write(outDir+"/Annotations/"+file_comp4, encoding="utf-8",xml_declaration=False)
                orig_pic=dir_1+"//JPEGImages/"+basename+".jpg"
                ppsring= "cp "+orig_pic+" "+outDir+"/JPEGImages/"
                assert Popen_do(ppsring),ppsring+" error!"

def cp_200_file(indir,name,outdir="/storage2/for_gs4/wangyinzhi_data/"):
    proj1name="/"+name+"proj66/"
    indir_path=indir+"/"+name+proj1name+"/JPEGImages/"
    bexist = os.path.exists(indir_path)
    if False==bexist:
        proj1name="/"+name+"proj2/"
        indir_path=indir+"/"+name+proj1name+"/JPEGImages/"
    f_list = os.listdir(indir_path)
    bexist = os.path.exists(outdir+"/"+name)
    if False==bexist:
        os.makedirs(outdir+"/"+name)
    for i_file,test_file in enumerate(f_list):
        orig_pic=indir_path+test_file
        ppsring= "cp "+orig_pic+" "+outdir+"/"+name+"/"
        assert Popen_do(ppsring),ppsring+" error!"
        if i_file>200:
            break

def cp_200_66329_file(indir,name,outdir="/storage2/for_gs4/wangyinzhi_data/"):
    proj1name="/"+name+"proj329/"
    indir_path=indir+"/"+name+proj1name+"/JPEGImages/"
    bexist = os.path.exists(indir_path)
    if False==bexist:
        proj1name="/"+name+"proj2/"
        indir_path=indir+"/"+name+proj1name+"/JPEGImages/"
    f_list = os.listdir(indir_path)
    bexist = os.path.exists(outdir+"/"+name+"proj329/")
    if False==bexist:
        os.makedirs(outdir+"/"+name+"proj329/")
    for i_file,test_file in enumerate(f_list):
        orig_pic=indir_path+test_file
        ppsring= "cp "+orig_pic+" "+outdir+"/"+name+"proj329/"
        assert Popen_do(ppsring),ppsring+" error!"
        if i_file>200:
            break

if __name__ == "__main__":
    dir_1="/storage2/tiannuodata/work/projdata/baiwei0317-2472-1/baiwei0317-2472-1proj1//"
    dir_2="/storage2/tiannuodata/work/projdata/baiwewi0301-2323/baiwewi0301-2323proj1//"
    des_dir="/home/liushuai/medical/kele/keleproj1/"
    test_filepath="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj1//ImageSets/Main/test.txt"
    # pp_stirng = "mv  "+unzip_dir+"/*.jpg "+unzip_dir+"/JPEG/"
    # assert Popen_do(pp_stirng),pp_stirng+" error!"

    # with open(test_filepath,"r") as f:
    #     test_filelist_lines = f.readlines();
    # test_list=[]
    # for i_file,test_file in enumerate(test_filelist_lines):
    #     #for i_file,test_file in enumerate(test_filelist_lines):
    #     print test_file
    #     temp_file = test_file.strip().strip('\n').strip('\r')
    #     test_list.append(temp_file)

    # f_list = os.listdir(dir_1+"/Annotations/")
    # for fname in f_list:
    #     if fname in test_list:
    #         continue
    #     #print fname
    #     basename=os.path.splitext(fname)[0]
    #     orig_xml=dir_1+"/Annotations/"+basename+".xml"
    #     orig_pic=dir_1+"/JPEGImages/"+basename+".jpg"
    #     ppsring= "cp "+orig_xml+" "+des_dir+"/Annotations/"
    #     assert Popen_do(ppsring),ppsring+" error!"
    #     ppsring= "cp "+orig_pic+" "+des_dir+"/JPEGImages/"
    #     assert Popen_do(ppsring),ppsring+" error!"

    # f_list = os.listdir(dir_2+"/Annotations/")
    # for fname in f_list:
    #     if fname in test_list:
    #         continue
    #     basename=os.path.splitext(fname)[0]
    #     orig_xml=dir_2+"/Annotations/"+basename+".xml"
    #     orig_pic=dir_2+"/JPEGImages/"+basename+".jpg"
    #     ppsring= "cp "+orig_xml+" "+des_dir+"/Annotations/"
    #     assert Popen_do(ppsring),ppsring+" error!"
    #     ppsring= "cp "+orig_pic+" "+des_dir+"/JPEGImages/"
    #     assert Popen_do(ppsring),ppsring+" error!"

        # tmp= test_file.split(' ')
        # if 2<len(tmp):
        #     #print test_file.split(' ')[2]
        #     ppsring= "cp "+tmp[2]+" "+des_dir
        #     assert Popen_do(ppsring),ppsring+" error!"
    AnnotationDir="/home/liushuai/medical/kele/keleproj1/Annotations_package/"
    outDir="/home/liushuai/medical/kele/keleproj1/Annotations/"
    #remove_Anotations(AnnotationDir,outDir)

    AnnotationDir="/home/liushuai/medical/kele/keleproj1/Annotations_package/"

    skufile="/storage2/tiannuodata/work/projdata/baiwei/193/skufile.txt"
    dir_1="/storage2/tiannuodata/work/projdata/baiwei0606-2450-1/baiwei0606-2450-1proj1//"
    outDir="/storage2/tiannuodata/work/projdata/baiwei/193/"
    #find_and_remove_cp_file(skufile,dir_1,outDir)
    
    skufile="/storage2/tiannuodata/work/projdata/baiwei/329/skufile.txt"
    dir_1="/storage2/tiannuodata/work/projdata/baiwei0606-2450-1/baiwei0606-2450-1proj1//"
    outDir="/storage2/tiannuodata/work/projdata/baiwei/329/"

    skufile="/storage2/tiannuodata/work/projdata/baiwei/329_tmp/skufile.txt"
    dir_1="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2//"
    outDir="/storage2/tiannuodata/work/projdata/baiwei//329_tmp/"
    find_and_remove_cp_file(skufile,dir_1,outDir)

    # skufile="/storage2/tiannuodata/work/projdata/baiwei/66/66.txt"
    # dir_1="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2//"
    # outDir="/storage2/tiannuodata/work/projdata/baiwei/66/"
    # find_and_remove_cp_file(skufile,dir_1,outDir)

    indir="/storage2/tiannuodata/work/projdata/"
    name_list=[]
    # name_list.append("shushida")
    # name_list.append("nielsenchips")
    # name_list.append("nestlecoffee")
    # name_list.append("nestle4goods")
    # name_list.append("nersen")
    # name_list.append("kele")
    #name_list.append("baiwei")
    # name_list.append("shape")

    # for name in name_list:
    #     #cp_200_file(indir,name)
    #     cp_200_66329_file(indir,name)

    # indir="/storage2/liushuai/data/data/"
    # name_list=[]
    # # name_list.append("nestlericeflour")
    # # name_list.append("nestleoatmeal")
    # # name_list.append("nestlemilkpowder")
    # # name_list.append("nestlemilk")
    # # name_list.append("nestlebiscuit")
    # # name_list.append("nestlecoffee")
    # name_list.append("milkpowder")
    # name_list.append("extra2")
    # name_list.append("cookie")
    # name_list.append("colgate")
    # name_list.append("chutty")
    # name_list.append("beer")

    # for name in name_list:
    #     cp_200_file(indir,name)
