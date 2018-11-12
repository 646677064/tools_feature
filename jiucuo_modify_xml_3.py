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
import   cv2
# import skimage.io
# from scipy.ndimage import zoom
# from skimage.transform import resize

  
  
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

def jiucuo_modify_xml_3(Anotation_dir,JPG_dir,patchdir,out_xmlDir,out_jpgdir):
    result_patchlist=patchdir+"/result_patchlist/"
    if not os.path.exists(out_jpgdir):
        os.mkdir(out_jpgdir)
    if not os.path.exists(out_xmlDir):
        os.mkdir(out_xmlDir)
    subpatchs = os.listdir(result_patchlist)
    for subpatch in subpatchs:
        if subpatch=="patchlist":
            continue
        if subpatch=="result_patchlist":
            continue
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
                rootA=treeA.getroot()
                children = rootA.findall('object')
                for obj in children:
                    xmlname = obj.find('name').text
                    xmlname = xmlname.strip()
                    xmin = int(obj.find('bndbox').find('xmin').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    if xmin_target==xmin and ymin_target == ymin and xmax_target==xmax and ymax_target==ymax:
                        obj.find('name').text=subpatch
                        bfind_one_space=True
                        if subpatch=="others"  or  subpatch=="Others" :
                            jpg_file=out_jpgdir+name+".jpg"
                            if False==os.path.exists(jpg_file):
                                jpg_file=JPG_dir+name+".jpg"
                            if os.path.exists(jpg_file):
                                im = cv2.imread(jpg_file)
                                im[ymin:ymax, xmin:xmax]=np.zeros((ymax-ymin)*(xmax-xmin)*3).reshape(ymax-ymin,xmax-xmin,3)
                                cv2.imwrite(out_jpgdir+name+".jpg", im)#test ,need to rewrite the orignal path jpg
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
                        break
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
                    jpg_file=out_jpgdir+name+".jpg"
                    if False==os.path.exists(jpg_file):
                        ppsring= "cp "+JPG_dir+name+".jpg "+out_jpgdir+name+".jpg"
                        if os.path.exists(JPG_dir+name+".jpg"):
                            assert Popen_do(ppsring),ppsring+" error!"

def only_getpatchlist(patchdir):
    listdir=patchdir+"/patchlist/"
    if not os.path.exists(listdir):
        os.mkdir(listdir)
    subpatchs = os.listdir(patchdir)
    for subpatch in subpatchs:
        subpatch=os.path.splitext(subpatch)[0]
        if subpatch=="patchlist":
            continue
        if subpatch=="result_patchlist":
            continue
        listfile = listdir+ subpatch+".txt"
        listw = open(listfile, 'w')
        subpaths = patchdir+subpatch
        print listfile,subpaths
        files = os.listdir(subpaths)
        for file in files:
            listw.write(file + '\n')
        listw.close()

def jiucuo_modify_xml_3_diff_file(Anotation_dir,JPG_dir,patchdir,out_xmlDir,out_jpgdir):
    if not os.path.exists(out_jpgdir):
        os.mkdir(out_jpgdir)
    if not os.path.exists(out_xmlDir):
        os.mkdir(out_xmlDir)
    subpatchs = os.listdir(patchdir+"/result_patchlist/")
    for subpatch in subpatchs:
        subpatch=os.path.splitext(subpatch)[0]
        print subpatch
        # if subpatch!="Others":
        #     break
        if subpatch=="patchlist":
            continue
        if subpatch=="result_patchlist":
            continue
        # if subpatch=="patchlist":
        #     continue
        # if subpatch=="result_patchlist":
        #     continue
        list_results=[]
        with open(patchdir+"/result_patchlist/"+subpatch+".txt","r") as f_result:
            lines_result = f_result.readlines()
        list_results=[labelname_result.strip().strip('\n').strip('\r') for labelname_result in lines_result]
        list_orignal=[]
        if os.path.exists(patchdir+"/patchlist/"+subpatch+".txt"):
            with open(patchdir+"/patchlist/"+subpatch+".txt","r") as f_orignal:
                lines_orignal = f_orignal.readlines()
                list_orignal = [labelname.strip().strip('\n').strip('\r') for labelname in lines_orignal]
        for file in list_results:
            if file not in list_orignal:
                file=os.path.splitext(file)[0]
                splitthins=file.split("_")
                name=splitthins[0]
                if len(splitthins)>5:
                    for ia in range(1,len(splitthins)-4):
                        name=name+"_"+splitthins[ia]
                ymax_target=int(splitthins[-1])
                xmax_target=int(splitthins[-2])
                ymin_target=int(splitthins[-3])
                xmin_target=int(splitthins[-4])
                file_tmp = Anotation_dir+name+".xml"
                if True==os.path.exists(out_xmlDir+name+".xml"):
                    file_tmp=out_xmlDir+name+".xml"
                print file_tmp
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
                rootA=treeA.getroot()
                children = rootA.findall('object')
                for obj in children:
                    xmlname = obj.find('name').text
                    xmlname = xmlname.strip()
                    xmin = int(obj.find('bndbox').find('xmin').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    #print xmin_target,xmin,ymin_target,ymin,xmax_target,xmax,ymax_target,ymax
                    if xmin_target==xmin and ymin_target == ymin and xmax_target==xmax and ymax_target==ymax:
                        #print xmin_target,xmin,ymin_target,ymin,xmax_target,xmax,ymax_target,ymax
                        obj.find('name').text=subpatch
                        bfind_one_space=True
                        #print name
                        if subpatch=="others" or  subpatch=="Others" :
                            jpg_file=out_jpgdir+name+".jpg"
                            if False==os.path.exists(jpg_file):
                                jpg_file=JPG_dir+name+".jpg"
                            #print jpg_file
                            if os.path.exists(jpg_file):
                                im = cv2.imread(jpg_file)
                                im[ymin:ymax, xmin:xmax]=np.zeros((ymax-ymin)*(xmax-xmin)*3).reshape(ymax-ymin,xmax-xmin,3)
                                cv2.imwrite(out_jpgdir+name+".jpg", im)#test ,need to rewrite the orignal path jpg
                            else:
                                print name,".jpg not exists!!!!!!!!!!!!!!!!!!!!!!!!!!!"
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
                        break
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
                    jpg_file=out_jpgdir+name+".jpg"
                    if False==os.path.exists(jpg_file):
                        ppsring= "cp "+JPG_dir+name+".jpg "+out_jpgdir+name+".jpg"
                        if os.path.exists(JPG_dir+name+".jpg"):
                            assert Popen_do(ppsring),ppsring+" error!"
def get_others_to_patch_result(patchdir):
    #os.mkdir(out_xmlDir)
    subpatchs = os.listdir(patchdir+"/patchlist/")
    all_file=[]
    for subpatch in subpatchs:
        # subpatch=os.path.splitext(subpatch)[0]
        # print subpatch
        # # if subpatch!="Others":
        # #     break
        # if subpatch=="patchlist":
        #     continue
        # if subpatch=="result_patchlist":
        #     continue
        # # if subpatch=="patchlist":
        # #     continue
        # # if subpatch=="result_patchlist":
        # #     continue
        # list_results=[]
        with open(patchdir+"/patchlist/"+subpatch,"r") as f_result:
            lines_result = f_result.readlines()
        list_results=[labelname_result.strip().strip('\n').strip('\r') for labelname_result in lines_result]
        all_file = all_file + list_results
    fff_list = os.listdir(patchdir)
    all_changefile=[]
    for subpatch in fff_list:
        if subpatch=="patchlist":
            continue
        if subpatch=="result_patchlist":
            continue
        localfile=os.listdir(patchdir+subpatch)
        all_changefile= all_changefile+localfile
    diff_list=[]
    for ff in all_file:
        if ff not in all_changefile:
            diff_list.append(ff+"\n")
    with open(patchdir+"/result_patchlist/others.txt","w") as f_others:
        f_others.writelines(diff_list)

if __name__=="__main__":
    JPG_dir="/data/tiannuodata/nestle4goods/nestle4goodsproj1///JPEGImages/"
    Anotation_dir="/data/tiannuodata/nestle4goods/nestle4goodsproj1///Annotations/"
    patchdir="/data/tiannuodata/nestle4goods/nestle4goodsproj1/patch//"
    out_jpgdir="/data/tiannuodata/nestle4goods/nestle4goodsproj1//output/jpg/"
    out_xmlDir="/data/tiannuodata/nestle4goods/nestle4goodsproj1//output/xml/"
    #jiucuo_modify_xml_3(Anotation_dir,JPG_dir,patchdir,out_xmlDir,out_jpgdir)
    #only_getpatchlist(patchdir)
    get_others_to_patch_result(patchdir)
    jiucuo_modify_xml_3_diff_file(Anotation_dir,JPG_dir,patchdir,out_xmlDir,out_jpgdir)