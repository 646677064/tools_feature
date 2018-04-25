# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:45:02 2017

@author: ysc
"""

#http://mp.weixin.qq.com/s/3QH9Tuim5yRX_F_yY3mtFQ
import cv2
import numpy as np
from numpy import *
from skimage import io,transform,data

import os
import shutil
#import skimage.io as io
#import sys
from math import *
import numpy as np
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement

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

def rotateimg2_manay(basefile,outdir,start):
    print "ssss"
    img=cv2.imread(basefile)
    flipped = cv2.flip(img, 1)
    cv2.imwrite(outdir+str(start)+".jpg", flipped)
    for i in range(0,361,30):
        print i
        img_dest = rotateimg2(img,i)
        start+=1
        cv2.imwrite(outdir+str(start)+".jpg", img_dest)

def rotate_save_as(basedir,xmldir,outdir):
    f_list = os.listdir(basedir)
    print basedir
    #picname = "IMG_1848.JPG"
    i=0
    for file_comp4 in f_list:
        #print file_comp4
        basename =os.path.splitext(file_comp4)[0]
        midlename = basename.split("_")[1]
        img=cv2.imread(basedir+file_comp4)
        rows,cols,ch = img.shape
        i +=1
        print file_comp4,rows,cols,ch
        if False:#len(midlename)==4:
            img_dest = rotateimg2(img,270)
            cv2.imwrite(outdir+basename+".jpg", img_dest)
            if not os.path.exists(xmldir+basename+".xml"):
                continue
            treeA = ElementTree()
            treeA.parse(xmldir+basename+".xml")
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            nodelist = treeA.findall("object")
            for obj in nodelist: 
                #num = classnum.get(node.find("name").text)
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                obj.find('bndbox').find('xmin').text = str(height - ymax)
                obj.find('bndbox').find('ymin').text = str(xmin)
                obj.find('bndbox').find('xmax').text = str(height - ymin)
                obj.find('bndbox').find('ymax').text = str(xmax)
            treeA.write(outdir+basename+".xml", encoding="utf-8",xml_declaration=False)
        else:
            img_dest = rotateimg2(img,270)
            cv2.imwrite(outdir+basename+".jpg", img_dest)
            if not os.path.exists(xmldir+basename+".xml"):
                continue
            shutil.copy(xmldir+basename+".xml",  outdir+basename+".xml")
    print "over,i=",i
    print 1
  
def rotate_save_many_name(basedir):
    listdic=["90","180","270","l_r_flip","t_b_flip"]
    listpath=[]
    for target_dir in listdic:
        tmpdir=os.path.join(basedir,target_dir)
        listpath.append(tmpdir)
    orignal_dir=os.path.join(basedir,"trainsimilary_data")
    f_list = os.listdir(orignal_dir)
    for file_comp4 in f_list:
        crop_path=os.path.join(orignal_dir,file_comp4)
        if os.path.isdir(crop_path):
            # for target_path_dir_1 in listpath:
            #     if not os.path.exists(os.path.join(target_path_dir_1,file_comp4)):
            #         os.mkdir(os.path.join(target_path_dir_1,file_comp4))
            f_list_sub1 = os.listdir(crop_path)
            for file_comp4_sub1 in f_list_sub1:
                sub1_path=os.path.join(crop_path,file_comp4_sub1)
                if os.path.isdir(sub1_path):
                    for subnametar,target_path_dir_2 in enumerate(listdic):
                        print os.path.join(sub1_path,file_comp4+target_path_dir_2)
                        if  os.path.exists(os.path.join(crop_path,file_comp4_sub1+target_path_dir_2)) and os.path.isdir(os.path.join(crop_path,file_comp4_sub1+target_path_dir_2)):
                        #     os.mkdir(os.path.join(target_path_dir_2,file_comp4,file_comp4_sub1))
                              os.rename(os.path.join(crop_path,file_comp4_sub1+target_path_dir_2),os.path.join(crop_path,file_comp4_sub1+"_"+target_path_dir_2))

def rotate_save_many(basedir):
    listdic=["90","180","270","l_r_flip","t_b_flip"]
    listpath=[]
    for target_dir in listdic:
        tmpdir=os.path.join(basedir,target_dir)
        listpath.append(tmpdir)
    orignal_dir=os.path.join(basedir,"trainsimilary_data")
    f_list = os.listdir(orignal_dir)
    for file_comp4 in f_list:
        crop_path=os.path.join(orignal_dir,file_comp4)
        if os.path.isdir(crop_path):
            for target_path_dir_1 in listpath:
                if not os.path.exists(os.path.join(target_path_dir_1,file_comp4)):
                    os.mkdir(os.path.join(target_path_dir_1,file_comp4))
            f_list_sub1 = os.listdir(crop_path)
            for file_comp4_sub1 in f_list_sub1:
                sub1_path=os.path.join(crop_path,file_comp4_sub1)
                if os.path.isdir(sub1_path):
                    for subnametar,target_path_dir_2 in enumerate(listpath):
                        if not os.path.exists(os.path.join(target_path_dir_2,file_comp4,file_comp4_sub1)):
                            os.mkdir(os.path.join(target_path_dir_2,file_comp4,file_comp4_sub1))
                        #os.rename(os.path.join(target_path_dir_2,file_comp4,file_comp4_sub1),os.path.join(target_path_dir_2,file_comp4,file_comp4_sub1+listdic[subnametar]))
                    # files= os.listdir(sub1_path)
                    # for sub_file in files:
                    #     basename =os.path.splitext(sub_file)[0]
                    #     print sub_file
                    #     img=cv2.imread(os.path.join(sub1_path,sub_file))
                    #     img_dest_90 = rotateimg2(img,90)
                    #     cv2.imwrite(os.path.join(listpath[0],file_comp4,file_comp4_sub1,basename+"__090.jpg"), img_dest_90)
                    #     img_dest_180 = rotateimg2(img,180)
                    #     cv2.imwrite(os.path.join(listpath[1],file_comp4,file_comp4_sub1,basename+"__180.jpg"), img_dest_180)
                    #     img_dest_270 = rotateimg2(img,270)
                    #     cv2.imwrite(os.path.join(listpath[2],file_comp4,file_comp4_sub1,basename+"__270.jpg"), img_dest_270)
                    #     flipped_l_r = cv2.flip(img, 1)
                    #     cv2.imwrite(os.path.join(listpath[3],file_comp4,file_comp4_sub1,basename+"__111.jpg"), flipped_l_r)
                    #     #cv2.imshow("Flipped Horizontally", flipped)
                    #     flipped_t_b = cv2.flip(img, 0)
                    #     cv2.imwrite(os.path.join(listpath[4],file_comp4,file_comp4_sub1,basename+"__000.jpg"), flipped_t_b)



def rotate_save_as1(basedir,xmldir,outdir,filelist_path):
    f_list = os.listdir(basedir)
    #picname = "IMG_1848.JPG"
    i=0
    #f = open(filelist_path, 'w')
    for file_comp4 in f_list:
        basename =os.path.splitext(file_comp4)[0]
        midlename = basename.split("_")[1]
        img=cv2.imread(basedir+file_comp4)
        # rows,cols,ch = img.shape
        i +=1
        #print file_comp4,rows,cols,ch
        if True:#len(midlename)==4:
            x=1
            #f.write(basename+"\n")
            img_dest = rotateimg2(img,270)
            cv2.imwrite(outdir+basename+".jpg", img_dest)
            # treeA = ElementTree()
            # treeA.parse(xmldir+basename+".xml")
            # width = int(treeA.find('size/width').text)
            # height = int(treeA.find('size/height').text)
            # depth = int(treeA.find('size/depth').text)
            # nodelist = treeA.findall("object")
            # for obj in nodelist: 
            #     #num = classnum.get(node.find("name").text)
            #     xmin = int(obj.find('bndbox').find('xmin').text)
            #     ymin = int(obj.find('bndbox').find('ymin').text)
            #     xmax = int(obj.find('bndbox').find('xmax').text)
            #     ymax = int(obj.find('bndbox').find('ymax').text)
            #     obj.find('bndbox').find('xmin').text = str(height - ymax)
            #     obj.find('bndbox').find('ymin').text = str(xmin)
            #     obj.find('bndbox').find('xmax').text = str(height - ymin)
            #     obj.find('bndbox').find('ymax').text = str(xmax)
            # treeA.write(outdir+basename+".xml", encoding="utf-8",xml_declaration=False)
        else:
            #cv2.imwrite(outdir+basename+".jpg", img)
            print file_comp4
            shutil.copy(xmldir+basename+".xml",  outdir+basename+".xml")
    #f.close()
    print "over,i=",i
    print 1

def cut_roi_from_image_2_onedir(JPEGImagesDir,AnnotationsDir,outDir_JPEGImages):
    print "==========="
    f_list = os.listdir(JPEGImagesDir)
    i=0
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == ".JPG" or os.path.splitext(file_comp4)[1] == ".jpg":
            basename =os.path.splitext(file_comp4)[0]
            im = cv2.imread(JPEGImagesDir+file_comp4)#jpegpath.format(imagename)
            #sp = im.shape
            #rows,cols,img_depth = img.shape
            img_height = im.shape[0]
            img_width = im.shape[1]
            img_depth = im.shape[2]
            if img_depth != 3 or img_height<=0 or img_depth<=0 :
                 print "depet wrong",file_comp4
            if os.path.splitext(file_comp4)[1] == '.png':
                file_comp4 = os.path.splitext(file_comp4)[0] + ".jpg"
            if not os.path.exists(AnnotationsDir+basename+".xml"):
                continue
            treeA = ElementTree()
            treeA.parse(AnnotationsDir+basename+".xml")
            width = int(treeA.find('size/width').text)
            height = int(treeA.find('size/height').text)
            depth = int(treeA.find('size/depth').text)
            nodelist = treeA.findall("object")
            i=0
            for obj in nodelist: 
                #num = classnum.get(node.find("name").text)
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                cropImg = im[ymin:ymax, xmin:xmax]
                cv2.imwrite(outDir_JPEGImages+"true_"+basename+"_"+str(i)+".jpg", cropImg)
                i+=1
    print "over,i=",i

        # cv2.imwrite(outDir_JPEGImages+"/w1_"+file_comp4, im)
        # print "w1_"+file_comp4

        # cropImg = im[0:img_height, 0:img_width/2]
        # cv2.imwrite(outDir_JPEGImages+"/w2_1_"+file_comp4, cropImg)
        # cropImg = im[0:img_height, img_width/2:img_width]
        # cv2.imwrite(outDir_JPEGImages+"/w2_2_"+file_comp4, cropImg)
        # print "w2_1_"+file_comp4
        # print "w2_2_"+file_comp4

        # cropImg = im[0:img_height, 0:img_width/3]
        # cv2.imwrite(outDir_JPEGImages+"/w3_1_"+file_comp4, cropImg)
        # cropImg = im[0:img_height, img_width/3:2*img_width/3]
        # cv2.imwrite(outDir_JPEGImages+"/w3_2_"+file_comp4, cropImg)
        # cropImg = im[0:img_height, 2*img_width/3:img_width]
        # cv2.imwrite(outDir_JPEGImages+"/w3_3_"+file_comp4, cropImg)
        # print "w3_1_"+file_comp4
        # print "w3_2_"+file_comp4
        # print "w3_3_"+file_comp4

        # cropImg = im[0:img_height, 0:img_width/4]
        # cv2.imwrite(outDir_JPEGImages+"/w4_1_"+file_comp4, cropImg)
        # cropImg = im[0:img_height, img_width/4:img_width/2]
        # cv2.imwrite(outDir_JPEGImages+"/w4_2_"+file_comp4, cropImg)
        # cropImg = im[0:img_height, img_width/2:3*img_width/4]
        # cv2.imwrite(outDir_JPEGImages+"/w4_3_"+file_comp4, cropImg)
        # cropImg = im[0:img_height, 3*img_width/4:img_width]
        # cv2.imwrite(outDir_JPEGImages+"/w4_4_"+file_comp4, cropImg)
        # print "w4_1_"+file_comp4
        # print "w4_2_"+file_comp4
        # print "w4_3_"+file_comp4
        # print "w4_4_"+file_comp4
def reverze_word(infile,outfile):
    print infile
    with  open(infile, 'r') as in_f:
        lines = in_f.readlines()
        splitlines1 = [x.replace('  ',' ') for x in lines]
        splitlines = [x.strip().split(' ') for x in splitlines1]
        print splitlines[0]
        #imagename,label=splitlines
        imagename =    [x[0] for x in splitlines]
        # for j in splitlines:
        #     print j[0]
        #     print j[1]
            # print splitlines[j][0]
            # print splitlines[j][1]
        label =    [x[1] for x in splitlines]
        with open(outfile, 'w') as o_f:
            for i,name in enumerate(imagename):
                print name
                o_f.write(imagename[i]+"_rev "+label[i][::-1]+"\n")
    return

def renamedir(JPEGImagesDir):
    f_list = os.listdir(JPEGImagesDir)
    for file_comp4 in f_list:
        if os.path.splitext(file_comp4)[1] == ".jpg":
            basename =os.path.splitext(file_comp4)[0]
            os.rename(JPEGImagesDir+basename+".jpg",JPEGImagesDir+basename+"_rev.jpg")

def testfile_width_height(filename):
    im = cv2.imread(filename)#jpegpath.format(imagename)
    #sp = im.shape
    #rows,cols,img_depth = img.shape
    img_height = im.shape[0]
    img_width = im.shape[1]
    img_depth = im.shape[2]
    print img_width,img_height,img_depth

if __name__=='__main__':
    basedir="C:\\Users\\ysc\\Desktop\\chouma\\totaldata\\totaldata\\"
    xmldir="C:\\Users\\ysc\\Desktop\\chouma\\totaldata\\"
    outdir="C:\\Users\\ysc\\Desktop\\chouma\\totaldata\\reverse\\"
    #testfile_width_height("C:\\Users\\ysc\\Desktop\\huojia\\1\\IMG_0024.JPG")
    xmldir="C:\\Users\\ysc\\Desktop\\huojia\\"
    outdir="C:\\Users\\ysc\\Desktop\\huojia\\right\\"
    #rotate_save_as1(xmldir+"1\\",xmldir,outdir,xmldir+"chouma_1.txt")
    #rotate_save_as1(xmldir+"2\\",xmldir,outdir,xmldir+"chouma_1.txt")
    basedir="F:\\trainsimilary_data\\"
    #rotate_save_many_name(basedir)
    # rotate_save_as1(xmldir+"chouma_2\\",xmldir,outdir,xmldir+"chouma_2.txt")
    # rotate_save_as1(xmldir+"chouma_3\\",xmldir,outdir,xmldir+"chouma_3.txt")
    # rotate_save_as1(xmldir+"chouma_4\\",xmldir,outdir,xmldir+"chouma_4.txt")
    # reverze_word(xmldir+"chouma_1.txt",xmldir+"chouma_1_reverse.txt")
    # reverze_word(xmldir+"chouma_2.txt",xmldir+"chouma_2_reverse.txt")
    # reverze_word(xmldir+"chouma_3.txt",xmldir+"chouma_3_reverse.txt")
    # reverze_word(xmldir+"chouma_4.txt",xmldir+"chouma_4_reverse.txt")
    # renamedir(outdir)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\file\\101.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1000)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\file\\102.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1020)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\file\\103.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1040)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\file\\104.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1060)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\file\\107.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1080)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\file\\111.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1100)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q0\\g6aaa.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1120)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q0\\g611alps_1_0.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1140)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q1\\g7.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1160)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q1\\g7_12.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1180)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q5\\g11.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1200)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q6\\g3500ml_1.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1220)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q6\\g3500ml_148.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1240)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q81\\g5IMG_2120_0.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1260)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q82\\g4400ml_126.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1280)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q83\\g24.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1300)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q83\\g2_15.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1320)
    # rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q83\\g23.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1340)
    #rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q6\\g3500ml_12.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1360) #g23_1
    #rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q83\\g23_1.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1380)
    #rotateimg2_manay("C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test\\query\\q1\\g114.jpg","C:\\Users\\ysc\\Desktop\\caffe-reid-master\\work\\test6\\aaa\\",1420)
    #cut_roi_from_image(outdir,outdir,roi_dir)
    # detect_jpg_dir = "C:\\Users\\ysc\\Desktop\\2\\"
    # detect_xmldir = "C:\\Users\\ysc\\Desktop\\2\\"
    # detect_roi_dir = "C:\\Users\\ysc\\Desktop\\2\\roitrue\\"
    # #cut_roi_from_image(detect_jpg_dir,detect_xmldir,detect_roi_dir)
    # detect_jpg_dir1 = "C:\\Users\\ysc\\Desktop\\new_detection\\jpg\\"
    # detect_xmldir1 = "C:\\Users\\ysc\\Desktop\\new_detection\\xml\\"
    # detect_roi_dir1 = "F:\\ccc\\roi\\roi_classfication\\"
    # todir_pre = "F:\\ccc\\roi\\roi_classfication"
    # listJpg = os.listdir(detect_roi_dir1)
    i=1
    j=1
    #cut_roi_from_image(detect_jpg_dir1,detect_xmldir1,detect_roi_dir1)
    # for filename in listJpg:
    #     basename =os.path.splitext(filename)[0]
    #     todir = todir_pre+str(i)+"\\"
    #     shutil.move(todir_pre+"\\"+filename,todir+filename)
    #     j+=1
    #     if j ==2166:
    #         j=1
    #         i+=1