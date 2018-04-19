
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
import skimage.io as io
import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from xml.etree.ElementTree import SubElement
import cPickle
import random
import math
import sys,os,subprocess,commands
from subprocess import Popen,PIPE

def Popen_do(pp_string,b_pip_stdout=True):
    print pp_string
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

def deal_data(data_dir,out_file_list):
    iclass = 0
    inum = 0
    #irename_class = 0
    irename_num = 0
    with open(out_file_list,"w") as out_f:
        dic_list = os.listdir(data_dir)
        dic_list.sort()
        for file_comp4 in dic_list:
            if True==os.path.isdir(os.path.join(data_dir,file_comp4)):
                #print os.path.join(data_dir,file_comp4)
                tmp_name=file_comp4
                tmp_name=tmp_name.replace(' ','')
                #print data_dir+file_comp4+"/",tmp_name
                os.rename(os.path.join(data_dir,file_comp4),os.path.join(data_dir,tmp_name))
                # tmp_dir = os.path.join(data_dir,file_comp4)
                # pp_string="mv "+data_dir+file_comp4+" "+data_dir+tmp_name
                # assert Popen_do(pp_string),pp_string+" error!!"


        dic_list = os.listdir(data_dir)
        dic_list.sort()
        for file_comp4 in dic_list:
            if True==os.path.isdir(os.path.join(data_dir,file_comp4)):
                dic_list_2 = os.listdir(os.path.join(data_dir,file_comp4))
                dic_list_2.sort()
                for file_comp_2 in dic_list_2:
                    #print os.path.join(data_dir+file_comp4+"/",file_comp_2)
                    if True==os.path.isdir(os.path.join(data_dir+file_comp4+"/",file_comp_2)):
                        tmp_name_2=file_comp_2
                        #print file_comp_2
                        tmp_name_2=tmp_name_2.replace(' ','')
                        print data_dir+file_comp4+"/",tmp_name_2
                        os.rename(os.path.join(data_dir+file_comp4+"/",file_comp_2),os.path.join(data_dir+file_comp4+"/",tmp_name_2))
                        # pp_string="mv "+data_dir+file_comp4+"/"+file_comp_2+" "+data_dir+file_comp4+"/"+tmp_name_2
                        # assert Popen_do(pp_string),pp_string+" error!!"


        # dic_list = os.listdir(data_dir)
        # for file_comp4 in dic_list:
        #     if True==os.path.isdir(data_dir+file_comp4):
                dic_list_2 = os.listdir(os.path.join(data_dir,file_comp4))
                dic_list_2.sort()
                for file_comp_2 in dic_list_2:
                    dir_2=data_dir+file_comp4+"/"+file_comp_2+"/"
                    if True==os.path.isdir(dir_2):
                        file_list_3 = os.listdir(dir_2)
                        file_list_3.sort()
                        for file_3 in file_list_3:
                            irename_num+=1
                            if True==os.path.isfile(os.path.join(dir_2,file_3)):
                                tmp_name_3=file_3
                                tmp_name_3=tmp_name_3.replace(' ','')
                                os.rename(os.path.join(dir_2,file_3),os.path.join(dir_2,str(iclass)+"_"+file_comp_2+str(irename_num)+".jpg"))
                                # pp_string="mv "+dir_2+file_3+" "+dir_2+tmp_name_3
                                # assert Popen_do(pp_string),pp_string+" error!!"

                        file_list_3 = os.listdir(dir_2)
                        file_list_3.sort()
                        for file_3 in file_list_3:
                            if True==os.path.isfile(dir_2+file_3):
                                #print dir_2+file_3," ",iclass
                                file_name = dir_2+file_3+" "+str(iclass)+"\n"
                                out_f.write(file_name)
                                inum+=1
                        iclass+=1

if __name__ == "__main__":
    data_dir="/storage2/liushuai/data/similary_data/new_trainsimilary/"
    out_file_list="/storage2/liushuai/data/similary_data/new_trainsimilary/list_file.txt"
    deal_data(data_dir,out_file_list)