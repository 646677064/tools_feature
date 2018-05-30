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
# # from fast_rcnn.config import cfg
# # from fast_rcnn.test import im_detect
# # from fast_rcnn.nms_wrapper import nms
# #from utils.timer import Timer
# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import ElementTree,Element
# from xml.etree.ElementTree import SubElement
# # import matplotlib.pyplot as plt
# # import numpy as np
# #import scipy.io as sio
# # import   cv2
# # import skimage.io
# # from scipy.ndimage import zoom
# # from skimage.transform import resize

  

def get_result_patchlist_windows(patchdir):
    if not os.path.exists(patchdir+"\\result_patchlist\\"):
        os.mkdir(patchdir+"\\result_patchlist\\")
    subpatchs = os.listdir(patchdir)
    for subpatch in subpatchs:
        if subpatch=="patchlist":
            continue
        if subpatch=="result_patchlist":
            continue
        with open(patchdir+"\\patchlist\\"+subpatch+".txt","r") as f_r:
            lines = f_r.readlines()
        with open(patchdir+"\\result_patchlist\\"+subpatch+".txt","w") as f_w:
            last_lines = [labelname.strip().strip('\n').strip('\r') for labelname in lines]
            file_lists=os.listdir(patchdir+subpatchs)
            for file in file_lists:
                if file in last_lines:
                    continue
                f_w.write(file+"\n")

def get_increase_result_patchlist(patchdir):
    if not os.path.exists(patchdir+"/result_patchlist/"):
        os.mkdir(patchdir+"/result_patchlist/")
    subpatchs = os.listdir(patchdir)
    for subpatch in subpatchs:
        if subpatch=="patchlist":
            continue
        if subpatch=="result_patchlist":
            continue
        with open(patchdir+"/patchlist/"+subpatch+".txt","r") as f_r:
            lines = f_r.readlines()
        with open(patchdir+"/result_patchlist/"+subpatch+".txt","w") as f_w:
            last_lines = [labelname.strip().strip('\n').strip('\r') for labelname in lines]
            file_lists=os.listdir(patchdir+subpatchs)
            for file in file_lists:
                if file in last_lines:
                    continue
                f_w.write(file+"\n")


def only_getpatchlist(patchdir):
    result_listdir=patchdir+"/result_patchlist/"
    if not os.path.exists(result_listdir):
        os.mkdir(result_listdir)
    subpatchs = os.listdir(patchdir)
    for subpatch in subpatchs:
        listfile = result_listdir+ subpatch+".txt"
        listw = open(listfile, 'w')
        subpaths = patchdir+subpatch
        print listfile,subpaths
        files = os.listdir(subpaths)
        for file in files:
            listw.write(file + '\n')
        listw.close()


if __name__=="__main__":
    patchdir="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/analysis/patch/"
    #get_result_patchlist(patchdir)
    get_result_patchlist_windows(patchdir)