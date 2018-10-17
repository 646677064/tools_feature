import cv2
import numpy as np
import os
import random
import math


def replace_space_jpgdir(inputdir,outputdir):
    filelist=os.listdir(inputdir)
    for file_one in filelist:
        #if file_one[-4:-1]==".jpg":
        if file_one.find(' (2)')>0:
            endfile=file_one.replace(' (2)','_2')
            os.rename(os.path.join(inputdir,file_one),os.path.join(inputdir,str(endfile)))

# def replace_space_xmldir(inputdir,outputdir):
#     filelist=os.listdir(inputdir)
#     for file_one in filelist:
#         if file_one[-4:-1]==".xml":
#             file_one.replace(' (2)','_2'):

if __name__ == "__main__":
    inputdir="/storage3/tiannuodata/ftpUpload/quecao1017-1163/"
    outputdir="/storage3/tiannuodata/ftpUpload/quecao1017-1163/"
    replace_space_jpgdir(inputdir,outputdir)