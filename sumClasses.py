# -*- coding: cp936 -*-
from xml.etree.ElementTree import ElementTree,Element
import os
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')


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
                    #print node.find("name").text
                    xmllist[index].append(filename)                    
                    #xmlmap[index][filename]=0


                
    fl = open(outfile,"w")
    index = 0
    for i in range(len(xmllist_top)):
        # if  xmllist_top[i]=="binggan":
        #     fl.write("\n\n\nWARN  WARN  WARN\n"+xmllist_top[i]+"\t"+bytes(classnum[xmllist_top[i]])+"\nWarning!!!this SKU is not exist!! appear files :\n")
        #     for key in xmlmap[i]:#enumrate(xmlmap[i])
        #         fl.write(key+"\n")
        #     fl.write("\n\n\n")

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
                

if __name__ == "__main__":
    base="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/"
    #base="/home/liushuai/medical/kele/keleproj1/"
    SKUfile=base+"/skufile.txt"
    xmldir=base+"/Annotations/"
    outfile=base+"baiwei_statistic.txt"
    stastic_all_SKU_NUM(SKUfile,xmldir,outfile)
    #static_SKU_pic_bbox(SKUfile,xmldir,outfile)
    #python /home/liushuai/storage/RFCN/R_fcn_bin/sumClasses.py
