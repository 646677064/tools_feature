# -*- coding: cp936 -*-
from xml.etree.ElementTree import ElementTree,Element
import os,shutil
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')

def stastic_all_class_flist(SKUfile,image_set_file,xmldir,outfile,threholdmin=50,threholdmax=1000):
    shutil.copyfile(image_set_file,image_set_file+"_bak")
    with open(image_set_file) as f:
        image_index = [x.strip().strip('\r\n') for x in f.readlines()]
    ft = open(SKUfile,"r")
    line = ft.readline()
    classnum = {}
    _class_file_list={}
    _class_file_list_det={}
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
        _class_file_list[classlen]=[]
        _class_file_list_det[classlen]=0
    ft.close()

    #xmldir = "/nas/public/liushuai/beer/baiwei_2143/xml_reset/"
    #files = os.listdir(xmldir)
    for filename in image_index:
        #if os.path.splitext(filename)[1] == '.xml':
        if os.path.exists(xmldir+filename+'.xml'):
            #print filename + "\n"
            xmldata = ElementTree()
            xmldata.parse(xmldir+filename+'.xml')
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
                    xmllist[index].append(filename)                    
                    xmlmap[index][filename]=0
                    if filename not in _class_file_list[index+1]:
                        if filename  in _class_file_list[index+1]:
                            print "in here"
                        _class_file_list[index+1].append(filename)
    for x in range(classlen):
        if classnum[xmllist_top[x]]<threholdmin:
            print xmllist_top[x]," num is less than ",threholdmin
        elif classnum[xmllist_top[x]]<threholdmax:
            print "before len: ",classnum[xmllist_top[x]]," ",len(_class_file_list[x+1])
            num_multi=int(threholdmax/classnum[xmllist_top[x]]+1)
            _class_file_list[x+1]=_class_file_list[x+1]*num_multi
            _class_file_list_det[x+1]=1
            print "after len: ",classnum[xmllist_top[x]]*num_multi," ",len(_class_file_list[x+1])
    name_copy_count={}
    for filename in image_index:
        name_copy_count[filename]=1
        for x in range(classlen):
            if _class_file_list_det[x+1]==1:
                for ff in _class_file_list[x+1]:
                    if filename==ff:
                        name_copy_count[filename]=name_copy_count[filename]+1
        if name_copy_count[filename]>1:
            print filename," times : ",name_copy_count[filename]
    newfilelist=[]
    for filename in image_index:
        newfilelist.append(filename)
        if name_copy_count[filename]>1:
            zzz=[]
            zzz.append(filename)
            zzz=zzz*name_copy_count[filename]
            newfilelist=newfilelist+zzz
    with open(image_set_file+".1","w") as newf:    
        for tmp in newfilelist:
            newf.writelines(tmp+"\n")





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
                    xmllist[index].append(filename)                    
                    xmlmap[index][filename]=0


                
    fl = open(outfile,"w")
    index = 0
    for i in range(len(xmllist_top)):
        if  xmllist_top[i]=="binggan":
            fl.write("\n\n\nWARN  WARN  WARN\n"+xmllist_top[i]+"\t"+bytes(classnum[xmllist_top[i]])+"\nWarning!!!this SKU is not exist!! appear files :\n")
            for key in xmlmap[i]:#enumrate(xmlmap[i])
                fl.write(key+"\n")
            fl.write("\n\n\n")

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
    base="/storage2/tiannuodata/work/projdata/nestle4goods/nestle4goodsproj1/"
    base="/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/"
    base="/storage2/tiannuodata/work/projdata/aofei/aofeiproj1/"
    SKUfile=base+"/skufile.txt"
    xmldir=base+"/Annotations/"
    outfile=base+"nestleconfectionery_statistic.txt"
    image_set_file=base+"/ImageSets/Main/trainval.txt"
    threholdmin=50
    threholdmax=2000
    sd=[]
    sd.append("aaaaaa")
    sd.append("bbbbb")
    sd.append("cccc")
    sd.append("dddd")
    sd.append("eee")
    d=2
    sd=sd*d
    print sd
    stastic_all_class_flist(SKUfile,image_set_file,xmldir,outfile,threholdmin,threholdmax)
    #stastic_all_SKU_NUM(SKUfile,xmldir,outfile)
    #static_SKU_pic_bbox(SKUfile,xmldir,outfile)
    #python /home/liushuai/storage/RFCN/R_fcn_bin/sumClasses.py