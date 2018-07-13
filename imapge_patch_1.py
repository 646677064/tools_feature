from xml.etree.ElementTree import ElementTree,Element
from PIL import Image
import os

def crop_patches(xmldir,jpgdir,patchdir,only_cutlist):
    if os.path.exists(patchdir)==False:
        os.makedirs(patchdir)
    files = os.listdir(xmldir)
    i = 1
    for filename in files:
        print str(i) + "\t" + filename + "\n"
        i = i + 1
        if filename[-3:]!='xml' :
            continue
        xmldata = ElementTree()
        xmldata.parse(xmldir+filename)
        if os.path.exists(jpgdir + filename[0:-4] + ".jpg"):
            img = Image.open(jpgdir + filename[0:-4] + ".jpg")
        elif os.path.exists(jpgdir + filename[0:-4] + ".jpeg"):
            img = Image.open(jpgdir + filename[0:-4] + ".jpeg")
        else:
            print jpgdir + filename[0:-4] + ".jpg:"+"  No that image!\n"
        nodelist = xmldata.findall("object")
        for n in range(len(nodelist)):
            nodename = nodelist[n].find("name").text
            nodename = nodename.replace("*","-")
            dirname = patchdir + nodename + "/"
            # if nodename not in only_cutlist:
            #     continue
            if os.path.isdir(dirname):
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
                #print xmin, ymin, xmax, ymax
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                img_crop.save(dirname + filename[0:-4] + "_" + str(xmin)+"_" + str(ymin)+"_" + str(xmax)+"_" + str(ymax)+ ".jpg")
            else:
                os.makedirs(dirname)
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
                #print xmin, ymin, xmax, ymax
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                img_crop.save(dirname + filename[0:-4] + "_" + str(xmin)+"_" + str(ymin)+"_" + str(xmax)+"_" + str(ymax) + ".jpg")
               
    
def crop_patches_2(indir,patchdir_out,name,only_cutlist):
    xmldir=indir+"/Annotations/"
    jpgdir=indir+"/JPEGImages/"
    patchdir=patchdir_out+name
    if os.path.exists(patchdir)==False:
        os.makedirs(patchdir)
    files = os.listdir(xmldir)
    i = 1
    for filename in files:
        print str(i) + "\t" + filename + "\n"
        i = i + 1
        if filename[-3:]!='xml' :
            continue
        xmldata = ElementTree()
        xmldata.parse(xmldir+filename)
        if os.path.exists(jpgdir + filename[0:-4] + ".jpg"):
            img = Image.open(jpgdir + filename[0:-4] + ".jpg")
        elif os.path.exists(jpgdir + filename[0:-4] + ".jpeg"):
            img = Image.open(jpgdir + filename[0:-4] + ".jpeg")
        else:
            print jpgdir + filename[0:-4] + ".jpg:"+"  No that image!\n"
        nodelist = xmldata.findall("object")
        for n in range(len(nodelist)):
            nodename = nodelist[n].find("name").text
            nodename = nodename.replace("*","-")
            dirname = patchdir + nodename + "/"
            # if nodename not in only_cutlist:
            #     continue
            if os.path.isdir(dirname):
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
                #print xmin, ymin, xmax, ymax
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                img_crop.save(dirname + filename[0:-4] + "_" + str(xmin)+"_" + str(ymin)+"_" + str(xmax)+"_" + str(ymax)+ ".jpg")
            else:
                os.makedirs(dirname)
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
                #print xmin, ymin, xmax, ymax
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                img_crop.save(dirname + filename[0:-4] + "_" + str(xmin)+"_" + str(ymin)+"_" + str(xmax)+"_" + str(ymax) + ".jpg")
               
    

if __name__=="__main__":
    # xmldir = "C:\\Users\\peng\\Desktop\\nestle4goods\\Annotations\\"
    # jpgdir = "C:\\Users\\peng\\Desktop\\nestle4goods\\JPEGImages\\"
    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    only_cutlist=[]
    indir=basedir+"/aofei/aofeiproj1/"
    name="/aofeiproj1/"
    crop_patches_2(indir,patchdir,only_cutlist)

    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/baiwei/baiweiproj2/"
    name="/baiweiproj2/"
    crop_patches_2(indir,patchdir,only_cutlist)

    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/kele/keleproj1/"
    name="/keleproj1/"
    crop_patches_2(indir,patchdir,only_cutlist)
    
    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/nersen/nersenproj1/"
    name="/nersenproj1/"
    crop_patches_2(indir,patchdir,only_cutlist)
    
    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/nestle4goods/nestle4goodsproj2/"
    name="/nestle4goodsproj2/"
    crop_patches_2(indir,patchdir,only_cutlist)
    
    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/nielsenchips/nielsenchipsproj1/"
    name="/nielsenchipsproj1/"
    crop_patches_2(indir,patchdir,only_cutlist)

    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/niersen799/niersen799proj1/"
    name="/niersen799proj1/"
    crop_patches_2(indir,patchdir,only_cutlist)
    
    basedir="/storage2/tiannuodata/work/projdata/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/shushida/shushidaproj1/"
    name="/shushidaproj1/"
    crop_patches_2(indir,patchdir,only_cutlist)

    basedir="/storage2/liushuai/data/data/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/extra2/extra2proj1/"
    name="/extra2proj1/"
    crop_patches_2(indir,patchdir,only_cutlist)
    
    basedir="/storage2/liushuai/data/data/"
    patchdir = "/data/tiannuodata/patches_all/"
    indir=basedir+"/colgate/colgateproj1/"
    name="/colgateproj1/"
    crop_patches_2(indir,patchdir,only_cutlist)





    
    
        

