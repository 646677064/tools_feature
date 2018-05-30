from xml.etree.ElementTree import ElementTree,Element
from PIL import Image
import os


def only_getpatchlist(patchdir):
    listdir=patchdir+"/patchlist/"
    if not os.path.exists(listdir):
        os.mkdir(listdir)
    subpatchs = os.listdir(patchdir)
    for subpatch in subpatchs:
        listfile = listdir+ subpatch+".txt"
        listw = open(listfile, 'w')
        subpaths = patchdir+subpatch
        print listfile,subpaths
        files = os.listdir(subpaths)
        for file in files:
            listw.write(file + '\n')
        listw.close()

def crops_all_and_getpatchlist(jpgdir,xmldir,patchdir):
    listdir=patchdir+"/patchlist/"
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
            if os.path.isdir(dirname):
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                #print xmin, ymin, xmax, ymax
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
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
    only_getpatchlist(patchdir)

def crops_and_getpatchlist(jpgdir,xmldir,patchdir,only_cutlist=[]):
    listdir=patchdir+"/patchlist/"
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
            if nodename not in only_cutlist:
                continue
            if os.path.isdir(dirname):
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                #print xmin, ymin, xmax, ymax
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
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
    only_getpatchlist(patchdir)           
    # if not os.path.exists(listdir):
    #     os.mkdir(listdir)
    # subpatchs = os.listdir(patchdir)
    # for subpatch in subpatchs:
    #     listfile = listdir+ subpatch+".txt"
    #     listw = open(listfile, 'w')
    #     subpaths = patchdir+subpatch
    #     print listfile,subpaths
    #     files = os.listdir(subpaths)
    #     for file in files:
    #         listw.write(file + '\n')
    #     listw.close()

if __name__=="__main__":

    # xmldir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/Annotations/"
    # jpgdir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/JPEGImages/"
    # patchdir = "/storage2/tiannuodata/work/projdata/baiwei/testdata/patch/"
    xmldir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/Annotations/"
    jpgdir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/JPEGImages/"
    patchdir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj329/analysis/patch/"
    only_cutlist=[]
    # only_cutlist.append("yanjing1")
    # only_cutlist.append("harbin31")
    # only_cutlist.append("snow10")
    #only_cutlist.append("snow129")
    only_cutlist.append("budweiser33")
    only_cutlist.append("budweiser4")
    only_cutlist.append("budweiser18")
    only_cutlist.append("budweiser21")
    only_cutlist.append("budweiser25")
    only_cutlist.append("budweiser23")
    only_cutlist.append("hoegaarden12")
    only_cutlist.append("hoegaarden3")
    only_cutlist.append("harbin4")
    only_cutlist.append("harbin8")
    only_cutlist.append("harbin16")
    only_cutlist.append("harbin15")
    only_cutlist.append("harbin25")
    only_cutlist.append("harbin23")
    only_cutlist.append("becks6")
    only_cutlist.append("sedrin6")
    only_cutlist.append("sedrin8")
    only_cutlist.append("sedrin19")
    only_cutlist.append("sedrin32")
    only_cutlist.append("sedrin18")
    only_cutlist.append("snow1")
    only_cutlist.append("snow13")
    only_cutlist.append("tsingtao4")
    only_cutlist.append("yanjing13")
    #crops_and_getpatchlist(jpgdir,xmldir,patchdir,only_cutlist)
    #crops_all_and_getpatchlist(patchdir)
    patchdir="/data/liushuai/baiweiproj66/66_patches/"
    only_getpatchlist(patchdir)
    #=============================================
    # listdir=patchdir+"/patchlist/"
    # files = os.listdir(xmldir)
    # i = 1
    # for filename in files:
    #     print str(i) + "\t" + filename + "\n"
    #     i = i + 1
    #     if filename[-3:]!='xml' :
    #         continue
    #     xmldata = ElementTree()
    #     xmldata.parse(xmldir+filename)
    #     if os.path.exists(jpgdir + filename[0:-4] + ".jpg"):
    #         img = Image.open(jpgdir + filename[0:-4] + ".jpg")
    #     elif os.path.exists(jpgdir + filename[0:-4] + ".jpeg"):
    #         img = Image.open(jpgdir + filename[0:-4] + ".jpeg")
    #     else:
    #         print jpgdir + filename[0:-4] + ".jpg:"+"  No that image!\n"
    #     nodelist = xmldata.findall("object")
    #     for n in range(len(nodelist)):
    #         nodename = nodelist[n].find("name").text
    #         nodename = nodename.replace("*","-")
    #         dirname = patchdir + nodename + "/"
    #         if nodename not in only_cutlist:
    #             continue
    #         # if os.path.isdir(dirname):
    #         #     nodexy = nodelist[n].findall("bndbox")
    #         #     xmin = int(nodexy[0].find("xmin").text)
    #         #     ymin = int(nodexy[0].find("ymin").text)
    #         #     xmax = int(nodexy[0].find("xmax").text)
    #         #     ymax = int(nodexy[0].find("ymax").text)
    #         #     #print xmin, ymin, xmax, ymax
    #         #     if xmin>=xmax or ymin>=ymax:
    #         #         print xmin, ymin, xmax, ymax,filename
    #         #         break
    #         #     img_crop = img.crop((xmin, ymin, xmax, ymax))
    #         #     if img_crop is None:
    #         #         print "is None",filename
    #         #         break
    #         #     img_crop.save(dirname + filename[0:-4] + "_" + str(n) + ".jpg")
    #         # else:
    #         #     os.makedirs(dirname)
    #         #     nodexy = nodelist[n].findall("bndbox")
    #         #     xmin = int(nodexy[0].find("xmin").text)
    #         #     ymin = int(nodexy[0].find("ymin").text)
    #         #     xmax = int(nodexy[0].find("xmax").text)
    #         #     ymax = int(nodexy[0].find("ymax").text)
    #         #     #print xmin, ymin, xmax, ymax
    #         #     img_crop = img.crop((xmin, ymin, xmax, ymax))
    #         #     if xmin>=xmax or ymin>=ymax:
    #         #         print  xmin, ymin, xmax, ymax,filename
    #         #         break
    #         #     if img_crop is None:
    #         #         print "is None",filename
    #         #         break
    #         #     img_crop.save(dirname + filename[0:-4] + "_" + str(n) + ".jpg")

    #         if os.path.isdir(dirname):
    #             nodexy = nodelist[n].findall("bndbox")
    #             xmin = int(nodexy[0].find("xmin").text)
    #             ymin = int(nodexy[0].find("ymin").text)
    #             xmax = int(nodexy[0].find("xmax").text)
    #             ymax = int(nodexy[0].find("ymax").text)
    #             #print xmin, ymin, xmax, ymax
    #             if xmin>=xmax or ymin>=ymax:
    #                 print  xmin, ymin, xmax, ymax,filename
    #                 break
    #             img_crop = img.crop((xmin, ymin, xmax, ymax))
    #             img_crop.save(dirname + filename[0:-4] + "_" + str(xmin)+"_" + str(ymin)+"_" + str(xmax)+"_" + str(ymax)+ ".jpg")
    #         else:
    #             os.makedirs(dirname)
    #             nodexy = nodelist[n].findall("bndbox")
    #             xmin = int(nodexy[0].find("xmin").text)
    #             ymin = int(nodexy[0].find("ymin").text)
    #             xmax = int(nodexy[0].find("xmax").text)
    #             ymax = int(nodexy[0].find("ymax").text)
    #             if xmin>=xmax or ymin>=ymax:
    #                 print  xmin, ymin, xmax, ymax,filename
    #                 break
    #             #print xmin, ymin, xmax, ymax
    #             img_crop = img.crop((xmin, ymin, xmax, ymax))
    #             img_crop.save(dirname + filename[0:-4] + "_" + str(xmin)+"_" + str(ymin)+"_" + str(xmax)+"_" + str(ymax) + ".jpg")
               
    # if not os.path.exists(listdir):
    #     os.mkdir(listdir)
    # subpatchs = os.listdir(patchdir)
    # for subpatch in subpatchs:
    #     listfile = listdir+ subpatch+".txt"
    #     listw = open(listfile, 'w')
    #     subpaths = patchdir+subpatch
    #     print listfile,subpaths
    #     files = os.listdir(subpaths)
    #     for file in files:
    #         listw.write(file + '\n')
    #     listw.close()
    
        

