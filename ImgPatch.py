from xml.etree.ElementTree import ElementTree,Element
from PIL import Image
import os


if __name__=="__main__":

    xmldir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/Annotations/"
    jpgdir = "/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/JPEGImages/"
    patchdir = "/storage2/tiannuodata/work/projdata/baiwei/testdata/patch/"
    only_cutlist=[]
    # only_cutlist.append("yanjing1")
    # only_cutlist.append("harbin31")
    # only_cutlist.append("snow10")
    #only_cutlist.append("snow129")
    only_cutlist.append("laoshan5")

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
                #print xmin, ymin, xmax, ymax
                if xmin>=xmax or ymin>=ymax:
                    print xmin, ymin, xmax, ymax,filename
                    break
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                if img_crop is None:
                    print "is None",filename
                    break
                img_crop.save(dirname + filename[0:-4] + "_" + str(n) + ".jpg")
            else:
                os.makedirs(dirname)
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                #print xmin, ymin, xmax, ymax
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                if xmin>=xmax or ymin>=ymax:
                    print  xmin, ymin, xmax, ymax,filename
                    break
                if img_crop is None:
                    print "is None",filename
                    break
                img_crop.save(dirname + filename[0:-4] + "_" + str(n) + ".jpg")
               
    
    
    
        

