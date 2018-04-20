from xml.etree.ElementTree import ElementTree,Element
from PIL import Image
import os


if __name__=="__main__":
    base_dir="/storage2/liushuai/gs6_env/market1501_extract_freature/dalu2/"
    xmldir = base_dir+"/NG/"
    jpgdir = base_dir+"/NG/"
    patchdir = base_dir+"/NG_patch_dir/"
	
	
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
            print len(nodelist),n,nodename
            if os.path.isdir(dirname):
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                img_crop.save(dirname + filename[0:-4] + "_" + str(n) + ".jpg")
            else:
                os.makedirs(dirname)
                nodexy = nodelist[n].findall("bndbox")
                xmin = int(nodexy[0].find("xmin").text)
                ymin = int(nodexy[0].find("ymin").text)
                xmax = int(nodexy[0].find("xmax").text)
                ymax = int(nodexy[0].find("ymax").text)
                img_crop = img.crop((xmin, ymin, xmax, ymax))
                img_crop.save(dirname + filename[0:-4] + "_" + str(n) + ".jpg")
               
    
    
    
        

