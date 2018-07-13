import os
import string
import numpy as np
import shutil
import os
from lxml import etree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
#2874 4033

class dobject:
	def __init__(self):
		self.name = ""
		self.xmin = 0
		self.xmax = 0
		self.ymin = 0
		self.ymax = 0
def Createxml(objwidth,objheight):

	root=etree.Element('annotation')
	foder = SubElement(root, 'foder')
	foder.text='newdata'

	filename = SubElement(root, 'filename')
	filename.text='1'

	path = SubElement(root, 'path')
	path.text="jpgpath"

	source = SubElement(root, 'source')
	database = SubElement(source, 'database')
	database.text="Unknown"

	size = SubElement(root, 'size')
	width = SubElement(size, 'width')
	width.text=str(objwidth)
	height = SubElement(size, 'height')
	height.text=str(objheight)
	depth = SubElement(size, 'depth')
	depth.text="3"

	segmented = SubElement(root, 'segmented')
	segmented.text="0"

	return root

	
def addobject(root,objname,objxmin,objymin,objxmax,objymax):

	obj = SubElement(root, 'object')
	name = SubElement(obj, 'name')
	name.text=objname

	pose = SubElement(obj, 'pose')
	pose.text="Unspecified"

	truncated = SubElement(obj, 'truncated')
	truncated.text="0"

	difficult = SubElement(obj, 'difficult')
	difficult.text="0"

	bndbox = SubElement(obj, 'bndbox')
	xmin = SubElement(bndbox, 'xmin')
	xmin.text = objxmin

	ymin = SubElement(bndbox, 'ymin')
	ymin.text = objymin
	
	xmax = SubElement(bndbox, 'xmax')
	xmax.text = objxmax

	ymax = SubElement(bndbox, 'ymax')
	ymax.text = objymax

def IOU(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
	p1x = max(x1,xmin)
	p1y = max(y1,ymin)
	recarea1 = (x2-x1)*(y2-y1)

	p2x = min(x2,xmax)
	p2y = min(y2,ymax)
	recarea2 = (xmax-xmin)*(ymax-ymin)

	orea = 0
	if p2x>p1x and p2y>p1y: 
		overarea =(p2x-p1x)*(p2y-p1y)
		arearate = float(overarea) / float(recarea1 + recarea2 - overarea);
	else:
		arearate = 0
	return arearate




def getpathlist(patchdir):
    listdir = "./listtmp/"
    if os.path.exists(listdir):
    	shutil.rmtree(listdir)
    if not os.path.exists(listdir):
    	os.mkdir(listdir)
    subpatchs = os.listdir(patchdir)
    for subpatch in subpatchs:
    	listfile = listdir+ subpatch+".txt"
    	listw = open(listfile, 'w')
    	subpaths = patchdir+subpatch
    	#print listfile,subpaths
    	files = os.listdir(subpaths)
    	for file in files:
            if file =="Thumbs.db":
                continue
            if "\xe5\x89\xaf\xe6\x9c\xac" in file:
                print "ssssssssssssssssssss"
                continue
    	    listw.write(file + '\n')
    	listw.close()


       
def list2txtfile(txtpath):
	listdir = "/data/liushuai/baiweiproj66/pathlist/"
	if not os.path.exists(txtpath):
		os.mkdir(txtpath)

	skulists = os.listdir(listdir)
	for skufile in skulists:
		skuname,ext= os.path.splitext(skufile)
		if ext!=".txt":
			continue

		patchfile = listdir+skufile
		file = open(patchfile,"r")
		lines = file.readlines()
		for line in lines:
			line=line.strip('\r\n')
			line=line.strip('\n')
			info,ext= os.path.splitext(line)

			print line
			file,xmin,ymin,xmax,ymax= info.split("_")
			print file,xmin,ymin,xmax,ymax
			txtfile = txtpath+file+".txt"
			if os.path.exists(txtfile):
				out_file=open(txtfile,'a')
				out_file.write(skuname+","+xmin+","+ymin+","+xmax+","+ymax + '\n')
				out_file.close
			else:
				out_file=open(txtfile,'w')
				out_file.write(skuname+","+xmin+","+ymin+","+xmax+","+ymax + '\n')
				out_file.close		

	if os.path.exists(listdir):
		shutil.rmtree(listdir)  


def correctxml(txtpath,xmlpath):
	listsxml = os.listdir(xmlpath)
	liststxt = os.listdir(txtpath)
	for i in range(len(liststxt)):
		txtfile = liststxt[i]
		print txtfile
		file,ext= os.path.splitext(txtfile)
		xmlfile = xmlpath+file+".xml"
		txtfile = txtpath+txtfile


		file = open(txtfile,"r")	
		lines = file.readlines()
		objectlist = []
		objectlist2 = []
		for line in lines:
			line=line.strip('\n')
			skuname,xmin,ymin,xmax,ymax= line.split(",")
			temp = dobject()
			temp.name = skuname
			temp.xmin = int(xmin)
			temp.xmax = int(xmax)
			temp.ymin = int(ymin)
			temp.ymax = int(ymax)
			objectlist.append(temp)


		tree=ET.parse(xmlfile)
		root = tree.getroot()
		size = root.find('size')
		width = int(size.find('width').text)
		height = int(size.find('height').text)
		

		for obj in root.findall('object'):        
			name = obj.find('name').text
			xmlbox = obj.find('bndbox')
			xmin = int(xmlbox.find('xmin').text)
			xmax = int(xmlbox.find('xmax').text)
			ymin = int(xmlbox.find('ymin').text)
			ymax = int(xmlbox.find('ymax').text)
			temp = dobject()
			temp.name = name
			temp.xmin = int(xmin)
			temp.xmax = int(xmax)
			temp.ymin = int(ymin)
			temp.ymax = int(ymax)
			objectlist2.append(temp)


		for obj2 in objectlist2:
			for obj1 in objectlist:
				overlap = IOU(obj1.xmin, obj1.ymin, obj1.xmax, obj1.ymax, obj2.xmin, obj2.ymin, obj2.xmax, obj2.ymax)
				if overlap>0.9:
					if obj2.name!=obj1.name:
						print obj2.name,obj1.name
						obj2.name=obj1.name
					break;

		root = Createxml(width,height)

		for obj2 in objectlist2:	
			addobject(root,obj2.name,str(obj2.xmin),str(obj2.ymin),str(obj2.xmax),str(obj2.ymax))
		tree=etree.ElementTree(root)
		tree.write(xmlfile,encoding='utf-8',pretty_print=True)
		
if __name__=="__main__":
	#patchdir = "/home/pengshengfeng/getwrong/patchothers/"
	txtpath = "/data/liushuai/baiweiproj66//pathlist11/"
	xmlpath = "/data/liushuai/baiweiproj66//Annotations/"
	#getpathlist(patchdir)
	#list2txtfile(txtpath)
	correctxml(txtpath,xmlpath)




