import os
import string
from lxml import etree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import cv2
import numpy as np






def Createxml(objwidth,objheight,jpgpath):

	root=etree.Element('annotation')
	foder = SubElement(root, 'folder')
	foder.text='newdata'

	filename = SubElement(root, 'filename')
	filename.text='1'

	path = SubElement(root, 'path')
	path.text=jpgpath

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

	


# jpgpath = "/home/pengshengfeng/rfcndemo/nielsencoco/img/"
# txtpath =  "/home/pengshengfeng/rfcndemo/nielsencoco/alltxt/"
# savepath = "/home/pengshengfeng/rfcndemo/nielsencoco/allxml/"
# savepath = "C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\jpegnest\\xml\\"
# if not os.path.exists(savepath):
#     os.mkdir(savepath)


# # txtfiles = os.listdir(txtpath)
# # print len(txtfiles)


# txtfile ="/home/pengshengfeng/rfcndemo/nielsencoco/listjpg.txt"
# txtfile="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\jpegnest\\29.txt"
# file = open(txtfile)
# lines = file.readlines();
# for line in lines:
# 	line = line.strip('\n')	
# 	image_pre,ext = os.path.splitext(line)
# 	jpgfile = jpgpath+line
# 	print jpgfile
# 	img = cv2.imread(jpgfile)
# 	print img.shape
# 	size = img.shape
# 	height = size[0]
# 	width = size[1]
# 	print jpgfile
# 	root = Createxml(width,height)

# 	image_pre2,ext = os.path.splitext(image_pre)

# 	txtfile = txtpath+image_pre2+".txt"
# 	savexml = savepath+image_pre+".xml"

# 	print txtfile
# 	print savexml
# 	file = open(txtfile)
# 	lines = file.readlines();
# 	for line in lines:
# 		line = line.strip('\r\n')	
# 		line = line.strip('\n')
# 		line = line.strip(' ')	
# 		#xmin,ymin,objwidth,objheight,name= line.split(",",4)
# 		name,path,xmin,ymin,objwidth,objheight,angle,sku,skuname=line.split(" ",8)
# 		xmax = int(xmin)+int(objwidth)
# 		ymax = int(ymin)+int(objheight)
# 		skuname=skuname.strip('\"')
# 		addobject(root,skuname,xmin,ymin,str(xmax),str(ymax))
# 	tree=etree.ElementTree(root)
# 	tree.write(savexml,encoding='utf-8',pretty_print=True)


testpath = "C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\package_500\\"

savepath = "C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\package_500\\xml\\"
if not os.path.exists(savepath):
    os.mkdir(savepath)

basedir="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\package_500\\package_500\\"
txtfile=basedir+"22.txt"

# #=====================================================================================================
# testpath = "C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\"

# savepath = "C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\xml\\"
# if not os.path.exists(savepath):
#     os.mkdir(savepath)

# basedir="C:\\Users\\ysc\\Documents\\WXWork\\1688853171285446\\Cache\\File\\2018-03\\jpegnest\\jpegnest\\"
# txtfile=basedir+"29.txt"


f_list = os.listdir(basedir)
file = open(txtfile)
lines = file.readlines();

for file_comp4 in f_list:
#file_comp4='budweiser22234.jpg'
#if True:
    if os.path.splitext(file_comp4)[1]in ['.jpg','.JPG','.jpeg','.JPEG']:
        basename = os.path.splitext(file_comp4)[0]
        bfind = False
        bmakehead = False
        for line in lines:
            line = line.strip('\r\n')	
            line = line.strip('\n')
            line = line.strip(' ')	
            name,path,xmin,ymin,objwidth,objheight,angle,sku,skuname=line.split(" ",8)
            if bfind== True:
                if file_comp4!=name:
                    tree=etree.ElementTree(root)
                    savexml=savepath+os.path.splitext(file_comp4)[0]+".xml"
                    tree.write(savexml,encoding='utf-8',pretty_print=True)
                    #cv2.imwrite(testpath+"test.jpg",img)
                	#cv2.imshow("Canvas", img)
                    break;
            if name==file_comp4:
                bfind = True
                if bmakehead==False:
                    bmakehead = True
                    jpgfile = basedir+name
                    img = cv2.imread(jpgfile)
                    print img.shape
                    size = img.shape
                    height = size[0]
                    width = size[1]
                    root = Createxml(width,height,jpgfile)

                theta = float(angle)*np.pi/180 
                r_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                    [np.sin(theta),np.cos(theta)]]);
                old_points = np.array([[0,int(objwidth),int(objwidth),0],
                                    [0,0,int(objheight),int(objheight)]]);
                x0=int(xmin)
                y0=int(ymin)
                print x0,y0
                new_points=np.dot(r_matrix,old_points)#r_matrix*old_points
                red = (0, 0, 255)
                cv2.line(img, (int(new_points[0][0]+x0), int(new_points[1][0]+y0)), (int(new_points[0][1]+x0), int(new_points[1][1]+y0)), red, 3)
                cv2.line(img, (int(new_points[0][0]+x0), int(new_points[1][0]+y0)), (int(new_points[0][3]+x0), int(new_points[1][3]+y0)), red, 3)
                cv2.line(img, (int(new_points[0][1]+x0), int(new_points[1][1]+y0)), (int(new_points[0][2]+x0), int(new_points[1][2]+y0)), red, 3)
                cv2.line(img, (int(new_points[0][3]+x0), int(new_points[1][3]+y0)), (int(new_points[0][2]+x0), int(new_points[1][2]+y0)), red, 3)
                #cv2.write(img,)
                print new_points
                xmin = np.floor(min(new_points[0,:])+x0);
                ymin = np.floor(min(new_points[1,:])+y0);
                xmax = np.ceil(max(new_points[0,:])+x0);
                ymax = np.ceil(max(new_points[1,:])+y0);
                #print xmin,ymin,xmax,ymax
                xmin=int(xmin)
                ymin=int(ymin)
                xmax=int(xmax)
                ymax=int(ymax)
                xmin=min(max(0,xmin),width)
                ymin=min(max(0,ymin),height)
                xmax=min(max(0,xmax),width)
                ymax=min(max(0,ymax),height)
                print xmin,ymin,xmax,ymax

                #================================================================
                # # # #old_points = [0,0;width,0;width,height;0,height];
                # xmax = int(xmin)+int(objwidth)
                # ymax = int(ymin)+int(objheight)
                #================================================================
                #skuname="label"
                skuname=skuname.strip('\"')
                addobject(root,skuname,str(xmin),str(ymin),str(xmax),str(ymax))
#         bfind = False
#         bmakehead = False
#         for i, pindexname in enumerate(Nametxt):
#             if bfind== True:
#                 if basename!=pindexname:
#                     four_root.write(outDir_Annotations+"/"+basename+".xml", encoding="utf-8",xml_declaration=False)
#                     break;
#             if basename ==pindexname:
#                 bfind = True
# for line in lines:
# 	line = line.strip('\r\n')	
# 	line = line.strip('\n')
# 	line = line.strip(' ')	
# 	name,path,xmin,ymin,objwidth,objheight,angle,sku,skuname=line.split(" ",8)
# 	jpgfile = basedir+name
# 	img = cv2.imread(jpgfile)
# 	print img.shape
# 	size = img.shape
# 	height = size[0]
# 	width = size[1]
# 	#if name!=pindexname:
# 	root = Createxml(width,height,jpgfile)

# 	xmax = int(xmin)+int(objwidth)
# 	ymax = int(ymin)+int(objheight)
# 	print name,path,xmin,ymin,objwidth,objheight,angle,sku,skuname
# 	skuname=skuname.strip('\"')
# 	print name,path,xmin,ymin,objwidth,objheight,angle,sku,skuname
# 	print xmax,ymax
# 	skuname="label"
# 	addobject(root,skuname,xmin,ymin,str(xmax),str(ymax))
# 	tree=etree.ElementTree(root)
# 	print os.path.splitext(name)[1]
# 	print os.path.splitext(name)[0]
# 	if os.path.splitext(name)[1]==".jpg":
# 		savexml=savepath+os.path.splitext(name)[0]+".xml"
# 		tree.write(savexml,encoding='utf-8',pretty_print=True)

















