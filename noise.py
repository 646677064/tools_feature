#http://mp.weixin.qq.com/s/3QH9Tuim5yRX_F_yY3mtFQ
import cv2
import numpy as np
from numpy import *

def SaltAndPepperAverage(src,percetage,type=0):
    NoiseImg=src.copy(); 
    print NoiseImg.shape
    #b, g, r = cv2.split(NoiseImg)
    redImg  = src[:,:,2]
    r_average = sum(redImg)/src.shape[0]/src.shape[1]
    greenImg = src[:,:,1]
    g_average = sum(greenImg)/src.shape[0]/src.shape[1]
    blueImg = src[:,:,0]
    b_average = sum(blueImg)/src.shape[0]/src.shape[1]

    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if type==0:
            NoiseImg[randX,randY,0]=b_average
            NoiseImg[randX,randY,1]=g_average
            NoiseImg[randX,randY,2]=r_average
        elif type ==1:
            NoiseImg[randX,randY,0]=r_average
            NoiseImg[randX,randY,1]=g_average
            NoiseImg[randX,randY,2]=b_average
        elif type ==2:
            if random.random_integers(0,1)==0:
                #NoiseImg[randX,randY]=0 
                NoiseImg[randX,randY,0]=b_average
                NoiseImg[randX,randY,1]=g_average
                NoiseImg[randX,randY,2]=r_average
            else:
                #NoiseImg[randX,randY]=255  
                NoiseImg[randX,randY,0]=r_average
                NoiseImg[randX,randY,1]=g_average
                NoiseImg[randX,randY,2]=b_average        
    return NoiseImg 

def SaltAndPepper(src,percetage):
    # print "depth ",src.depth
    # b = cv.CreateImage(cv.GetSize(src), src.depth, 1)
    # g = cv.CloneImage(b)
    # r = cv.CloneImage(b)
    # cv.Split(src, b, g, r, None)
    # merged = cv.CreateImage(cv.GetSize(src), 8, 3)
    # cv.Merge(g, b, r, None, merged)
    NoiseImg=src.copy(); 
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if random.random_integers(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255          
    return NoiseImg 
    
def scaleimg(im):
    #res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    thumb = cv.CreateImage((im.width / 2, im.height / 2), 8, 3)
    cv.Resize(im, thumb)

def moveimg(img):
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))

def rotateimg(src):
    img=src.copy(); 
    rows,cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-30,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def affineimg(src):
    img=src.copy(); 
    rows,cols,ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def colorchange(img):
    # res = cv.CreateImage(cv.GetSize(im), cv.CV_8UC2, 3) #cv.CV_32F, cv.IPL_DEPTH_16S, ...
    # cv.Convert(im, res) cv.Convert()
    # cv.ShowImage("Converted",res)
    res2 = cv.CreateImage(cv.GetSize(im), cv.CV_8UC2, 3)
    cv.CvtColor(im, res2, cv.CV_RGB2BGR) # HLS, HSV, YCrCb, .... #COLOR_BGR2GRAY

def MakeBorder(img1):
    BLUE = [255,0,0]
    img1 = cv2.imread('opencv_logo.png')
    replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
    constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

def split2rgb(img):
    b = cv.CreateImage(cv.GetSize(orig), orig.depth, 1)
    g = cv.CloneImage(b)
    r = cv.CloneImage(b)
    cv.Split(orig, b, g, r, None)

    merged = cv.CreateImage(cv.GetSize(orig), 8, 3)
    cv.Merge(g, b, r, None, merged)

def change_channel_color(src,scale = 0):
    NoiseImg=src.copy(); 
    #b, g, r = cv2.split(NoiseImg)
    redImg  = NoiseImg[:,:,2]
    greenImg = NoiseImg[:,:,1]
    blueImg = NoiseImg[:,:,0]
    r_average = sum(redImg)/NoiseImg.shape[0]/NoiseImg.shape[1]
    g_average = sum(greenImg)/NoiseImg.shape[0]/NoiseImg.shape[1]
    b_average = sum(blueImg)/NoiseImg.shape[0]/NoiseImg.shape[1]
    # print r_average
    # print g_average
    # print b_average
    # redImg = redImg+r_average/5
    # greenImg = greenImg-g_average/5
    # blueImg = blueImg-b_average/5
    # print redImg
    # np.maximum(redImg,1)
    # np.maximum(greenImg,1)
    # np.maximum(blueImg,1)
    # np.minimum(redImg,255)
    # np.minimum(greenImg,255)
    # np.minimum(blueImg,255)
    # merged = cv.CreateImage(cv.GetSize(orig), 8, 3)
    if scale == 0:
        NoiseImg[:,:,2] = (redImg+2*r_average)/3
        NoiseImg[:,:,1] = (greenImg+2*g_average)/3
        NoiseImg[:,:,0] = (blueImg+2*b_average)/3
    elif scale==1:
        NoiseImg[:,:,2] = (redImg+2*r_average+255)/4
        NoiseImg[:,:,1] = (greenImg+2*g_average+255)/4
        NoiseImg[:,:,0] = (blueImg+2*b_average+255)/4
    np.maximum(NoiseImg,1)
    np.minimum(NoiseImg,255)
    return NoiseImg

def gausenoise(src,param=20):
    grayscale=255
    NoiseImg=src.copy(); 
    w=NoiseImg.shape[1]  
    h=NoiseImg.shape[0]
    for x in xrange(0,h):  
        for y in xrange(0,w,2):  
            r1=np.random.random_sample()  
            r2=np.random.random_sample()  
            z1=param*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))  
            z2=param*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
            fxy_val_0=int(img[x,y,0]+z1) 
            fxy_val_1=int(img[x,y,1]+z1)
            fxy_val_2=int(img[x,y,2]+z1)
            fxy1_val_0=int(img[x,y+1,0]+z2)
            fxy1_val_1=int(img[x,y+1,1]+z2)
            fxy1_val_2=int(img[x,y+1,2]+z2)
            NoiseImg[x,y,0]=fxy_val_0
            NoiseImg[x,y,1]=fxy_val_1
            NoiseImg[x,y,2]=fxy_val_2
            NoiseImg[x,y+1,0]=fxy1_val_0 
            NoiseImg[x,y+1,1]=fxy1_val_1
            NoiseImg[x,y+1,2]=fxy1_val_2
    np.maximum(NoiseImg,255)
    np.minimum(NoiseImg,0)
    return NoiseImg

def rotateimg2(src,degree=270):
    img=src.copy(); 
    rows,cols,ch = img.shape
    height = rows
    width = cols
    #degree = 270
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    matRotation[0,2] +=(widthNew-width)/2
    matRotation[1,2] +=(heightNew-height)/2
    dst = cv2.warpAffine(img,matRotation,(widthNew,heightNew))
    return dst

def rotate_save_as(basedir,outdir):
    f_list = os.listdir(basedir)
    #picname = "IMG_1848.JPG"
    i=0
    for file_comp4 in f_list:
        basename =os.path.splitext(file_comp4)[0]
        midlename = basename.split("_")[1]
        img=cv2.imread(basedir+file_comp4)
        rows,cols,ch = img.shape
        i +=1
        print file_comp4,rows,cols,ch
        if len(midlename)==4:
            img_dest = rotateimg2(img,270)
            cv2.imwrite(outdir+basename+".jpg", img_dest)
        else:
            cv2.imwrite(outdir+basename+".jpg", img)
    print "over,i=",i
    print 1

if __name__=='__main__':
    basedir = "/mnt/storage/dataset/CompCars/data/image/102/257/2015/"
    picname = "2e3ea853301bd4.jpg"
    outdir = "/mnt/storage/liushuai/RFCN/R_fcn_bin/Gausenoise/"
    img=cv2.imread(basedir+picname)#,flags=cv2.IMREAD_COLOR
    # gausecopyimg = img.copy()
    # gausecopyimg=cv2.GaussianBlur(gausecopyimg,(7,7),sigmaX=0)
    NoiseImg=change_channel_color(img)
    fileName=outdir+'color_change.jpg'
    cv2.imwrite(fileName,NoiseImg,[cv2.IMWRITE_JPEG_QUALITY,100])

    NoiseImg1=gausenoise(img)
    fileName=outdir+'gausenoise.jpg'
    cv2.imwrite(fileName,NoiseImg1,[cv2.IMWRITE_JPEG_QUALITY,100])

    Pers=[0.01,0.03,0.05,0.1,0.2]
    for i in Pers:
        NoiseImg=SaltAndPepperAverage(img,i)
        fileName=outdir+'SaltPepper_'+str(i)+'.jpg'
        cv2.imwrite(fileName,NoiseImg,[cv2.IMWRITE_JPEG_QUALITY,100])


        aff_img=affineimg(NoiseImg)
        aff_fileName=outdir+'affineimg'+str(i)+'.jpg'
        cv2.imwrite(aff_fileName,aff_img,[cv2.IMWRITE_JPEG_QUALITY,100])

        ratate_img=rotateimg(NoiseImg)
        ratate_fileName=outdir+'ratate_img'+str(i)+'.jpg'
        cv2.imwrite(ratate_fileName,ratate_img,[cv2.IMWRITE_JPEG_QUALITY,100])


    basedir = "C:\\Users\\ysc\\Desktop\\detection\\"
    outdir = "C:\\Users\\ysc\\Desktop\\tmp\\"
    #rotate_save_as(basedir,outdir)