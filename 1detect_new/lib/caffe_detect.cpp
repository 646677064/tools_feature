#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "set_python.hpp"
#include "caffe_detect.hpp"
using namespace caffe;
using namespace std;

float GetOverLap(float i_x1, float i_y1,float i_x2,float i_y2,  float j_x1,float j_y1,float j_x2,float j_y2);
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  EV2641_InitCarDetector
 *  Description:  Load the model file and weights file ,set GPUID
 * =====================================================================================
 */
int EV2641_InitCarDetector(const char * model_file, const  char * weights_file, int classnum,const int GPUID , Detector * &handle,const char * lib_dir){
    handle = new Detector(model_file, weights_file,classnum, GPUID);
    string tmplibdir = lib_dir;
    int ret = set_config(GPUID,tmplibdir);//"/storage2/liushuai/RFCN/py-R-FCN-master/lib/fast_rcnn/"
    assert (ret==1);
	return EV2641_ERR_SUCCESS;
}
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  EV2641_ReleaseCarDetector
 *  Description:  Release required resource
 * =====================================================================================
 */
int EV2641_ReleaseCarDetector(){
	 return EV2641_ERR_SUCCESS;
}
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detector
 *  Description:  Detect Car and return detection result
 * =====================================================================================
 */
int EV2641_A_GetCarRect(const EV2641Image * image, int &max_ret_num, EV2641Rect * rect, Detector * &handle){
    vector<cv::Rect>  detection_result;

	IplImage * img = cvCreateImage(cvSize(image->width, image->height), 8, 3);
	memcpy(img->imageData, image->imagedata, img->imageSize);

	cv::Mat inputImg = cv::cvarrToMat(img, true);
	handle->Detect(inputImg, detection_result);

	for (int j = 0; j < detection_result.size(); j++)
	{
		if(j >= max_ret_num)
		{
			max_ret_num=j;
			break;
		}
		rect[j].x = detection_result[j].x ;
		rect[j].y = detection_result[j].y ;
		rect[j].w = detection_result[j].width ;
		rect[j].h = detection_result[j].height ;
	}
	cvReleaseImage(&img);
	inputImg.release();
	return EV2641_ERR_SUCCESS;
}

typedef struct Points
{
  float Xmin;
  float Ymin;
  float Xmax;
  float Ymax;
  Points(float xmin, float ymin, float xmax, float ymax):Xmin(xmin),Ymin(ymin),Xmax(xmax),Ymax(ymax){}
};

bool AreaRate(Points& pointsi, Points & pointsj,const float &area,float threshold = 0.3)
{
  float minx,miny,maxx,maxy,areaover;
  minx = (pointsi.Xmin>=pointsj.Xmin?pointsi.Xmin:pointsj.Xmin);
  miny = (pointsi.Ymin>=pointsj.Ymin?pointsi.Ymin:pointsj.Ymin);
  maxx = (pointsi.Xmax<=pointsj.Xmax?pointsi.Xmax:pointsj.Xmax);
  maxy = (pointsi.Ymax<=pointsj.Ymax?pointsi.Ymax:pointsj.Ymax);
  
  
  if (maxx > minx && maxy > miny)
  {
    areaover = (maxx-minx)*(maxy-miny);
    if ((areaover / area) >= threshold)
    {
      return true;
    }
    else
      return false;
  }
  else
    return false;
  
  
}

void DelDetect_after_cut_detect(vector<Result_detect> &detections)
{
  float areai, areaj;

  for (int i = 0; i < detections.size() - 1; ++i){
    for (int j = i+1; j < detections.size(); ++j){
      Points pointsi(detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2);
      Points pointsj(detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2);
      float threshold = 0.3;
      if(detections[i].iclass !=detections[j].iclass)
      {
        threshold = 0.8;
        //continue;
      }
      
      areai = (pointsi.Xmax - pointsi.Xmin)*(pointsi.Ymax - pointsi.Ymin);
      areaj = (pointsj.Xmax - pointsj.Xmin)*(pointsj.Ymax - pointsj.Ymin);
      if (areai <= areaj)
      {
        if (AreaRate(pointsi, pointsj, areai,threshold)){
          detections.erase(detections.begin() + i);
          i--;
          break;
        }
      }
      else if (AreaRate(pointsi, pointsj, areaj,threshold))
      {
        detections.erase(detections.begin() + j);
        j--;
      }

    }
  }
}



/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detector
 *  Description:  Load the model file and weights file
 * =====================================================================================
 */
//load modelfile and weights
Detector::Detector(const string& model_file, const string& weights_file, int classnum,const int GPUID)
{
	m_iclass_num = classnum;
	m_gpuid = GPUID;
    if(GPUID == -1)
    {
        std::cout<<"cpu mode "<<endl;
    	Caffe::set_mode(Caffe::CPU);
    }
    else
    {
        std::cout<<"gpu mode "<<GPUID<<endl;
	    Caffe::set_mode(Caffe::GPU);
      // if(GPUID!=0)
      // {
      // Caffe::SetDevice(0);
      // }
	    Caffe::SetDevice(GPUID);
    }
	net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
}

void Detector::Cut9ImageDetect(cv::Mat& img, vector<Result_detect> & detection_result_class,float nms_threhold,float conf_threhold)
{
	vector<Result_detect >  tmp_detections;//detections,
  cv::Mat tmp_img;
  for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++){
          img(cv::Rect(i*0.3*img.cols, j*0.3*img.rows, img.cols*0.4, img.rows*0.4)).copyTo(tmp_img);
 
          tmp_detections.clear();
          Detect(tmp_img,tmp_detections,nms_threhold,conf_threhold);
	        for (int k = 0; k < tmp_detections.size(); ++k)
	        {
	            tmp_detections[k].x1= (tmp_detections[k].x1 + i*0.3*img.cols) ;
	            tmp_detections[k].y1 = (tmp_detections[k].y1 + j*0.3*img.rows) ;
	            tmp_detections[k].x2 = (tmp_detections[k].x2 + i*0.3*img.cols) ;
	            tmp_detections[k].y2 = (tmp_detections[k].y2 + j*0.3*img.rows) ;

              detection_result_class.push_back(tmp_detections[k]);
          }
  }
  DelDetect_after_cut_detect(detection_result_class);
}

template <class Type>
Type stringToNum(const string& str)
{
  istringstream iss(str);
  Type num;
  iss >> num;
  return num;    
}
#define IN
#define OUT

void read_cfg(char * list_file,OUT  std::vector<int>& labels_size_vec,
  OUT  std::vector<float>& lenght_size_vec,OUT  std::vector<float>& height_size_vec)
{
    labels_size_vec.clear();
    lenght_size_vec.clear();
    height_size_vec.clear();
    int nIndex=0;
    std::ifstream infile(list_file);
    std::string size_str,tmp_str;
    while (!infile.eof())//(infile >> size_str) 
    {
      getline(infile,size_str);
          std::cout<<size_str<<" sssssss";
      stringstream ss(size_str);
      //ss << size_str;
      nIndex=0;
      while(getline(ss, tmp_str, ' '))
      {
        ++nIndex;
        if (1==nIndex)
        {
          std::cout<<tmp_str<<" ";
          int tmp=stringToNum<int>(tmp_str);
          labels_size_vec.push_back(tmp);
        }
        else if(2==nIndex)
        {
          std::cout<<tmp_str<<" ";
          float tmp=stringToNum<float>(tmp_str);
          lenght_size_vec.push_back(tmp);
        } 
        else if(3==nIndex)
        {
          std::cout<<tmp_str<<" ";
          float tmp=stringToNum<float>(tmp_str);
          height_size_vec.push_back(tmp);
          break;//每行3个的格式
        }
      }
      std::cout<<"over "<<std::endl;
      if (1==nIndex)
      {
          lenght_size_vec.push_back(-1.0);
      }
      if (2==nIndex)
      {
          height_size_vec.push_back(-1.0);
      }
    }
    infile.close();
}

bool check_label_exist_cfg(IN int iclass,IN const std::vector<int>& labels_size_vec,
  IN const std::vector<float>& lenght_size_vec,IN const std::vector<float>& height_size_vec,OUT float& outlenght,OUT float& outheight)
{
    outlenght=-1.0;
    outheight=-1.0;
    bool bfind=false;
    for (int z = 0; z < labels_size_vec.size(); ++z)
    {
        float j_lenght=-1.0;
        float j_height=-1.0;
        if (iclass==labels_size_vec[z])//查看对比物体在配置文件中是否有配置
        {
            outlenght =lenght_size_vec[z];
            outheight =height_size_vec[z];
            bfind=true;
            break;
        }
        // if (-1.0==outlenght)
        // {
        //   bfind=false;
        //   //continue;
        // }
      }
      return bfind;
}

void determine_from_size_1(/*list_file,*/vector<Result_detect> & detection_result_class,bool bcheckconfuse=false)
{

    int confuse_label[]={2,3,4,
                 -1,5,6,
                 -1,8,9,
                 12,13,14,
                 -1,36,37,
                 39,40,41,
                 -1,43,44};
    char * list_file="/home/liushuai/tiannuocaffe/1detect_bak/size.txt";
    std::vector<int> labels_size_vec;
    std::vector<float> lenght_size_vec;
    std::vector<float> height_size_vec;
    read_cfg(list_file,labels_size_vec,lenght_size_vec,height_size_vec);

      int size_confuse=sizeof(confuse_label)/sizeof(confuse_label[0]);
    for (int i = 0; i < detection_result_class.size(); ++i)
    {
        for (int k = 0; k < size_confuse; ++k)
        {
            if (confuse_label[k] == detection_result_class[i].iclass)//如果目标是属于大小标签中的
            {
                std::cout<<"confuse_label orignal class:"<<detection_result_class[i].iclass<<std::endl;
                float i_det_lenght=max(detection_result_class[i].x2-detection_result_class[i].x1,
                  detection_result_class[i].y2-detection_result_class[i].y1);
                float i_lenght=-1.0;
                float i_height=-1.0;
                bool bfind = check_label_exist_cfg(detection_result_class[i].iclass,labels_size_vec,lenght_size_vec,height_size_vec,i_lenght,i_height);
                if (bfind)
                {
                    if (-1.0==i_lenght)
                    {
                      continue;
                    }
                }
                else
                {
                      std::cout<<"the lenght of confuse label "<<detection_result_class[i].iclass<<" not in size.txt"<<std::endl; 
                }
                int i_start = 3*(k/3);
                std::vector<Min_distace_index_> distance_vector;
                for (int j = 0; j < detection_result_class.size(); ++j)//从附近的检测到的物体中的尺寸关系
                {
                  if (j==i)
                  {
                    continue;
                  }
                    //因为易混淆的类别的分类可能是错误的，极大的影响准确性，所以需要把易混淆的类别排除在外
                  bool bj_confuse=false;
                  int j_confuse_index=-1;
                    bool bexcpt=false;
                    for (int m = 0; m < size_confuse; ++m)
                    {
                        if (detection_result_class[j].iclass==confuse_label[m])
                        {
                          bj_confuse=true;
                          j_confuse_index=m;
                        }
                        if (bcheckconfuse)
                        {
                          if (i_start==3*(m/3))
                          {
                              bexcpt=true;
                              break;
                          }
                        }
                        else
                        {
                          if (detection_result_class[j].iclass==confuse_label[m] && detection_result_class[j].bcalibrate==false)//是否让纠正后的obj可以去纠正其他obj
                          {
                              bexcpt=true;
                              break;
                          }
                        }
                    }
                    // if (confuse_label[i_start]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+1]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+2]!=detection_result_class[j].iclass)//排除在同一个混淆类别下的物体 中去比较
                    if(!bexcpt)
                    {//do
                        float j_lenght=-1.0;
                        float j_height=-1.0;
                        bool bfind_j = check_label_exist_cfg(detection_result_class[j].iclass,labels_size_vec,lenght_size_vec,height_size_vec,j_lenght,j_height);
                        if (bfind_j)
                        {
                            if (-1.0==j_lenght)
                            {
                              continue;
                            }
                            if (-1.0==j_height)
                            {
                              continue;
                            }
                            float j_det_lenght=max(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float j_det_height=min(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float right_ratio=j_lenght*j_det_height/j_height/j_det_lenght;

                            if (0.8<right_ratio && right_ratio<1.2)//仅根据摆放的和原比例差不多的进行校正
                            {
                                //calibrate
                                Min_distace_index_ min_distace_index;
                                min_distace_index.lenght = j_lenght;
                                min_distace_index.height = j_height;
                                min_distace_index.index=j;//detection_result_class[j].iclass;
                                min_distace_index.bconfuse=bj_confuse;
                                min_distace_index.confuse_index=j_confuse_index;
                                min_distace_index.distace=sqrt(pow(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x1-detection_result_class[i].x2 , 2)
                                              +pow(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y1-detection_result_class[i].y2,2))/4;

                            //std::cout<<"right_ratio "<<right_ratio<<" distace "<<min_distace_index.distace<<std::endl;
                                //std::cout<<"min_distace_index.distace "<<min_distace_index.distace<<std::endl;//4*6.25*i_det_lenght*i_det_lenght
                                // if (2.5*i_det_lenght>min_distace_index.distace)//当距离大于2.5倍长度的物体 不具有对比意义
                                // {
                                //     distance_vector.push_back(min_distace_index);
                                // }
                                // if ((abs(detection_result_class[j].x1-detection_result_class[i].x1)<2.5*i_det_lenght 
                                //   ||abs(detection_result_class[j].x2-detection_result_class[i].x2)<2.5*i_det_lenght)
                                //   && (abs(detection_result_class[j].y1-detection_result_class[i].y1)<2.5*i_det_lenght
                                //     ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<2.5*i_det_lenght))
                                  //if(min_distace_index.distace<2.0*i_det_lenght)
                                if ((min_distace_index.distace<2.0*i_det_lenght 
                                    && ((abs(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x2-detection_result_class[i].x1)<2.0*i_det_lenght))
                                    || (abs(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y2-detection_result_class[i].y1)<1.6*i_det_lenght)))
                                // if ((min_distace_index.distace<2.0*i_det_lenght 
                                //     && (abs(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x2-detection_result_class[i].x1)<3.0*i_det_lenght))
                                //     && (abs(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y2-detection_result_class[i].y1)<3.0*i_det_lenght))
                                {
                                  distance_vector.push_back(min_distace_index);
                                  //std::cout<<" x dis "<<abs(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x2-detection_result_class[i].x1) <<" y dis "<<abs(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y2-detection_result_class[i].y1)<<std::endl;
                                }
                            }
                        }
                    }
                }
                    std::cout<<"i_det_lenght "<<i_det_lenght<<" distance_vector size "<<distance_vector.size()<<std::endl;
                if (distance_vector.size()>1)
                {
                    std::sort(distance_vector.begin(), distance_vector.end(), comparedistance);//根据距离大小 从小到大排序
                    for (int j = distance_vector.size()-1; j>=0;j--)
                    {
                        float j_det_lenght=max(detection_result_class[distance_vector[j].index].x2-detection_result_class[distance_vector[j].index].x1,
                          detection_result_class[distance_vector[j].index].y2-detection_result_class[distance_vector[j].index].y1);
                        float f_min=1000;
                        int min_index=-1;
                        float ratio_three[3]={1.0,1.0,1.0};
                        int ratio_index=-1;
                        bool bfind_very_close=false;
                        for (int ichoose = 0; ichoose < 3; ++ichoose)
                        {
                            float choose_lenght=-1.0;
                            float choose_height=-1.0;
                            if (confuse_label[i_start+ichoose]==detection_result_class[i].iclass)
                            {
                                ratio_index=ichoose;
                            }
                            bool bfind_choose = check_label_exist_cfg(confuse_label[i_start+ichoose],labels_size_vec,lenght_size_vec,height_size_vec,choose_lenght,choose_height);
                            if (bfind_choose)
                            {   
                                float tmp_ratio=j_det_lenght*choose_lenght/distance_vector[j].lenght/i_det_lenght;
                                tmp_ratio = abs(1.0-tmp_ratio);
                                ratio_three[ichoose]=tmp_ratio;
                                if (f_min > tmp_ratio)
                                {
                                  f_min = tmp_ratio;
                                  min_index=ichoose;
                                }
                                if (choose_lenght==distance_vector[j].lenght && abs(j_det_lenght-i_det_lenght)/i_det_lenght<0.05)
                                {
                                  f_min = 0.01;
                                  min_index=ichoose;
                                  bfind_very_close=true;
                                  std::cout<<"bfind_very_close distace "<<distance_vector[j].distace<<std::endl;
                                //std::cout<<"bfind_very_close "<<detection_result_class[distance_vector[j].index].iclass<<" j_det_lenght "<<j_det_lenght<<std::endl;
                                
                                  break;
                                }
                                std::cout<<"judge "<<tmp_ratio<<" j_det_lenght "<<j_det_lenght<<std::endl;
                                // if (tmp_ratio<0.11)
                                // {
                                //   std::cout<<"change label  "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+ichoose]<<std::endl;
                                //   detection_result_class[i].iclass=confuse_label[i_start+ichoose];
                                // }
                            }
                        }
                        if (min_index!=-1)
                        {
                            bool bfalsemin=false;
                            float substract=1000.0;
                            if (-1!=ratio_index)
                            {
                                substract=abs(ratio_three[ratio_index]-f_min);
                                if (substract<0.1)
                                {//自己与最小相减为0，或者与最小的差距不明显
                                    bfalsemin=true;
                                }
                            }
                            if (bfind_very_close)
                            {
                              bfalsemin=false;
                            }
                            if (!bfalsemin && f_min<0.1)//差距较大，就进行更正label
                            {
                                std::cout<<"compare info "<<detection_result_class[distance_vector[j].index].iclass<<" j_det_lenght "<<j_det_lenght<<" distance "<<distance_vector[j].distace<<std::endl;
                                
                                  std::cout<<" x dis "<<abs(detection_result_class[distance_vector[j].index].x1+detection_result_class[distance_vector[j].index].x2-detection_result_class[i].x2-detection_result_class[i].x1)/2 \
                                  <<" y dis "<<abs(detection_result_class[distance_vector[j].index].y1+detection_result_class[distance_vector[j].index].y2-detection_result_class[i].y2-detection_result_class[i].y1)/1.6<<std::endl;
                                std::cout<<"size change label "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+min_index]<<" "<<f_min<<"compare from:"<<distance_vector[j].index<<std::endl;
                                detection_result_class[i].iclass=confuse_label[i_start+min_index];
                                detection_result_class[i].bcalibrate=true;//是否让纠正后的obj可以去纠正其他obj
                                if (bfind_very_close)
                                {
                                    // if (distance_vector[j].bconfuse)
                                    // {
                                    //     detection_result_class[i].iclass=confuse_label[i_start+distance_vector[j].confuse_index%3];
                                    // }
                                    // else
                                    {
                                      if (distance_vector[j].lenght==16.5)
                                      {
                                        detection_result_class[i].iclass=confuse_label[i_start+0];
                                      }
                                      else if (distance_vector[j].lenght==18)
                                      {
                                        detection_result_class[i].iclass=confuse_label[i_start+1];
                                      }
                                      else if (distance_vector[j].lenght==22)
                                      {
                                        detection_result_class[i].iclass=confuse_label[i_start+2];
                                      }
                                    }
                                    std::cout<<"find close  "<<std::endl;
                                    break;//如果有最接近的，就取最接近的进行处理了
                                }
                            }
                        }
                        std::cout<<" "<<std::endl;
                    }
                }
            }
        }
    }
}

void determine_from_size_0(/*list_file,*/vector<Result_detect> & detection_result_class,bool bcheckconfuse=false)
{

    int confuse_label[]={2,3,4,
                 -1,5,6,
                 -1,8,9,
                 12,13,14,
                 -1,36,37,
                 39,40,41,
                 -1,43,44};
    char * list_file="/home/liushuai/tiannuocaffe/1detect_bak/size.txt";
    std::vector<int> labels_size_vec;
    std::vector<float> lenght_size_vec;
    std::vector<float> height_size_vec;
    read_cfg(list_file,labels_size_vec,lenght_size_vec,height_size_vec);

      int size_confuse=sizeof(confuse_label)/sizeof(confuse_label[0]);
    for (int i = 0; i < detection_result_class.size(); ++i)
    {
        for (int k = 0; k < size_confuse; ++k)
        {
            if (confuse_label[k] == detection_result_class[i].iclass)//如果目标是属于大小标签中的
            {
                std::cout<<"confuse_label orignal class:"<<detection_result_class[i].iclass<<std::endl;
                float i_det_lenght=max(detection_result_class[i].x2-detection_result_class[i].x1,
                  detection_result_class[i].y2-detection_result_class[i].y1);
                float i_lenght=-1.0;
                float i_height=-1.0;
                bool bfind = check_label_exist_cfg(detection_result_class[i].iclass,labels_size_vec,lenght_size_vec,height_size_vec,i_lenght,i_height);
                if (bfind)
                {
                    if (-1.0==i_lenght)
                    {
                      continue;
                    }
                }
                else
                {
                      std::cout<<"the lenght of confuse label "<<detection_result_class[i].iclass<<" not in size.txt"<<std::endl; 
                }
                int i_start = 3*(k/3);
                std::vector<Min_distace_index_> distance_vector;
                for (int j = 0; j < detection_result_class.size(); ++j)//从附近的检测到的物体中的尺寸关系
                {
                    //因为易混淆的类别的分类可能是错误的，极大的影响准确性，所以需要把易混淆的类别排除在外
                    bool bexcpt=false;
                    for (int m = 0; m < size_confuse; ++m)
                    {
                        if (bcheckconfuse)
                        {
                          if (i_start==3*(m/3))
                          {
                              bexcpt=true;
                              break;
                          }
                        }
                        else
                        {
                          if (detection_result_class[j].iclass==confuse_label[m] && detection_result_class[j].bcalibrate==false)//是否让纠正后的obj可以去纠正其他obj
                          {
                              bexcpt=true;
                              break;
                          }
                        }
                    }
                    // if (confuse_label[i_start]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+1]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+2]!=detection_result_class[j].iclass)//排除在同一个混淆类别下的物体 中去比较
                    if(!bexcpt)
                    {//do
                        float j_lenght=-1.0;
                        float j_height=-1.0;
                        bool bfind_j = check_label_exist_cfg(detection_result_class[j].iclass,labels_size_vec,lenght_size_vec,height_size_vec,j_lenght,j_height);
                        if (bfind_j)
                        {
                            if (-1.0==j_lenght)
                            {
                              continue;
                            }
                            if (-1.0==j_height)
                            {
                              continue;
                            }
                            float j_det_lenght=max(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float j_det_height=min(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float right_ratio=j_lenght*j_det_height/j_height/j_det_lenght;
                            if (0.8<right_ratio && right_ratio<1.2)//仅根据摆放的和原比例差不多的进行校正
                            {
                                //calibrate
                                Min_distace_index_ min_distace_index;
                                min_distace_index.lenght = j_lenght;
                                min_distace_index.height = j_height;
                                min_distace_index.index=j;//detection_result_class[j].iclass;
                                min_distace_index.distace=sqrt(pow(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x1-detection_result_class[i].x2 , 2)
                                              +pow(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y1-detection_result_class[i].y2,2))/4;

                                //std::cout<<"min_distace_index.distace "<<min_distace_index.distace<<std::endl;//4*6.25*i_det_lenght*i_det_lenght
                                // if (2.5*i_det_lenght>min_distace_index.distace)//当距离大于2.5倍长度的物体 不具有对比意义
                                // {
                                //     distance_vector.push_back(min_distace_index);
                                // }
                                if ((abs(detection_result_class[j].x1-detection_result_class[i].x1)<2.5*i_det_lenght 
                                  ||abs(detection_result_class[j].x2-detection_result_class[i].x2)<2.5*i_det_lenght)
                                  && (abs(detection_result_class[j].y1-detection_result_class[i].y1)<2.5*i_det_lenght
                                    ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<2.5*i_det_lenght))
                                {
                                  distance_vector.push_back(min_distace_index);
                                }
                            }
                        }
                    }
                }
                    std::cout<<"i_det_lenght "<<i_det_lenght<<" distance_vector size "<<distance_vector.size()<<std::endl;
                if (distance_vector.size()>1)
                {
                    std::sort(distance_vector.begin(), distance_vector.end(), comparedistance);//根据距离大小 从小到大排序
                    for (int j = distance_vector.size(); j < distance_vector.size();j--)
                    {
                        float j_det_lenght=max(detection_result_class[distance_vector[j].index].x2-detection_result_class[distance_vector[j].index].x1,
                          detection_result_class[distance_vector[j].index].y2-detection_result_class[distance_vector[j].index].y1);
                        float f_min=1000;
                        int min_index=-1;
                        float ratio_three[3]={1.0,1.0,1.0};
                        int ratio_index=-1;
                        for (int ichoose = 0; ichoose < 3; ++ichoose)
                        {
                            float choose_lenght=-1.0;
                            float choose_height=-1.0;
                            if (confuse_label[i_start+ichoose]==detection_result_class[i].iclass)
                            {
                                ratio_index=ichoose;
                            }
                            bool bfind_choose = check_label_exist_cfg(confuse_label[i_start+ichoose],labels_size_vec,lenght_size_vec,height_size_vec,choose_lenght,choose_height);
                            if (bfind_choose)
                            {   
                                float tmp_ratio=j_det_lenght*choose_lenght/distance_vector[j].lenght/i_det_lenght;
                                tmp_ratio = abs(1.0-tmp_ratio);
                                ratio_three[ichoose]=tmp_ratio;
                                if (f_min > tmp_ratio)
                                {
                                  f_min = tmp_ratio;
                                  min_index=ichoose;
                                }
                                std::cout<<"judge "<<tmp_ratio<<std::endl;
                                // if (tmp_ratio<0.11)
                                // {
                                //   std::cout<<"change label  "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+ichoose]<<std::endl;
                                //   detection_result_class[i].iclass=confuse_label[i_start+ichoose];
                                // }
                            }
                        }
                        if (min_index!=-1)
                        {
                            bool bfalsemin=false;
                            if (-1!=ratio_index)
                            {
                                float substract=abs(ratio_three[ratio_index]-f_min);
                                if (substract<0.1)
                                {//自己与最小相减为0，或者与最小的差距不明显
                                    bfalsemin=true;
                                }
                            }
                            if (!bfalsemin && f_min<0.1)//差距较大，就进行更正label
                            {
                                std::cout<<"change label "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+min_index]<<" "<<f_min<<"compare from:"<<distance_vector[j].index<<std::endl;
                                detection_result_class[i].iclass=confuse_label[i_start+min_index];
                                detection_result_class[i].bcalibrate=true;//是否让纠正后的obj可以去纠正其他obj
                            }
                        }
                        std::cout<<" "<<std::endl;
                        if (abs(j_det_lenght-i_det_lenght)/i_det_lenght<0.05)
                        {
                            break;//如果有最接近的，就取最接近的进行处理了
                        }
                    }
                }
            }
        }
    }
}

void determine_from_size(/*list_file,*/vector<Result_detect> & detection_result_class ,bool bcheckconfuse=false,const float range_thredhold=4.5)
{

    int confuse_label[]={2,3,4,
                 -1,5,6,
                 -1,8,9,
                 12,13,14,
                 -1,36,37,
                 39,40,41,
                 -1,43,44};
    char * list_file="/home/liushuai/tiannuocaffe/1detect_bak/size.txt";
    std::vector<int> labels_size_vec;
    std::vector<float> lenght_size_vec;
    std::vector<float> height_size_vec;
    read_cfg(list_file,labels_size_vec,lenght_size_vec,height_size_vec);

    int size_confuse=sizeof(confuse_label)/sizeof(confuse_label[0]);
    for (int i = 0; i < detection_result_class.size(); ++i)
    {
        if (detection_result_class[i].bcalibrate)
        {
            continue;
        }
        for (int k = 0; k < size_confuse; ++k)
        {
            if (confuse_label[k] == detection_result_class[i].iclass)//如果目标是属于大小标签中的
            {
                float i_det_lenght=max(detection_result_class[i].x2-detection_result_class[i].x1,
                  detection_result_class[i].y2-detection_result_class[i].y1);
                float i_lenght=-1.0;
                float i_height=-1.0;
                bool bfind = check_label_exist_cfg(detection_result_class[i].iclass,labels_size_vec,lenght_size_vec,height_size_vec,i_lenght,i_height);
                if (bfind)
                {
                    if (-1.0==i_lenght)
                    {
                      continue;
                    }
                }
                else
                {
                      std::cout<<"the lenght of confuse label "<<detection_result_class[i].iclass<<" not in size.txt"<<std::endl; 
                }
                int i_start = 3*(k/3);
                float x_ilenght=2.0;
                std::vector<Min_distace_index_> distance_vector;
                bool bcheck_gate=bcheckconfuse;
distance_vector_size:
                for (int j = 0; j < detection_result_class.size(); ++j)//从附近的检测到的物体中的尺寸关系
                {
                    if (j==i)
                    {
                      continue;
                    }
                      //因为易混淆的类别的分类可能是错误的，极大的影响准确性，所以需要把易混淆的类别排除在外
                    bool bj_confuse=false;
                    int j_confuse_index=-1;
                    bool bexcpt=false;
                    for (int m = 0; m < size_confuse; ++m)
                    {
                        if (detection_result_class[j].iclass==confuse_label[m])
                        {
                          bj_confuse=true;
                          j_confuse_index=m;
                        }
                        if (bcheck_gate)
                        {
                          // if (detection_result_class[j].iclass==confuse_label[m] && i_start==3*(m/3))
                          // {
                          //     bexcpt=true;
                          //     break;
                          // }
                          if (detection_result_class[j].iclass==confuse_label[m] && detection_result_class[j].bcalibrate==false)//是否让纠正后的obj可以去纠正其他obj
                          {
                              bexcpt=true;
                              break;
                          }
                        }
                        else
                        {
                          if (detection_result_class[j].iclass==confuse_label[m] /*&& detection_result_class[j].bcalibrate==false*/)//是否让纠正后的obj可以去纠正其他obj
                          {
                              bexcpt=true;
                              break;
                          }
                        }
                    }
                    // if (confuse_label[i_start]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+1]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+2]!=detection_result_class[j].iclass)//排除在同一个混淆类别下的物体 中去比较
                    if(!bexcpt)
                    {//do
                        float j_lenght=-1.0;
                        float j_height=-1.0;
                        bool bfind_j = check_label_exist_cfg(detection_result_class[j].iclass,labels_size_vec,lenght_size_vec,height_size_vec,j_lenght,j_height);
                        if (bfind_j)
                        {
                            if (-1.0==j_lenght)
                            {
                              continue;
                            }
                            if (-1.0==j_height)
                            {
                              continue;
                            }
                            float j_det_lenght=max(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float j_det_height=min(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float right_ratio=j_lenght*j_det_height/j_height/j_det_lenght;

                            if (0.8<right_ratio && right_ratio<1.2)//仅根据摆放的和原比例差不多的进行校正
                            {
                                //calibrate
                                Min_distace_index_ min_distace_index;
                                min_distace_index.lenght = j_lenght;
                                min_distace_index.height = j_height;
                                min_distace_index.index=j;//detection_result_class[j].iclass;
                                min_distace_index.bconfuse=bj_confuse;
                                min_distace_index.confuse_index=j_confuse_index;
                                min_distace_index.distace=sqrt(pow(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x1-detection_result_class[i].x2 , 2)
                                              +pow(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y1-detection_result_class[i].y2,2))/2;

                            std::cout<<"right_ratio "<<right_ratio<<" distace "<<min_distace_index.distace<<std::endl;
                                //std::cout<<"min_distace_index.distace "<<min_distace_index.distace<<std::endl;//4*6.25*i_det_lenght*i_det_lenght
                                // if (2.5*i_det_lenght>min_distace_index.distace)//当距离大于2.5倍长度的物体 不具有对比意义
                                // {
                                //     distance_vector.push_back(min_distace_index);
                                // }
                                // if ((abs(detection_result_class[j].x1-detection_result_class[i].x1)<2.5*i_det_lenght 
                                //   ||abs(detection_result_class[j].x2-detection_result_class[i].x2)<2.5*i_det_lenght)
                                //   && (abs(detection_result_class[j].y1-detection_result_class[i].y1)<2.5*i_det_lenght
                                //     ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<2.5*i_det_lenght))
                                  //if(min_distace_index.distace<2.0*i_det_lenght)
                                if ((min_distace_index.distace<x_ilenght*i_det_lenght )
                                    && ((abs(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x2-detection_result_class[i].x1)<2.0*i_det_lenght)
                                    || (abs(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y2-detection_result_class[i].y1)<2.0*i_det_lenght)))
                                // if ((min_distace_index.distace<2.0*i_det_lenght 
                                //     && (abs(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x2-detection_result_class[i].x1)<3.0*i_det_lenght))
                                //     && (abs(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y2-detection_result_class[i].y1)<3.0*i_det_lenght))
                                {
                                  std::cout<<" dis "<<min_distace_index.distace<<std::endl;
                                  distance_vector.push_back(min_distace_index);
                                  //std::cout<<" x dis "<<abs(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x2-detection_result_class[i].x1) <<" y dis "<<abs(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y2-detection_result_class[i].y1)<<std::endl;
                                }
                            }
                        }
                    }
                }
                if (distance_vector.size()==0 && x_ilenght<range_thredhold)//range_thredhold=3.5
                {
                  x_ilenght=x_ilenght+0.5;
                  bcheck_gate=bcheckconfuse;
                  goto distance_vector_size;
                }
                std::cout<<"confuse_label orignal class:"<<detection_result_class[i].iclass<<std::endl;
                    std::cout<<"i_det_lenght "<<i_det_lenght<<" distance_vector size "<<distance_vector.size()<<std::endl;
                if (distance_vector.size()>0)
                {
                    std::sort(distance_vector.begin(), distance_vector.end(), comparedistance);//根据距离大小 从小到大排序
                    for (int j = distance_vector.size()-1; j>=0;j--)
                    {
                        float j_det_lenght=max(detection_result_class[distance_vector[j].index].x2-detection_result_class[distance_vector[j].index].x1,
                          detection_result_class[distance_vector[j].index].y2-detection_result_class[distance_vector[j].index].y1);
                        float f_min=1000;
                        int min_index=-1;
                        float ratio_three[3]={1.0,1.0,1.0};
                        int ratio_index=-1;
                        bool bfind_very_close=false;
                        for (int ichoose = 0; ichoose < 3; ++ichoose)
                        {
                            float choose_lenght=-1.0;
                            float choose_height=-1.0;
                            if (confuse_label[i_start+ichoose]==detection_result_class[i].iclass)
                            {
                                ratio_index=ichoose;
                            }
                            bool bfind_choose = check_label_exist_cfg(confuse_label[i_start+ichoose],labels_size_vec,lenght_size_vec,height_size_vec,choose_lenght,choose_height);
                            if (bfind_choose)
                            {   
                                float tmp_ratio=j_det_lenght*choose_lenght/distance_vector[j].lenght/i_det_lenght;
                                tmp_ratio = abs(1.0-tmp_ratio);
                                ratio_three[ichoose]=tmp_ratio;
                                if (f_min > tmp_ratio)
                                {
                                  f_min = tmp_ratio;
                                  min_index=ichoose;
                                }
                                if (choose_lenght==distance_vector[j].lenght && abs(j_det_lenght-i_det_lenght)/i_det_lenght<0.05)
                                {
                                  f_min = 0.01;
                                  min_index=ichoose;
                                  bfind_very_close=true;
                                  std::cout<<"bfind_very_close distace "<<distance_vector[j].distace<<std::endl;
                                //std::cout<<"bfind_very_close "<<detection_result_class[distance_vector[j].index].iclass<<" j_det_lenght "<<j_det_lenght<<std::endl;
                                
                                  break;
                                }
                                std::cout<<"judge "<<tmp_ratio<<" j_det_lenght "<<" distace "<<distance_vector[j].distace<<j_det_lenght<<std::endl;
                                // if (tmp_ratio<0.11)
                                // {
                                //   std::cout<<"change label  "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+ichoose]<<std::endl;
                                //   detection_result_class[i].iclass=confuse_label[i_start+ichoose];
                                // }
                            }
                        }
                        if (min_index!=-1)
                        {
                            bool bfalsemin=false;
                            float substract=1000.0;
                            if (-1!=ratio_index)
                            {
                                substract=abs(ratio_three[ratio_index]-f_min);
                                if (substract<0.1)
                                {//自己与最小相减为0，或者与最小的差距不明显
                                    bfalsemin=true;
                                }
                            }
                            if (bfind_very_close)
                            {
                              bfalsemin=false;
                            }
                            if (!bfalsemin && f_min<0.1)//差距较大，就进行更正label
                            {
                                std::cout<<"compare info "<<detection_result_class[distance_vector[j].index].iclass<<" j_det_lenght "<<j_det_lenght<<" distance "<<distance_vector[j].distace<<std::endl;
                                
                                  std::cout<<" x dis "<<abs(detection_result_class[distance_vector[j].index].x1+detection_result_class[distance_vector[j].index].x2-detection_result_class[i].x2-detection_result_class[i].x1)/2 \
                                  <<" y dis "<<abs(detection_result_class[distance_vector[j].index].y1+detection_result_class[distance_vector[j].index].y2-detection_result_class[i].y2-detection_result_class[i].y1)/2<<std::endl;
                                std::cout<<"size change label "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+min_index]<<" "<<f_min<<"compare from:"<<distance_vector[j].index<<std::endl;
                                detection_result_class[i].iclass=confuse_label[i_start+min_index];
                                detection_result_class[i].bcalibrate=true;//是否让纠正后的obj可以去纠正其他obj
                                if (bfind_very_close)
                                {
                                    // if (distance_vector[j].bconfuse)
                                    // {
                                    //     detection_result_class[i].iclass=confuse_label[i_start+distance_vector[j].confuse_index%3];
                                    // }
                                    // else
                                    {
                                      if (distance_vector[j].lenght==16.5)
                                      {
                                        detection_result_class[i].iclass=confuse_label[i_start+0];
                                      }
                                      else if (distance_vector[j].lenght==18)
                                      {
                                        detection_result_class[i].iclass=confuse_label[i_start+1];
                                      }
                                      else if (distance_vector[j].lenght==22)
                                      {
                                        detection_result_class[i].iclass=confuse_label[i_start+2];
                                      }
                                    }
                                    std::cout<<"find close  "<<std::endl;
                                    break;//如果有最接近的，就取最接近的进行处理了
                                }
                            }
                        }
                        std::cout<<" "<<std::endl;
                    }
                }
            }
        }
    }
}

void confuse_vote_from_size(/*list_file,*/vector<Result_detect> & detection_result_class)
{

    int confuse_label[]={2,3,4,
                 -1,5,6,
                 -1,8,9,
                 12,13,14,
                 -1,36,37,
                 39,40,41,
                 -1,43,44};
    char * list_file="/home/liushuai/tiannuocaffe/1detect_bak/size.txt";
    std::vector<int> labels_size_vec;
    std::vector<float> lenght_size_vec;
    std::vector<float> height_size_vec;
    read_cfg(list_file,labels_size_vec,lenght_size_vec,height_size_vec);

    for (int i = 0; i < detection_result_class.size(); ++i)
    {
        // if (detection_result_class[i].bcalibrate==true)//现在校正的是没有经过校正的，比如附近没有可以进行比较的，我们采用投票
        // {
        //     continue;
        // }
      int size_confuse=sizeof(confuse_label)/sizeof(confuse_label[0]);
        for (int k = 0; k < size_confuse; ++k)
        {
            if (confuse_label[k] == detection_result_class[i].iclass)//如果目标是属于大小标签中的
            {
                std::cout<<"confuse_vote_label orignal class:"<<detection_result_class[i].iclass<<std::endl;
                float i_det_lenght=max(detection_result_class[i].x2-detection_result_class[i].x1,
                  detection_result_class[i].y2-detection_result_class[i].y1);
                float i_lenght=-1.0;
                float i_height=-1.0;
                bool bfind = check_label_exist_cfg(detection_result_class[i].iclass,labels_size_vec,lenght_size_vec,height_size_vec,i_lenght,i_height);
                if (bfind)
                {
                    if (-1.0==i_lenght)
                    {
                      continue;
                    }
                }
                else
                {
                      std::cout<<"the lenght of confuse label "<<detection_result_class[i].iclass<<" not in size.txt"<<std::endl; 
                }
                int i_start = 3*(k/3);
                std::cout<<" k "<<k<<" istart "<<i_start<<std::endl;
                std::vector<Min_distace_index_> distance_vector;
                for (int j = 0; j < detection_result_class.size(); ++j)//从附近的检测到的物体中的尺寸关系
                {
                    //因为易混淆的类别的分类可能是错误的，极大的影响准确性，所以需要把易混淆的类别排除在外
                    bool bconfuse_vote=false;
                    bool bj_confuse=false;
                    int j_confuse_index=-1;
                    for (int m = 0; m < size_confuse; ++m)
                    {
                        // if (detection_result_class[j].iclass==confuse_label[m])
                        // {
                        //   bj_confuse=true;
                        //   j_confuse_index=m;
                        //   break;
                        // }
                        if (detection_result_class[j].iclass==confuse_label[m] && i_start==3*(m/3))//是否让纠正后的obj可以去纠正其他obj
                        {
                            bconfuse_vote=true;
                            break;
                        }
                    }
                    bconfuse_vote=true;
                    // if (confuse_label[i_start]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+1]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+2]!=detection_result_class[j].iclass)//排除在同一个混淆类别下的物体 中去比较
                    if(bconfuse_vote)
                    {//do
                        float j_lenght=-1.0;
                        float j_height=-1.0;
                        bool bfind_j = check_label_exist_cfg(detection_result_class[j].iclass,labels_size_vec,lenght_size_vec,height_size_vec,j_lenght,j_height);
                        if (bfind_j)
                        {
                            if (-1.0==j_lenght)
                            {
                              continue;
                            }
                            if (-1.0==j_height)
                            {
                              continue;
                            }
                            float j_det_lenght=max(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float j_det_height=min(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float right_ratio=j_det_lenght/i_det_lenght;
                            //j_det_lenght*choose_lenght/distance_vector[j].lenght/i_det_lenght
                            if (0.93<right_ratio && right_ratio<1.08)//仅根据大小特别接近的进行校正
                            {
                                //calibrate
                                Min_distace_index_ min_distace_index;
                                min_distace_index.lenght = j_lenght;
                                min_distace_index.height = j_height;
                                min_distace_index.index=j;//detection_result_class[j].iclass;
                                min_distace_index.bconfuse=bj_confuse;
                                min_distace_index.confuse_index=j_confuse_index;
                                min_distace_index.distace=sqrt(pow(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x1-detection_result_class[i].x2 , 2)
                                              +pow(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y1-detection_result_class[i].y2,2))/2;

                                //std::cout<<"min_distace_index.distace "<<min_distace_index.distace<<std::endl;//4*6.25*i_det_lenght*i_det_lenght
                                // if (2.5*i_det_lenght>min_distace_index.distace)//当距离大于2.5倍长度的物体 不具有对比意义
                                // {
                                //     distance_vector.push_back(min_distace_index);
                                // }
                               // if ((abs(detection_result_class[j].x1-detection_result_class[i].x1)<1.5*i_det_lenght 
                               //    ||abs(detection_result_class[j].x2-detection_result_class[i].x2)<1.5*i_det_lenght)
                               //    && (abs(detection_result_class[j].y1-detection_result_class[i].y1)<1.5*i_det_lenght
                               //      ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<1.5*i_det_lenght))
                                if(min_distace_index.distace<1.5*i_det_lenght)
                                {
                                  distance_vector.push_back(min_distace_index);
                                } 
                                // if ( (abs(detection_result_class[j].y1-detection_result_class[i].y1)<1.5*i_det_lenght
                                //     ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<1.5*i_det_lenght))//竖直方向
                                // {
                                //   distance_vector.push_back(min_distace_index);
                                // }
                            }
                        }
                    }
                }
                    std::cout<<"vetical i_det_lenght "<<i_det_lenght<<" distance_vector size "<<distance_vector.size()<<std::endl;
                if (distance_vector.size()>2)
                {
                    std::map<int,int> map_histogram;
                    //std::sort(distance_vector.begin(), distance_vector.end(), comparedistance);//根据距离大小 从小到大排序
                    for (int j = 0; j < distance_vector.size(); ++j)
                    {
                        // if (map_histogram.find(detection_result_class[distance_vector[j].index].iclass)==map_histogram.end())
                        // {
                        //     map_histogram[detection_result_class[distance_vector[j].index].iclass]=0;
                        // }
                        // else
                        // {
                        //     map_histogram[detection_result_class[distance_vector[j].index].iclass]++;
                        // }
                      int key=-1;
                        if (distance_vector[j].lenght==16.5)
                        {
                          key=0;
                        }
                        else if (distance_vector[j].lenght==18)
                        {
                          key=1;
                        }
                        else if (distance_vector[j].lenght==22)
                        {
                          key=2;
                        }

                        if (key!=-1)
                        {
                          if (map_histogram.find(key)==map_histogram.end())
                          {
                              map_histogram[key]=0;
                              //map_histogram.insert(std::pair<int,int>(key,0));
                          }
                          else
                          {
                              map_histogram[key]++;
                          }
                        }
                    }
                    std::map<int,int>::iterator iter;
                    int max_his_class=k%3;//detection_result_class[i].iclass;
                    int max_his_count=0;
                    for (iter=map_histogram.begin(); iter!=map_histogram.end(); iter++)
                    {
                      if (iter->second>max_his_count)
                      {
                        max_his_count=iter->second;
                        max_his_class=iter->first;
                      }
                    }
                    if (max_his_count>=2 && max_his_class!=k%3)
                    {
                        std::cout<<"confuse_vote change label "<<detection_result_class[i].iclass<<" to "<<confuse_label[i_start+max_his_class]<<" max_his_count:"<<max_his_count<<std::endl;
                        std::cout<<"confuse_vote change label i_start"<<i_start<<" max_his_class "<<max_his_class<<std::endl;
                        detection_result_class[i].bcalibrate=true;//是否让纠正后的obj可以去纠正其他obj
                        detection_result_class[i].iclass=confuse_label[i_start+max_his_class];//是否让纠正后的obj可以去纠正其他obj
                    }
                }
            }
        }
    }
}

void confuse_vote_from_size_0(/*list_file,*/vector<Result_detect> & detection_result_class)
{

    int confuse_label[]={2,3,4,
                 -1,5,6,
                 -1,8,9,
                 12,13,14,
                 -1,36,37,
                 39,40,41,
                 -1,43,44};
    char * list_file="/home/liushuai/tiannuocaffe/1detect_bak/size.txt";
    std::vector<int> labels_size_vec;
    std::vector<float> lenght_size_vec;
    std::vector<float> height_size_vec;
    read_cfg(list_file,labels_size_vec,lenght_size_vec,height_size_vec);

      int size_confuse=sizeof(confuse_label)/sizeof(confuse_label[0]);
    for (int i = 0; i < detection_result_class.size(); ++i)
    {
        // if (detection_result_class[i].bcalibrate==true)//现在校正的是没有经过校正的，比如附近没有可以进行比较的，我们采用投票
        // {
        //     continue;
        // }
        for (int k = 0; k < size_confuse; ++k)
        {
            if (confuse_label[k] == detection_result_class[i].iclass)//如果目标是属于大小标签中的
            {
                std::cout<<"confuse_vote_label orignal class:"<<detection_result_class[i].iclass<<std::endl;
                float i_det_lenght=max(detection_result_class[i].x2-detection_result_class[i].x1,
                  detection_result_class[i].y2-detection_result_class[i].y1);
                float i_lenght=-1.0;
                float i_height=-1.0;
                bool bfind = check_label_exist_cfg(detection_result_class[i].iclass,labels_size_vec,lenght_size_vec,height_size_vec,i_lenght,i_height);
                if (bfind)
                {
                    if (-1.0==i_lenght)
                    {
                      continue;
                    }
                }
                else
                {
                      std::cout<<"the lenght of confuse label "<<detection_result_class[i].iclass<<" not in size.txt"<<std::endl; 
                }
                int i_start = 3*(k/3);
                std::vector<Min_distace_index_> distance_vector;
                for (int j = 0; j < detection_result_class.size(); ++j)//从附近的检测到的物体中的尺寸关系
                {
                    //因为易混淆的类别的分类可能是错误的，极大的影响准确性，所以需要把易混淆的类别排除在外
                    bool bconfuse_vote=false;
                    bool bj_confuse=false;
                    int j_confuse_index=-1;
                    for (int m = 0; m < size_confuse; ++m)
                    {
                        if (detection_result_class[j].iclass==confuse_label[m] && i_start==3*(m/3))//是否让纠正后的obj可以去纠正其他obj
                        {
                            bconfuse_vote=true;
                            break;
                        }
                    }
                    // if (confuse_label[i_start]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+1]!=detection_result_class[j].iclass
                    //   &&confuse_label[i_start+2]!=detection_result_class[j].iclass)//排除在同一个混淆类别下的物体 中去比较
                    if(bconfuse_vote)
                    {//do
                        float j_lenght=-1.0;
                        float j_height=-1.0;
                        bool bfind_j = check_label_exist_cfg(detection_result_class[j].iclass,labels_size_vec,lenght_size_vec,height_size_vec,j_lenght,j_height);
                        if (bfind_j)
                        {
                            if (-1.0==j_lenght)
                            {
                              continue;
                            }
                            if (-1.0==j_height)
                            {
                              continue;
                            }
                            float j_det_lenght=max(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float j_det_height=min(detection_result_class[j].x2-detection_result_class[j].x1,
                                                detection_result_class[j].y2-detection_result_class[j].y1);
                            float right_ratio=j_det_lenght/i_det_lenght;
                            //j_det_lenght*choose_lenght/distance_vector[j].lenght/i_det_lenght
                            if (0.93<right_ratio && right_ratio<1.07)//仅根据大小特别接近的进行校正
                            {
                                //calibrate
                                Min_distace_index_ min_distace_index;
                                min_distace_index.lenght = j_lenght;
                                min_distace_index.height = j_height;
                                min_distace_index.index=j;//detection_result_class[j].iclass;
                                min_distace_index.bconfuse=bj_confuse;
                                min_distace_index.confuse_index=j_confuse_index;
                                min_distace_index.distace=sqrt(pow(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x1-detection_result_class[i].x2 , 2)
                                              +pow(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y1-detection_result_class[i].y2,2))/4;

                                //std::cout<<"min_distace_index.distace "<<min_distace_index.distace<<std::endl;//4*6.25*i_det_lenght*i_det_lenght
                                // if (2.5*i_det_lenght>min_distace_index.distace)//当距离大于2.5倍长度的物体 不具有对比意义
                                // {
                                //     distance_vector.push_back(min_distace_index);
                                // }
                               // if ((abs(detection_result_class[j].x1-detection_result_class[i].x1)<1.5*i_det_lenght 
                               //    ||abs(detection_result_class[j].x2-detection_result_class[i].x2)<1.5*i_det_lenght)
                               //    && (abs(detection_result_class[j].y1-detection_result_class[i].y1)<1.5*i_det_lenght
                               //      ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<1.5*i_det_lenght))
                                if(min_distace_index.distace<1.0*i_det_lenght)
                                {
                                  distance_vector.push_back(min_distace_index);
                                } 
                                // if ( (abs(detection_result_class[j].y1-detection_result_class[i].y1)<1.5*i_det_lenght
                                //     ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<1.5*i_det_lenght))//竖直方向
                                // {
                                //   distance_vector.push_back(min_distace_index);
                                // }
                            }
                        }
                    }
                }
                    std::cout<<"vetical i_det_lenght "<<i_det_lenght<<" distance_vector size "<<distance_vector.size()<<std::endl;
                if (distance_vector.size()>2)
                {
                    std::map<int,int> map_histogram;
                    //std::sort(distance_vector.begin(), distance_vector.end(), comparedistance);//根据距离大小 从小到大排序
                    for (int j = 0; j < distance_vector.size(); ++j)
                    {
                        if (map_histogram.find(detection_result_class[distance_vector[j].index].iclass)==map_histogram.end())
                        {
                            map_histogram[detection_result_class[distance_vector[j].index].iclass]=0;
                        }
                        else
                        {
                            map_histogram[detection_result_class[distance_vector[j].index].iclass]++;
                        }
                        
                    }
                    std::map<int,int>::iterator iter;
                    int max_his_class=detection_result_class[i].iclass;
                    int max_his_count=0;
                    for (iter=map_histogram.begin(); iter!=map_histogram.end(); iter++)
                    {
                      if (iter->second>max_his_count)
                      {
                        max_his_count=iter->second;
                        max_his_class=iter->first;
                      }
                    }
                    if (max_his_count>=2 && max_his_class!=detection_result_class[i].iclass)
                    {
                        std::cout<<"confuse_vote change label  "<<detection_result_class[i].iclass<<" to "<<max_his_class<<" max_his_count:"<<max_his_count<<std::endl;
                        detection_result_class[i].bcalibrate=true;//是否让纠正后的obj可以去纠正其他obj
                        detection_result_class[i].iclass=max_his_class;//是否让纠正后的obj可以去纠正其他obj
                    }
                }
            }
        }
    }
}

// void confuse_compare_from_size(/*list_file,*/vector<Result_detect> & detection_result_class)
// {

//     int confuse_label[]={2,3,4,
//                  -1,5,6,
//                  -1,8,9,
//                  12,13,14,
//                  -1,36,37,
//                  39,40,41,
//                  -1,43,44};
//     char * list_file="/home/liushuai/tiannuocaffe/1detect_bak/size.txt";
//     std::vector<int> labels_size_vec;
//     std::vector<float> lenght_size_vec;
//     std::vector<float> height_size_vec;
//     read_cfg(list_file,labels_size_vec,lenght_size_vec,height_size_vec);

 //     int size_confuse=sizeof(confuse_label)/sizeof(confuse_label[0]);
//     for (int i = 0; i < detection_result_class.size(); ++i)
//     {
//         for (int k = 0; k < size_confuse; ++k)
//         {
//             if (confuse_label[k] == detection_result_class[i].iclass)//如果目标是属于大小标签中的
//             {
//                 std::cout<<"confuse_compare_label orignal class:"<<detection_result_class[i].iclass<<std::endl;
//                 float i_det_lenght=max(detection_result_class[i].x2-detection_result_class[i].x1,
//                   detection_result_class[i].y2-detection_result_class[i].y1);
//                 float i_det_height=min(detection_result_class[i].x2-detection_result_class[i].x1,
//                   detection_result_class[i].y2-detection_result_class[i].y1);
//                 float i_lenght=-1.0;
//                 float i_height=-1.0;
//                 bool bfind = check_label_exist_cfg(detection_result_class[i].iclass,labels_size_vec,lenght_size_vec,height_size_vec,i_lenght,i_height);
//                 if (bfind)
//                 {
//                     if (-1.0==i_lenght)
//                     {
//                       continue;
//                     }
//                 }
//                 else
//                 {
//                       std::cout<<"the lenght of confuse label "<<detection_result_class[i].iclass<<" not in size.txt"<<std::endl; 
//                 }
//                 int i_start = 3*(k/3);
//                 std::vector<Min_distace_index_> distance_vector;
//                 // std::vector<Min_distace_index_> distance_vector_0;
//                 // std::vector<Min_distace_index_> distance_vector_1;
//                 // std::vector<Min_distace_index_> distance_vector_2;
//                 int iave=0;
//                 // int ilenght_ave_0=i_det_lenght;
//                 // int ilenght_ave_1=0;
//                 // int ilenght_ave_2=0;
//                 int ilenght_ave[3]={0.0,0.0,0.0};
//                 ilenght_ave[0]=i_det_lenght;
//                 int iheight_ave[3]={0.0,0.0,0.0};
//                 iheight_ave[0]=i_det_height;
//                 int icount[3]={1,0,0};
//                 // int icount_0=1;
//                 // int icount_1=0;
//                 // int icount_2=0;
//                 for (int j = 0; j < detection_result_class.size(); ++j)//从附近的检测到的物体中的尺寸关系
//                 {
//                     //因为易混淆的类别的分类可能是错误的，极大的影响准确性，所以需要把易混淆的类别排除在外
//                     bool bconfuse_vote=false;
//                     for (int m = 0; m < size_confuse; ++m)
//                     {
//                         if (detection_result_class[j].iclass==confuse_label[m] && i_start==3*(m/3))//是否让纠正后的obj可以去纠正其他obj
//                         {
//                             bconfuse_vote=true;
//                             break;
//                         }
//                     }
//                     // if (confuse_label[i_start]!=detection_result_class[j].iclass
//                     //   &&confuse_label[i_start+1]!=detection_result_class[j].iclass
//                     //   &&confuse_label[i_start+2]!=detection_result_class[j].iclass)//排除在同一个混淆类别下的物体 中去比较
//                     if(bconfuse_vote)
//                     {//do
//                         float j_lenght=-1.0;
//                         float j_height=-1.0;
//                         bool bfind_j = check_label_exist_cfg(detection_result_class[j].iclass,labels_size_vec,lenght_size_vec,height_size_vec,j_lenght,j_height);
//                         if (bfind_j)
//                         {
//                             if (-1.0==j_lenght)
//                             {
//                               continue;
//                             }
//                             if (-1.0==j_height)
//                             {
//                               continue;
//                             }
//                             float j_det_lenght=max(detection_result_class[j].x2-detection_result_class[j].x1,
//                                                 detection_result_class[j].y2-detection_result_class[j].y1);
//                             float j_det_height=min(detection_result_class[j].x2-detection_result_class[j].x1,
//                                                 detection_result_class[j].y2-detection_result_class[j].y1);
//                             //j_det_lenght*choose_lenght/distance_vector[j].lenght/i_det_lenght
//                             float right_ratio=j_lenght*j_det_height/j_height/j_det_lenght;
//                             if (0.8<right_ratio && right_ratio<1.2)//仅根据摆放的和原比例差不多的进行校正
//                             {
//                                if ((abs(detection_result_class[j].x1-detection_result_class[i].x1)<1.5*i_det_lenght 
//                                   ||abs(detection_result_class[j].x2-detection_result_class[i].x2)<1.5*i_det_lenght)
//                                   && (abs(detection_result_class[j].y1-detection_result_class[i].y1)<1.5*i_det_lenght
//                                     ||abs(detection_result_class[j].y2-detection_result_class[i].y2)<1.5*i_det_lenght))
//                                 {

//                                 Min_distace_index_ min_distace_index;
//                                 min_distace_index.lenght = j_lenght;
//                                 min_distace_index.height = j_height;
//                                 min_distace_index.index=j;//detection_result_class[j].iclass;
//                                 min_distace_index.distace=0;//sqrt(pow(detection_result_class[j].x1+detection_result_class[j].x2-detection_result_class[i].x1-detection_result_class[i].x2 , 2)
//                                              // +pow(detection_result_class[j].y1+detection_result_class[j].y2-detection_result_class[i].y1-detection_result_class[i].y2,2))/4;

//                                   distance_vector.push_back(min_distace_index);

//                                   float compare_ratio[3];
//                                   compare_ratio[0]=icount[0]>0 ? j_det_lenght*j_det_height/iheight_ave[0]/ilenght_ave[0] :0.0;
//                                   compare_ratio[1]=icount[1]>0 ? j_det_lenght*j_det_height/iheight_ave[1]/ilenght_ave[1] :0.0;
//                                   compare_ratio[2]=icount[2]>0 ? j_det_lenght*j_det_height/iheight_ave[2]/ilenght_ave[2] :0.0;
//                                   float abs_ratio[3];
//                                   abs_ratio[0]=abs(compare_ratio[0]-1.0);
//                                   abs_ratio[1]=abs(compare_ratio[1]-1.0);
//                                   abs_ratio[2]=abs(compare_ratio[2]-1.0);
//                                   // float ave_ratio[3];
//                                   // ave_ratio[0]= abs((icount_0>0 ? ilenght_ave[0]/icount_0/j_det_lenght : 0.0)-1.0); 
//                                   // ave_ratio[1]= abs((icount_1>0 ? ilenght_ave[1]/icount_1/j_det_lenght : 0.0)-1.0);
//                                   // ave_ratio[2]= abs((icount_2>0 ? ilenght_ave[2]/icount_2/j_det_lenght : 0.0)-1.0);
//                                   float lenght_ratio=j_det_lenght/i_det_lenght;
//                                   if (abs_ratio[0]>0.05 && abs_ratio[1]>0.05&& abs_ratio[2]>0.05)
//                                   {
//                                     if (compare_ratio[0]<0.95 )
//                                     {
//                                       if (icount[2]==0)
//                                       {
//                                         icount[2]=icount[1];
//                                         ilenght_ave[2]=ilenght_ave[1];
//                                         iheight_ave[2]=iheight_ave[1];
//                                         icount[1]=icount[0];
//                                         ilenght_ave[1]=ilenght_ave[0];
//                                         iheight_ave[1]=iheight_ave[0];
//                                         icount[0]=1;
//                                         ilenght_ave[0]=j_det_lenght;
//                                         iheight_ave[0]=j_det_height;
//                                       }
//                                     }
//                                     else if (compare_ratio[0]>1.05 && compare_ratio[1]<0.95)
//                                     {
//                                       if (icount[2]==0)
//                                       {
//                                         icount[2]=icount[1];
//                                         ilenght_ave[2]=ilenght_ave[1];
//                                         iheight_ave[2]=iheight_ave[1];
//                                         icount[1]=1;
//                                         ilenght_ave[1]=j_det_lenght;
//                                         iheight_ave[1]=j_det_height;
//                                       }
//                                     }
//                                     else if (compare_ratio[1]>1.05)
//                                     {
//                                       if (icount[2]==0)
//                                       {
//                                         icount[2]=1;
//                                         ilenght_ave[2]=j_det_lenght;
//                                         iheight_ave[2]=j_det_height;
//                                       }
//                                     }
//                                   }
//                                   else if (abs_ratio[0]<=0.05 && abs_ratio[1]>0.05&& abs_ratio[2]>0.05)
//                                   {
                                    
//                                         icount[0]=+1;
//                                         ilenght_ave[0]=+j_det_lenght;
//                                         iheight_ave[0]=+j_det_height;
//                                   }
//                                   else if (abs_ratio[0]>0.05 && abs_ratio[1]<=0.05&& abs_ratio[2]>0.05)
//                                   {
                                    
//                                         icount[1]=+1;
//                                         ilenght_ave[1]=+j_det_lenght;
//                                         iheight_ave[1]=+j_det_height;
//                                   }
//                                   else if (abs_ratio[0]>0.05 && abs_ratio[1]>0.05&& abs_ratio[2]<=0.05)
//                                   {
                                    
//                                         icount[2]=+1;
//                                         ilenght_ave[2]=+j_det_lenght;
//                                         iheight_ave[2]=+j_det_height;
//                                   }
//                                 } 
//                             }
//                         }
//                     }
//                 }
//                 //std::cout<<"vetical i_det_lenght "<<i_det_lenght<<" distance_vector size "<<distance_vector.size()<<std::endl;
//                 if (distance_vector.size()>2)
//                 {
//                     float compare_ratio[3];
//                     compare_ratio[0]=abs((icount[0]>0?i_det_lenght/ilenght_ave[0]:0.0)-1.0);
//                     compare_ratio[1]=abs((icount[1]>0?i_det_lenght/ilenght_ave[1]:0.0)-1.0);
//                     compare_ratio[2]=abs((icount[2]>0?i_det_lenght/ilenght_ave[2]:0.0)-1.0);
//                     float min_comp=compare_ratio[0];
//                     int min_index=i_start;
//                     if (min_comp>compare_ratio[1])
//                     { 
//                       min_comp=compare_ratio[1];
//                       min_index=i_start+1;
//                     }
//                     if (min_comp>compare_ratio[2])
//                     {   
//                       min_comp=compare_ratio[2];
//                       min_index=i_start+2;
//                     }
                    
//                     std::cout<<"confuse_compare change label  "<<detection_result_class[i].iclass<<" to "<<min_comp<<std::endl;
//                     detection_result_class[i].bcalibrate=true;//是否让纠正后的obj可以去纠正其他obj
//                     detection_result_class[i].iclass=confuse_label[min_index];//是否让纠正后的obj可以去纠正其他obj
                    
//                 }
//             }
//         }
//     }
// }

void Detector::Cut4ImageDetect(cv::Mat& img, vector<Result_detect> & detection_result_class,float nms_threhold,float conf_threhold)
{
	vector<Result_detect >  tmp_detections;//detections,
  cv::Mat tmp_img;
  for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++){
          img(cv::Rect(i*0.4*img.cols, j*0.4*img.rows, img.cols*0.6, img.rows*0.6)).copyTo(tmp_img);
 
          tmp_detections.clear();
          Detect(tmp_img,tmp_detections,nms_threhold,conf_threhold);
	        for (int k = 0; k < tmp_detections.size(); ++k)
	        {
	            tmp_detections[k].x1= (tmp_detections[k].x1 + i*0.4*img.cols) ;
	            tmp_detections[k].y1 = (tmp_detections[k].y1 + j*0.4*img.rows) ;
	            tmp_detections[k].x2 = (tmp_detections[k].x2 + i*0.4*img.cols) ;
	            tmp_detections[k].y2 = (tmp_detections[k].y2 + j*0.4*img.rows) ;

              detection_result_class.push_back(tmp_detections[k]);
          }
  }
  DelDetect_after_cut_detect(detection_result_class);

    for (int i = 0; i < detection_result_class.size(); ++i)
    {
      if (i%10==0)
      {
        std::cout<<std::endl;
      }
      std::cout<<detection_result_class[i].iclass<<"->"<<detection_result_class[i].score<<" ";
    }
    std::cout<<std::endl;
  determine_from_size(detection_result_class,false,2.5);//通过周围的非混淆物品进行校验，同时通过校验后的进行校验
  determine_from_size(detection_result_class,true,2.0);//因为顺序的原因，可能校验过后没有成为可校验的对象。再次根据校验后的进行校验
  determine_from_size(detection_result_class,true,4.5);//因为顺序的原因，可能校验过后没有成为可校验的对象。再次根据校验后的进行校验
  confuse_vote_from_size(detection_result_class);//对于周围既没有非混淆物品，也没有校验成功的物品；则进行投票周边同混淆的商品进行校验
  //confuse_compare_from_size(detection_result_class);//对于在同一个混淆类别里头，根据大小尺寸聚类再处理一次
  //determine_from_size(detection_result_class,true);//因为顺序的原因，可能校验过后没有成为可校验的对象。再次根据校验后的进行校验
}

void Do_overlap_confuseclass(vector<Result_detect> &detections,float confidence_threshold)
{
  if (detections.size()<=0)
    return;
  float areai, areaj;
  std::sort(detections.begin(), detections.end(), comparescore);

  for (int i = 0; i < detections.size() - 1; ++i){
    int countsame_score=0;
    bool bfind=false;
    for (int j = i+1;  j < detections.size(); ++j){
      Points pointsi(detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2);
      Points pointsj(detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2);
      float threshold = 0.3;
      // if(detections[i].iclass !=detections[j].iclass)
      // {
      //   threshold = 0.5;
      //   //continue;
      // }
      areai = (pointsi.Xmax - pointsi.Xmin)*(pointsi.Ymax - pointsi.Ymin);
      areaj = (pointsj.Xmax - pointsj.Xmin)*(pointsj.Ymax - pointsj.Ymin);
      //float f_overlap=GetOverLap(detections[i].x1,detections[i].y1,detections[i].x2,detections[i].y2,detections[j].x1,detections[j].y1,detections[j].x2,detections[j].y2);
      if (/*f_overlap>threshold ||*/AreaRate(pointsi, pointsj, areai,threshold) || AreaRate(pointsi, pointsj, areaj,threshold))
      {
          bfind=true;
        // if((detections[i].score-detections[j].score>0) && (detections[i].score-detections[j].score<0.05)
        //   ||(detections[i].score-detections[j].score<0) && (detections[i].score-detections[j].score>-0.05))
        // {
        // }    
        detections.erase(detections.begin() + j);
         j--;
         if (j<=i)
         {
           j=i;
         }
      }
      // if (areai <= areaj)
      // {
      //   if (AreaRate(pointsi, pointsj, areai,threshold)){
      //     detections.erase(detections.begin() + i);
      //     i--;
      //     break;
      //   }
      // }
      // else if (AreaRate(pointsi, pointsj, areaj,threshold))
      // {
      //   detections.erase(detections.begin() + j);
      //   j--;
      // }

    }
    if(!bfind)
    {
      if(detections[i].score<confidence_threshold)
      {
        detections.erase(detections.begin() + i);
        i--;
        if(i<0)
        {
          i=-1;
        }
      }
    }
  }
}

void Detector::Detect(cv::Mat & cv_img, vector<Result_detect> & detection_result_class,float nms_threhold,float conf_threhold )
{
	//const float CONF_THRESH = 0.45;
	const int class_others = 10000;
	const float CONF_for_others =  0.35;
	//const float NMS_THRESH = 0.3;
    const int  max_input_side=1000;
    const int  min_input_side=600;
  float CONF_THRESH = conf_threhold;
  float NMS_THRESH = nms_threhold;

    std::cout<<" enter Detect NMS: "<<NMS_THRESH<<" CONF_THRESH: "<<conf_threhold<<endl;
	cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
	if(cv_img.empty())
    {
        std::cout<<"Can not get the image"<<endl;
        return;
    }
    int max_side = max(cv_img.rows, cv_img.cols);
    int min_side = min(cv_img.rows, cv_img.cols);

    float max_side_scale = float(max_side) / float(max_input_side);
    float min_side_scale = float(min_side) /float( min_input_side);
    float max_scale=max(max_side_scale, min_side_scale);

    float img_scale = 1;

    if(max_scale > 1)
    {
        img_scale = float(1) / max_scale;
    }

	int height = int(cv_img.rows * img_scale);
	int width = int(cv_img.cols * img_scale);
	int num_out;
	cv::Mat cv_resized;

	float im_info[3];
	float data_buf[height*width*3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt;
	const float* rois;
	const float* pred_cls;
	int num;

	for (int h = 0; h < cv_img.rows; ++h )
	{
		for (int w = 0; w < cv_img.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);

		}
	}
	// cv_img.copyTo(cv_new);

	cv::resize(cv_new, cv_resized, cv::Size(width, height));
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;

	for (int h = 0; h < height; ++h )
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}

	net_->blob_by_name("data")->Reshape(1, 3, height, width);
	//net_->blob_by_name("data")->set_cpu_data(data_buf);
	Blob<float> * input_blobs= net_->input_blobs()[0];
    std::cout<<" copy data  "<<endl;
    switch(Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    case Caffe::GPU:
        caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
        break;
    default:
        LOG(FATAL)<<"Unknow Caffe mode";
    }
	net_->blob_by_name("im_info")->set_cpu_data(im_info);
    std::cout<<" ForwardFrom  "<<endl;
	net_->ForwardFrom(0);
    std::cout<<" ForwardFrom over "<<endl;
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	num = net_->blob_by_name("rois")->num();

	rois = net_->blob_by_name("rois")->cpu_data();
	pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	boxes = new float[num*4];
	pred = new float[num*5*m_iclass_num];
	pred_per_class = new float[num*5];
	sorted_pred_cls = new float[num*5];
	keep = new int[num];

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < 4; c++)
		{
			boxes[n*4+c] = rois[n*5+c+1] / img_scale;
		}
	}

	bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
	for (int i = 1; i < m_iclass_num; i ++)
	{
		for (int j = 0; j< num; j++)
		{
			for (int k=0; k<5; k++)
				pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
		}
		boxes_sort(num, pred_per_class, sorted_pred_cls);
		
		switch (Caffe::mode()) 
		{
		  case Caffe::CPU:
		  {
			cpu_nms(sorted_pred_cls, num, 5, NMS_THRESH,( i == class_others ? CONF_for_others : CONF_THRESH ), detection_result_class,i);
		  }
		  break;
		  case Caffe::GPU:
		  {
			_nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, m_gpuid>=0 ? m_gpuid: 0);
			int k=0;
			while(k < num_out  && sorted_pred_cls[keep[k]*5+4]>( i == class_others ? CONF_for_others : CONF_THRESH ) )
      //while(k < num_out  && sorted_pred_cls[keep[k]*5+4]>0.25)//0.16
			{
        //   if(sorted_pred_cls[keep[k]*5+4]>( i == class_others ? CONF_for_others : CONF_THRESH ) )//231
        //   {
        //     k++;
        //     continue;
        //   }
        // else  if(sorted_pred_cls[keep[k]*5+4]>0.1 &&i==231)
        // {
          
        // }
        // else{k++;continue;}
	        	//std::cout<<"3num  "<<num<<"num_out  "<<num_out<<" k "<<k<<endl;
	        	//std::cout<<"4num  "<<num<<"num_out  "<<num_out<<" keep "<<keep[k]<<endl;
				if(k>=num_out)
					break;
				//detection format x1 y1 width height
				Result_detect obj_Result_detect;
				obj_Result_detect.x1 = sorted_pred_cls[keep[k]*5+0];
				obj_Result_detect.y1 = sorted_pred_cls[keep[k]*5+1];
				obj_Result_detect.x2 = sorted_pred_cls[keep[k]*5+2];
				obj_Result_detect.y2 = sorted_pred_cls[keep[k]*5+3];
				obj_Result_detect.score = sorted_pred_cls[keep[k]*5+4];
				obj_Result_detect.iclass = i;
        obj_Result_detect.calibrate_preclass=i;
				detection_result_class.push_back(obj_Result_detect);
		        // detection_result_class.push_back(cv::Rect(sorted_pred_cls[keep[k]*5+0],
		        //                                     sorted_pred_cls[keep[k]*5+1],
		        //                                     sorted_pred_cls[keep[k]*5+2]-sorted_pred_cls[keep[k]*5+0],
		        //                                     sorted_pred_cls[keep[k]*5+3]-sorted_pred_cls[keep[k]*5+1]));
		        k++;
	        //std::cout<<"5num  "<<num<<"num_out  "<<num_out<<" k "<<k<<endl;
			}
		  }
		  break;
		  default:
    		std::cout<<"no device for nms"<<endl;
    		break;
    	}
		
		//wheen paramter sortout,gpu error,may need alignment
		// int sortout = 0;
		// boxes_sort(num, pred_per_class, sorted_pred_cls,CONF_THRESH,sortout);
  //       //std::cout<<"num  "<<num<<"sortout  "<<sortout<<endl;
		// _nms(keep, &num_out, sorted_pred_cls, /*num*/sortout, 5, NMS_THRESH, m_gpuid>=0 ? m_gpuid: 0);

        //std::cout<<"2num  "<<num<<"num_out  "<<num_out<<endl;
        //for visualize only
		//vis_detections(cv_img, keep, num_out, sorted_pred_cls, CONF_THRESH);
    	//std::cout<<" m_iclass_num  "<<i<<endl;
	}
  Do_overlap_confuseclass(detection_result_class,CONF_THRESH);

    //cv::imwrite("vis.jpg",cv_img);
	delete []boxes;
	delete []pred;
	delete []pred_per_class;
	delete []keep;
	delete []sorted_pred_cls;
  //DelDetect_after_cut_detect(detection_result_class);
    std::cout<<" over  "<<endl;
}

float GetOverLap(float i_x1, float i_y1,float i_x2,float i_y2,  float j_x1,float j_y1,float j_x2,float j_y2) 
{
    // int C0 = r1.x;
    // int C1 = r1.x + r1.width;
    // int R0 = r1.y;
    // int R1 = r1.y + r1.height;
    // int pC0 = r2.x;
    // int pC1 = r2.x + r2.width;
    // int pR0 = r2.y;
    // int pR1 = r2.y + r2.height;
    float C0 = i_x1;
    float C1 = i_x2;
    float R0 = i_y1;
    float R1 = i_y2;
    float pC0 = j_x1;
    float pC1 = j_x2;
    float pR0 = j_y1;
    float pR1 = j_y2;


    float mR0, mR1, mC0, mC1;
    float midArea, preArea, currArea;
    float overlapRat = -1;
    currArea = (C1 - C0) * (R1 - R0);
    mR0 = (R0 > pR0 ? R0 : pR0);
    mC0 = (C0 > pC0 ? C0 : pC0);
    mR1 = (R1 < pR1 ? R1 : pR1);
    mC1 = (C1 < pC1 ? C1 : pC1);
    midArea = (mR1 - mR0) * (mC1 - mC0);    // intersection area
    preArea = (pR1 - pR0) * (pC1 - pC0);

   // float totalarea = MIN(currArea, preArea);
    //totalarea = MIN(totalarea, currArea + preArea - midArea);
    //overlapRat = (float) (midArea) / (float) (totalarea);
    if((currArea + preArea - midArea) == 0)
    {
    	return -1;
    }
    overlapRat = (float)(midArea) / (float)(currArea + preArea - midArea);

    if (pC1 < C0 || pC0 > C1 || pR1 < R0 || pR0 > R1) {
        overlapRat = -1;
    }


    return overlapRat;
}

void Detector::cpu_nms(const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, float confidence_threshold,vector<Result_detect> & detection_result_class ,int iclass)
{
	const float* tmp_box_head = boxes_host;
	bool* keep_out = new bool[boxes_num];
	if(!keep_out)
	{
		return;
	}
	memset(keep_out,1,boxes_num);

    for (int i = 0; i < boxes_num; i++)
    {
        //const vector<float> &d = detections[i];
        float score1 = tmp_box_head[i*boxes_dim + 4];
        if (score1 < confidence_threshold)
        {
        	keep_out[i] = false;
        	break;
        }
        if(false == keep_out[i])
        {
        	continue;
        }
        float i_x1 = tmp_box_head[i*boxes_dim];
        float i_y1 = tmp_box_head[i*boxes_dim + 1];
        float i_x2 = tmp_box_head[i*boxes_dim + 2];
        float i_y2 = tmp_box_head[i*boxes_dim + 3];
        keep_out[i] = true;
		Result_detect obj_Result_detect;
		obj_Result_detect.x1 = i_x1;
		obj_Result_detect.y1 = i_y1;
		obj_Result_detect.x2 = i_x2;
		obj_Result_detect.y2 = i_y2;
		obj_Result_detect.score = score1;
		obj_Result_detect.iclass = iclass;
		detection_result_class.push_back(obj_Result_detect);
        // Rect rt;
        // rt.x = d[3] * width;
        // rt.y = d[4] * height;
        // rt.width = (d[5] - d[3]) * width;
        // rt.height = (d[6] - d[4]) * height;

        for (int j = i + 1; j < boxes_num; j++) 
        {
            float score2 = tmp_box_head[j*boxes_dim + 4];
            // const vector<float> &d2 = detections[j];
            // float score2 = d2[2];
            if (score2 < confidence_threshold)
            {
            	keep_out[j]=false;
            	break;
            }
	        float j_x1 = tmp_box_head[j*boxes_dim];
	        float j_y1 = tmp_box_head[j*boxes_dim + 1];
	        float j_x2 = tmp_box_head[j*boxes_dim + 2];
	        float j_y2 = tmp_box_head[j*boxes_dim + 3];

            // Rect rt2;
            // rt2.x = d2[3] * width;
            // rt2.y = d2[4] * height;
            // rt2.width = (d2[5] - d2[3]) * width;
            // rt2.height = (d2[6] - d2[4]) * height;


            float overlapRat = GetOverLap(i_x1,i_y1,i_x2,i_y2,j_x1,j_y1,j_x2,j_y2);
            if (overlapRat > nms_overlap_thresh) 
            {
            	keep_out[j]=false;
                // if (score1 > score2) {
                //     detections.erase(detections.begin() + j);
                //     curSize--;
                //     j--;
                // } else {

                //     detections.erase(detections.begin() + i);
                //     curSize--;
                //     i--;
                //     break;

                // }
            }

        }

    }
    if(keep_out)
    {
    	delete []keep_out;
    	keep_out = NULL;
    }
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detect
 *  Description:  Perform detection operation
 *                 Warning the max input size should less than 1000*600
 * =====================================================================================
 */
//perform detection operation
//input image max size 1000*600
void Detector::Detect(cv::Mat & cv_img, vector<cv::Rect> & detection_result ,float nms_threhold,float conf_threhold)
{
	float CONF_THRESH = conf_threhold;
	float NMS_THRESH = nms_threhold;
    const int  max_input_side=1000;
    const int  min_input_side=600;

	cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
	if(cv_img.empty())
    {
        std::cout<<"Can not get the image"<<endl;
        return;
    }
    int max_side = max(cv_img.rows, cv_img.cols);
    int min_side = min(cv_img.rows, cv_img.cols);

    float max_side_scale = float(max_side) / float(max_input_side);
    float min_side_scale = float(min_side) /float( min_input_side);
    float max_scale=max(max_side_scale, min_side_scale);

    float img_scale = 1;

    if(max_scale > 1)
    {
        img_scale = float(1) / max_scale;
    }

	int height = int(cv_img.rows * img_scale);
	int width = int(cv_img.cols * img_scale);
	int num_out;
	cv::Mat cv_resized;

	float im_info[3];
	float data_buf[height*width*3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt;
	const float* rois;
	const float* pred_cls;
	int num;

	for (int h = 0; h < cv_img.rows; ++h )
	{
		for (int w = 0; w < cv_img.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);

		}
	}
	// cv_img.copyTo(cv_new);

	cv::resize(cv_new, cv_resized, cv::Size(width, height));
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;

	for (int h = 0; h < height; ++h )
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}

	net_->blob_by_name("data")->Reshape(1, 3, height, width);
	//net_->blob_by_name("data")->set_cpu_data(data_buf);
	Blob<float> * input_blobs= net_->input_blobs()[0];
    switch(Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    case Caffe::GPU:
        caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
        break;
    default:
        LOG(FATAL)<<"Unknow Caffe mode";
    }
	net_->blob_by_name("im_info")->set_cpu_data(im_info);
	net_->ForwardFrom(0);
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	num = net_->blob_by_name("rois")->num();

	rois = net_->blob_by_name("rois")->cpu_data();
	pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	boxes = new float[num*4];
	pred = new float[num*5*m_iclass_num];
	pred_per_class = new float[num*5];
	sorted_pred_cls = new float[num*5];
	keep = new int[num];

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < 4; c++)
		{
			boxes[n*4+c] = rois[n*5+c+1] / img_scale;
		}
	}

	bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
	for (int i = 1; i < m_iclass_num; i ++)
	{
		for (int j = 0; j< num; j++)
		{
			for (int k=0; k<5; k++)
				pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
		}
		int sortout = 0;
		//boxes_sort(num, pred_per_class, sorted_pred_cls,CONF_THRESH,sortout);
		boxes_sort(num, pred_per_class, sorted_pred_cls);
		_nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, m_gpuid>=0 ? m_gpuid: 0);
        //for visualize only
		//vis_detections(cv_img, keep, num_out, sorted_pred_cls, CONF_THRESH);
		int k=0;
		while(sorted_pred_cls[keep[k]*5+4]>CONF_THRESH && k < num_out)
		{
			if(k>=num_out)
				break;
			//detection format x1 y1 width height
	        detection_result.push_back(cv::Rect(sorted_pred_cls[keep[k]*5+0],
	                                            sorted_pred_cls[keep[k]*5+1],
	                                            sorted_pred_cls[keep[k]*5+2]-sorted_pred_cls[keep[k]*5+0],
	                                            sorted_pred_cls[keep[k]*5+3]-sorted_pred_cls[keep[k]*5+1]));
	        k++;
		}
	}


    //cv::imwrite("vis.jpg",cv_img);
	delete []boxes;
	delete []pred;
	delete []pred_per_class;
	delete []keep;
	delete []sorted_pred_cls;
  //DelDetect_after_cut_detect(detection_result_class);

}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  vis_detections
 *  Description:  Visuallize the detection result
 * =====================================================================================
 */
void Detector::vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH)
{
	int i=0;
	while(sorted_pred_cls[keep[i]*5+4]>CONF_THRESH && i < num_out)
	{
		if(i>=num_out)
			return;
		cv::rectangle(image,cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]),cv::Point(sorted_pred_cls[keep[i]*5+2], sorted_pred_cls[keep[i]*5+3]),cv::Scalar(255,0,0));
		i++;
	}
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
void Detector::boxes_sort(int num, const float* pred, float* sorted_pred)
{
	vector<Info> my;
	Info tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i*5 + 4];
		tmp.head = pred + i*5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare);
	for (int i=0; i<num; i++)
	{
		for (int j=0; j<5; j++)
			sorted_pred[i*5+j] = my[i].head[j];
	}
}

void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred,float fTHRESH,int& num_THRESHout)
{
	num_THRESHout=0;
	vector<Info> my;
	Info tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i*5 + 4];
		tmp.head = pred + i*5;
		//my.push_back(tmp);
		if(tmp.score > fTHRESH)
		{
			num_THRESHout++;
			my.push_back(tmp);
		}
	}
	std::sort(my.begin(), my.end(), compare);
	for (int i=0; i<num_THRESHout/*num*/; i++)
	{
		for (int j=0; j<5; j++)
			sorted_pred[i*5+j] = my[i].head[j];
	}
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
{
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	for(int i=0; i< num; i++)
	{
		width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
		height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
		ctr_x = boxes[i*4+0] + 0.5 * width;
		ctr_y = boxes[i*4+1] + 0.5 * height;
		for (int j=0; j< m_iclass_num; j++)
		{
			if(bagnostic)
			{
				dx = box_deltas[(i*2+1)*4+0];
				dy = box_deltas[(i*2+1)*4+1];
				dw = box_deltas[(i*2+1)*4+2];
				dh = box_deltas[(i*2+1)*4+3];	
			}
			else
			{
				// dx = box_deltas[(i*m_iclass_num+j)*4+0];
				// dy = box_deltas[(i*m_iclass_num+j)*4+1];
				// dw = box_deltas[(i*m_iclass_num+j)*4+2];
				// dh = box_deltas[(i*m_iclass_num+j)*4+3];
			}
			pred_ctr_x = ctr_x + width*dx;
			pred_ctr_y = ctr_y + height*dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);
			pred[(j*num+i)*5+0] = max(min(pred_ctr_x - 0.5* pred_w, img_width -1), 0);
			pred[(j*num+i)*5+1] = max(min(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
			pred[(j*num+i)*5+2] = max(min(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
			pred[(j*num+i)*5+3] = max(min(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
			pred[(j*num+i)*5+4] = pred_cls[i*m_iclass_num+j];			
		}
	}

}
