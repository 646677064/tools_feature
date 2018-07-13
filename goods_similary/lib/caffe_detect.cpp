#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe_detect.hpp"
using namespace caffe;
using namespace std;

// int sumarray[100]={7535,6957,7966,6009,2301,13429,12191,3373,18461,12564,\
//   3214,14970,2966,14208,13417,6718,3373,5367,5291,4196,\
//   1995,6435,8218,10368,6508,2813,5323,6991,2187,6995,\
//   15378,9028,8684,4866,9008,18954,5655,12478,8691,9528,\
//   4142,7932,9119,10037,12101,10298,7520,8960,18866,5725,\
//   4550,3508,2535,5573,2857,3509,3352,3735,21552,15246,17069,\
//   4509,13479,10691,10072,3829,9143,14722,16917,6318,13765,16307,\
//   11619,9669,17382,4746,1999,16816,1980,2469,4405,5884,3674,\
//   10437,15487,11433,14302,11582,4993,9342,8349,9137,15664,\
//   1420,7764,1589,20238,9886,12130,6414};
 /*
 * ===  FUNCTION  ======================================================================
 *         Name:  EV2641_InitCarDetector
 *  Description:  Load the model file and weights file ,set GPUID
 * =====================================================================================
 */
int EV2641_InitCarDetector(const char * model_file, const  char * weights_file, int classnum,const int GPUID , Detector * &handle){
    handle = new Detector(model_file, weights_file,classnum, GPUID);
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
        //std::cout<<"cpu mode "<<endl;
      Caffe::set_mode(Caffe::CPU);
    }
    else
    {
        //std::cout<<"gpu mode "<<GPUID<<endl;
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(GPUID);
    }
  net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  batch_size_=8;
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
        threshold = 0.7;
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


void Detector::Cut9ImageDetect(cv::Mat& img, vector<Result_detect> & detection_result_class)
{
  vector<Result_detect >  tmp_detections;//detections,
  cv::Mat tmp_img;
  for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++){
          img(cv::Rect(i*0.3*img.cols, j*0.3*img.rows, img.cols*0.4, img.rows*0.4)).copyTo(tmp_img);
 
          tmp_detections.clear();
          Detect(tmp_img,tmp_detections);
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
struct area_index
{
  int area;
  int index;
  int iclass;
  bool bdelete;
};

bool compareb(const area_index& Info1, const area_index& Info2)
{
  return Info1.area > Info2.area;
}

void delete_very_largebox(vector<Result_detect> & tmp_detections,int num)
{
  std::vector<area_index> vector_area_index;
  for(int ix = 1;ix<num;ix++)
  {
    vector_area_index.clear();
    vector<Result_detect>::iterator iter = tmp_detections.begin();
    int index_det = 0;
    for(;iter !=tmp_detections.end();iter++)
    {
      if(iter->iclass == ix)
      {
        area_index tmp_area_index;
        tmp_area_index.index = index_det;
        tmp_area_index.iclass = iter->iclass;
        tmp_area_index.area = iter->area;
        tmp_area_index.bdelete = false;
        vector_area_index.push_back(tmp_area_index);
      }
      index_det++;
    }
    //std::cout<<ix<<"111aaa size:"<<endl;
    int size = vector_area_index.size();
    if(size<3)
    {
      continue;
    }
    //std::cout<<ix<<"aaa size:"<<size<<endl;
    std::sort(vector_area_index.begin(), vector_area_index.end(), compareb);
    int iaverage = 0;
    //std::cout<<ix<<"bbbb"<<endl;
    vector<area_index>::iterator iter1 = vector_area_index.begin();
    for(;iter1 != vector_area_index.end();iter1++)
    {
      iaverage += (iter1->area);
    }
    iaverage = iaverage/size ;
    //std::cout<<ix<<"cccc"<<endl;
    if(size>3)//拉依达准则
    {
      if((vector_area_index[0].area - vector_area_index[1].area) >iaverage)
      {
        vector_area_index[0].bdelete = true;
        tmp_detections[vector_area_index[0].index].bdelete = true;
      }
      if((vector_area_index[1].area - vector_area_index[2].area) >iaverage)
      {
        vector_area_index[1].bdelete = true;
        vector_area_index[0].bdelete = true;
        tmp_detections[vector_area_index[0].index].bdelete = true;
        tmp_detections[vector_area_index[1].index].bdelete = true;
      }
    }
    //std::cout<<ix<<"dddd"<<endl;
  }
}
void Detector::Cut4ImageDetect(cv::Mat& img, vector<Result_detect> & detection_result_class)
{
  vector<Result_detect >  tmp_detections;//detections,
  cv::Mat tmp_img;
  for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++){
          img(cv::Rect(i*0.4*img.cols, j*0.4*img.rows, img.cols*0.6, img.rows*0.6)).copyTo(tmp_img);
 
          tmp_detections.clear();
          Detect(tmp_img,tmp_detections);
          for (int k = 0; k < tmp_detections.size(); ++k)
          {
              tmp_detections[k].x1= (tmp_detections[k].x1 + i*0.4*img.cols) ;
              tmp_detections[k].y1 = (tmp_detections[k].y1 + j*0.4*img.rows) ;
              tmp_detections[k].x2 = (tmp_detections[k].x2 + i*0.4*img.cols) ;
              tmp_detections[k].y2 = (tmp_detections[k].y2 + j*0.4*img.rows) ;

              detection_result_class.push_back(tmp_detections[k]);
          }
  }

  //statistic average 
  //remove large obj
  // tmp_detections.clear();
  // //tmp_detections = detection_result_class;
  // vector<Result_detect>::iterator iter_detection_result_class = detection_result_class.begin();
  // for(;iter_detection_result_class !=detection_result_class.end();iter_detection_result_class++)
  // {
  //   tmp_detections.push_back(*iter_detection_result_class);
  // }
  // detection_result_class.clear();
  // delete_very_largebox(tmp_detections,m_iclass_num);
  // vector<Result_detect>::iterator iter = tmp_detections.begin();
  // for(;iter !=tmp_detections.end();iter++)
  // {
  //   if(iter->bdelete == false)
  //   {
  //     detection_result_class.push_back(*iter);
  //   }
  // }

  DelDetect_after_cut_detect(detection_result_class);
}

void Detector::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
          cv::Mat channel(height, width, CV_32FC1, input_data);
          input_channels.push_back(channel);
          input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
    // cv::imshow("bla", input_batch->at(1).at(0));
    // cv::waitKey(1);
}

void Detector::PreprocessBatch(const vector<cv::Mat> imgs,
                                      std::vector< std::vector<cv::Mat> >* input_batch){
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
          cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
          cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
          cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
          cv::cvtColor(img, sample, CV_GRAY2BGR);
        else
          sample = img;

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
          cv::resize(sample, sample_resized, input_geometry_);
        else
          sample_resized = sample;

        cv::Mat sample_float;
        if (num_channels_ == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);

        cv::Mat sample_normalized;
  cv::Scalar channel_mean(110.676,115.771,123.191);
  //cv::Scalar channel_mean(123.191,115.771,110.676);
  cv::Mat mean_(input_geometry_, sample_float.type(), channel_mean);
  std::cout<<"mean:"<<mean_.rows<<" "<<mean_.cols<<" "<<mean_.channels()<<std::endl;
  std::cout<<"sample_float:"<<sample_float.rows<<" "<<sample_float.cols<<" "<<sample_float.channels()<<std::endl;
  cv::subtract(sample_float, mean_, sample_normalized);

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        cv::split(sample_normalized, *input_channels);

//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//              == net_->input_blobs()[0]->cpu_data())
//          << "Input channels are not wrapping the input layer of the network.";
    }
}

void Detector::get_feature(vector< cv::Mat >& vec_imgs, vector<vector<float> > & vec_feature ,int ibatch,const string layername)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  batch_size_=ibatch;
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);

  /* Forward dimension change to all layers. */
  net_->Reshape();
  std::vector< std::vector<cv::Mat> > input_batch;
  WrapBatchInputLayer(&input_batch);
  PreprocessBatch(vec_imgs, &input_batch);

  net_->ForwardFrom(0);
    std::cout<<" ForwardFrom over "<<endl;
  // p_freature = net_->blob_by_name(layername)->cpu_data(); //pool5 1 2048 1 1 fc368
  // num = net_->blob_by_name(layername)->channels();
  //Blob<float>* output_layer = net_->output_blobs()[0];
  shared_ptr<Blob<float> > output_layer=net_->blob_by_name(layername);
  std::cout<<"shape size :"<<output_layer->shape().size()<<endl;
  std::cout<<"shape 0 :"<<output_layer->shape()[0]<<endl;
  std::cout<<"shape 1 :"<<output_layer->shape()[1]<<endl;
  std::cout<<"shape 2 :"<<output_layer->shape()[2]<<endl;
  std::cout<<"shape 3 :"<<output_layer->shape()[3]<<endl;
  std::cout<<"count 1 :"<<output_layer->count(1)<<endl;
  for(int i=0;i<ibatch;i++)
  {
  const float* begin = output_layer->cpu_data()+i*output_layer->count(1);
  const float* end = begin + output_layer->count(1);
    std::vector<float> tmp_float_vec(begin, end);
  std::cout<<"count push_back :"<<endl;
    vec_feature.push_back(tmp_float_vec);
    // begin=end;
    // end = begin + output_layer->count(1);
  }
  //return std::vector<float>(begin, end);



  //   std::cout<<" enter get_feature  "<<endl;
  
  //   if(cv_img.empty())
  //   {
  //       std::cout<<"Can not get the image"<<endl;
  //       return;
  //   }
  //   cv::Mat bgrImg;
  //   if (cv_img.channels() == 4)
  //   {
  //       cv::cvtColor(cv_img, bgrImg, CV_BGRA2BGR);
  //   } 
  //   else if (cv_img.channels() == 1)
  //   {
  //       cv::cvtColor(cv_img, bgrImg, CV_GRAY2BGR);
  //   } 
  //   else
  //   {
  //       bgrImg = cv_img;
  //   }
  // cv::Mat cv_new(bgrImg.rows, bgrImg.cols, CV_32FC3, cv::Scalar(0,0,0));

  //    float img_scale = 1;


  // int height = 224;//int(cv_img.rows * img_scale);
  // int width = 224;//int(cv_img.cols * img_scale);
  // int num_out;
  // cv::Mat cv_resized;

  // float im_info[3];
  // float* data_buf = new float[height*width*3];
  // if(!data_buf)
  // {
  //     std::cout<<"data_buf new error"<<endl;
  //   return;
  // }
  // const float *p_freature = NULL;
  // // float *pred = NULL;
  // // float *pred_per_class = NULL;
  // // float *sorted_pred_cls = NULL;
  // //int *keep = NULL;
  // // const float* bbox_delt;
  // // const float* rois;
  // // const float* pred_cls;
  // int num=0;

  // for (int h = 0; h < bgrImg.rows; ++h )
  // {
  //   for (int w = 0; w < bgrImg.cols; ++w)
  //   {
  //     // cv_new.at<cv::Vec3f>(cv::Point(h, w))[0] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[0])-float(110.676);
  //     // cv_new.at<cv::Vec3f>(cv::Point(h, w))[1] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[1])-float(115.771);
  //     // cv_new.at<cv::Vec3f>(cv::Point(h, w))[2] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[2])-float(123.191);
  //     cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[0])-float(110.676);
  //     cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.771);
  //     cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[2])-float(123.191);

  //   }
  // }
  // // cv_img.copyTo(cv_new);

  // cv::resize(cv_new, cv_resized, cv::Size(width, height));
  // im_info[0] = cv_resized.rows;
  // im_info[1] = cv_resized.cols;
  // im_info[2] = img_scale;

  // for (int h = 0; h < height; ++h )
  // {
  //   for (int w = 0; w < width; ++w)
  //   {
  //     data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
  //     data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
  //     data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);      
  //     // data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[0]);
  //     // data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[1]);
  //     // data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[2]);
  //   }
  // }
  // //*/

  // //net_->Reshape();
  //   //std::cout<<(net_->blob_names()[net_->input_blob_indices()[0]])<<net_->blob_by_name("data")->shape_string()<<endl;
  // net_->blob_by_name("data")->Reshape(1, 3, height, width);
  // Blob<float> * input_blobs= net_->input_blobs()[0];
  // Blob<float> * input_blobs_image= net_->input_blobs()[1];
  // net_->blob_by_name("data")->set_cpu_data(data_buf);
  //   std::cout<<" ForwardFrom  "<<endl;
  // net_->ForwardFrom(0);
  //   std::cout<<" ForwardFrom over "<<endl;
  // p_freature = net_->blob_by_name(layername)->cpu_data(); //pool5 1 2048 1 1 fc368
  // num = net_->blob_by_name(layername)->channels();//pool5 fc368
  // //std::cout<<" pool5 num:  "<<net_->blob_by_name("pool5")->num()<<endl;
  //   // std::cout<<" pool5 channels:  "<<net_->blob_by_name("pool5")->channels()<<endl;
  //   // std::cout<<" pool5 height:  "<<net_->blob_by_name("pool5")->height()<<endl;
  //   // std::cout<<" pool5 width:  "<<net_->blob_by_name("pool5")->width()<<endl;
  // // num = net_->blob_by_name("rois")->num();

  // // rois = net_->blob_by_name("rois")->cpu_data();
  // // pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
  // // boxes = new float[num*4];
  // // pred = new float[num*5*m_iclass_num];
  // //p_freature = new float[num];
  // // sorted_pred_cls = new float[num*5];
  // //keep = new int[num];

  // for (int n = 0; n < num; n++)
  // {
  //   //p_freature[n] = num[n];
  //   vec_feature.push_back(p_freature[n]);
  //   // for (int c = 0; c < 4; c++)
  //   // {
  //   //   boxes[n*4+c] = rois[n*5+c+1] / img_scale;
  //   // }
  // }



  // //delete []p_freature;
  //   //cv::imwrite("vis.jpg",cv_img);
  // //delete []boxes;
  // //delete []pred;
  // //delete []pred_per_class;
  // //delete []keep;
  // //delete []sorted_pred_cls;
  // delete []data_buf;
}

void Detector::get_feature(cv::Mat & cv_img, vector<float> & vec_feature ,const string layername)
{
    std::cout<<" enter get_feature  "<<endl;
  
  if(cv_img.empty())
    {
        std::cout<<"Can not get the image"<<endl;
        return;
    }
    cv::Mat bgrImg;
    if (cv_img.channels() == 4)
    {
        cv::cvtColor(cv_img, bgrImg, CV_BGRA2BGR);
    } else if (cv_img.channels() == 1)
    {
        cv::cvtColor(cv_img, bgrImg, CV_GRAY2BGR);
    } else
    {
        bgrImg = cv_img;
    }
  /*
    cv::Mat cv_resized;
  int height = 224;//int(cv_img.rows * img_scale);
  int width = 224;//int(cv_img.cols * img_scale);
  cv::resize(bgrImg, cv_resized, cv::Size(width, height));
  cv::Mat cv_new(cv_resized.rows, cv_resized.cols, CV_32FC3, cv::Scalar(0,0,0));
  float im_info[3];
  //float data_buf[height*width*3];
  float* data_buf = new float[height*width*3];
  if(!data_buf)
  {
      std::cout<<"data_buf new error"<<endl;
    return;
  }
  float img_scale = 1.0;
  for (int h = 0; h < cv_resized.rows; ++h )
  {
    for (int w = 0; w < cv_resized.cols; ++w)
    {
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[0] = float(cv_resized.at<cv::Vec3b>(cv::Point(h, w))[0])-float(110.676);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[1] = float(cv_resized.at<cv::Vec3b>(cv::Point(h, w))[1])-float(115.771);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[2] = float(cv_resized.at<cv::Vec3b>(cv::Point(h, w))[2])-float(123.191);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[0])-float(110.676);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.771);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[2])-float(123.191);

    }
    //cout<<endl;
  }
  int num=0;
  const float *p_freature = NULL;
  im_info[0] = cv_new.rows;
  im_info[1] = cv_new.cols;
  im_info[2] = img_scale;
      std::cout<<"im_info:"<<im_info[0]<<im_info[1]<<im_info[2]<<endl;

  for (int h = 0; h < height; ++h )
  {
    for (int w = 0; w < width; ++w)
    {
      data_buf[(0*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[0]);
      data_buf[(1*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[1]);
      data_buf[(2*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[2]);
      // data_buf[(0*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(h, w))[0]);
      // data_buf[(1*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(h, w))[1]);
      // data_buf[(2*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(h, w))[2]);
      //std::cout<<float(cv_new.at<cv::Vec3b>(cv::Point(w, h))[0])<<" ";
      //std::cout<<data_buf[(0*height+h)*width+w]<<" ";
    }
    //cout<<endl;
  }
  //memcpy(data_buf,cv_new.data,sizeof(float)*height*width*3);
  //*/
  // ///============================================================================================
///*
  cv::Mat cv_new(bgrImg.rows, bgrImg.cols, CV_32FC3, cv::Scalar(0,0,0));

     float img_scale = 1;


  int height = 224;//int(cv_img.rows * img_scale);
  int width = 224;//int(cv_img.cols * img_scale);
  int num_out;
  cv::Mat cv_resized;

    //std::cout<<"data_buf"<<endl;
  float im_info[3];
  //float data_buf[height*width*3];
  float* data_buf = new float[height*width*3];
  if(!data_buf)
  {
      std::cout<<"data_buf new error"<<endl;
    return;
  }
    //std::cout<<"data_buf1"<<endl;
  const float *p_freature = NULL;
  // float *pred = NULL;
  // float *pred_per_class = NULL;
  // float *sorted_pred_cls = NULL;
  //int *keep = NULL;
  // const float* bbox_delt;
  // const float* rois;
  // const float* pred_cls;
  int num=0;

  for (int h = 0; h < bgrImg.rows; ++h )
  {
    for (int w = 0; w < bgrImg.cols; ++w)
    {
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[0] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[0])-float(110.676);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[1] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[1])-float(115.771);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[2] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[2])-float(123.191);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[0])-float(110.676);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.771);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[2])-float(123.191);

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
      // data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[0]);
      // data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[1]);
      // data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[2]);
    }
  }
  //*/

  //net_->Reshape();
    //std::cout<<(net_->blob_names()[net_->input_blob_indices()[0]])<<net_->blob_by_name("data")->shape_string()<<endl;
  net_->blob_by_name("data")->Reshape(1, 3, height, width);
  Blob<float> * input_blobs= net_->input_blobs()[0];
  Blob<float> * input_blobs_image= net_->input_blobs()[1];


  //       // memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
  //       //memcpy(input_blobs_image->mutable_cpu_data(), im_info, sizeof(float) * input_blobs_image->count());
  //   switch(Caffe::mode()){
  //   case Caffe::CPU:
  //   {
  //   //std::cout<<"cpu copy data  "<<endl;

  // //net_->blob_by_name("im_info")->set_cpu_data(im_info);
  //       memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
  //       //memcpy(input_blobs_image->mutable_cpu_data(), im_info, sizeof(float) * input_blobs_image->count());
  //   }
  //       break;
  //   case Caffe::GPU:
  //   {
  //   std::cout<<"gpu copy data  "<<endl;
  //       caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
  //       //caffe_gpu_memcpy(sizeof(float)* input_blobs_image->count(), im_info, input_blobs_image->mutable_gpu_data());
  //   }
  //       break;
  //   default:
  //       LOG(FATAL)<<"Unknow Caffe mode";
  //   }
  net_->blob_by_name("data")->set_cpu_data(data_buf);
    std::cout<<" ForwardFrom  "<<endl;
  net_->ForwardFrom(0);
    std::cout<<" ForwardFrom over "<<endl;
  p_freature = net_->blob_by_name(layername)->cpu_data(); //pool5 1 2048 1 1 fc368
  num = net_->blob_by_name(layername)->channels();//pool5 fc368
  //std::cout<<" pool5 num:  "<<net_->blob_by_name("pool5")->num()<<endl;
    // std::cout<<" pool5 channels:  "<<net_->blob_by_name("pool5")->channels()<<endl;
    // std::cout<<" pool5 height:  "<<net_->blob_by_name("pool5")->height()<<endl;
    // std::cout<<" pool5 width:  "<<net_->blob_by_name("pool5")->width()<<endl;
  // num = net_->blob_by_name("rois")->num();

  // rois = net_->blob_by_name("rois")->cpu_data();
  // pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
  // boxes = new float[num*4];
  // pred = new float[num*5*m_iclass_num];
  //p_freature = new float[num];
  // sorted_pred_cls = new float[num*5];
  //keep = new int[num];

  for (int n = 0; n < num; n++)
  {
    //p_freature[n] = num[n];
    vec_feature.push_back(p_freature[n]);
    // for (int c = 0; c < 4; c++)
    // {
    //   boxes[n*4+c] = rois[n*5+c+1] / img_scale;
    // }
  }



  //delete []p_freature;
    //cv::imwrite("vis.jpg",cv_img);
  //delete []boxes;
  //delete []pred;
  //delete []pred_per_class;
  //delete []keep;
  //delete []sorted_pred_cls;
  delete []data_buf;
}

void Detector::get_feature_blob(cv::Mat & cv_img, Blob<float> & out_blob ,const string layername)
{
    std::cout<<" enter get_feature  "<<endl;
  
  if(cv_img.empty())
    {
        std::cout<<"Can not get the image"<<endl;
        return;
    }
    cv::Mat bgrImg;
    if (cv_img.channels() == 4)
    {
        cv::cvtColor(cv_img, bgrImg, CV_BGRA2BGR);
    } else if (cv_img.channels() == 1)
    {
        cv::cvtColor(cv_img, bgrImg, CV_GRAY2BGR);
    } else
    {
        bgrImg = cv_img;
    }
  /*
    cv::Mat cv_resized;
  int height = 224;//int(cv_img.rows * img_scale);
  int width = 224;//int(cv_img.cols * img_scale);
  cv::resize(bgrImg, cv_resized, cv::Size(width, height));
  cv::Mat cv_new(cv_resized.rows, cv_resized.cols, CV_32FC3, cv::Scalar(0,0,0));
  float im_info[3];
  //float data_buf[height*width*3];
  float* data_buf = new float[height*width*3];
  if(!data_buf)
  {
      std::cout<<"data_buf new error"<<endl;
    return;
  }
  float img_scale = 1.0;
  for (int h = 0; h < cv_resized.rows; ++h )
  {
    for (int w = 0; w < cv_resized.cols; ++w)
    {
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[0] = float(cv_resized.at<cv::Vec3b>(cv::Point(h, w))[0])-float(110.676);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[1] = float(cv_resized.at<cv::Vec3b>(cv::Point(h, w))[1])-float(115.771);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[2] = float(cv_resized.at<cv::Vec3b>(cv::Point(h, w))[2])-float(123.191);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[0])-float(110.676);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.771);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[2])-float(123.191);

    }
    //cout<<endl;
  }
  int num=0;
  const float *p_freature = NULL;
  im_info[0] = cv_new.rows;
  im_info[1] = cv_new.cols;
  im_info[2] = img_scale;
      std::cout<<"im_info:"<<im_info[0]<<im_info[1]<<im_info[2]<<endl;

  for (int h = 0; h < height; ++h )
  {
    for (int w = 0; w < width; ++w)
    {
      data_buf[(0*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[0]);
      data_buf[(1*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[1]);
      data_buf[(2*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[2]);
      // data_buf[(0*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(h, w))[0]);
      // data_buf[(1*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(h, w))[1]);
      // data_buf[(2*height+h)*width+w] = float(cv_new.at<cv::Vec3f>(cv::Point(h, w))[2]);
      //std::cout<<float(cv_new.at<cv::Vec3b>(cv::Point(w, h))[0])<<" ";
      //std::cout<<data_buf[(0*height+h)*width+w]<<" ";
    }
    //cout<<endl;
  }
  //memcpy(data_buf,cv_new.data,sizeof(float)*height*width*3);
  //*/
  // ///============================================================================================
///*
  cv::Mat cv_new(bgrImg.rows, bgrImg.cols, CV_32FC3, cv::Scalar(0,0,0));

     float img_scale = 1;


  int height = 224;//int(cv_img.rows * img_scale);
  int width = 224;//int(cv_img.cols * img_scale);
  int num_out;
  cv::Mat cv_resized;

    //std::cout<<"data_buf"<<endl;
  float im_info[3];
  //float data_buf[height*width*3];
  float* data_buf = new float[height*width*3];
  if(!data_buf)
  {
      std::cout<<"data_buf new error"<<endl;
    return;
  }
    //std::cout<<"data_buf1"<<endl;
  const float *p_freature = NULL;
  // float *pred = NULL;
  // float *pred_per_class = NULL;
  // float *sorted_pred_cls = NULL;
  //int *keep = NULL;
  // const float* bbox_delt;
  // const float* rois;
  // const float* pred_cls;
  int num=0;

  for (int h = 0; h < bgrImg.rows; ++h )
  {
    for (int w = 0; w < bgrImg.cols; ++w)
    {
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[0] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[0])-float(110.676);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[1] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[1])-float(115.771);
      // cv_new.at<cv::Vec3f>(cv::Point(h, w))[2] = float(bgrImg.at<cv::Vec3b>(cv::Point(h, w))[2])-float(123.191);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[0])-float(110.676);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.771);
      cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(bgrImg.at<cv::Vec3b>(cv::Point(w, h))[2])-float(123.191);

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
      // data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[0]);
      // data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[1]);
      // data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(h, w))[2]);
    }
  }
  //*/

  //net_->Reshape();
    //std::cout<<(net_->blob_names()[net_->input_blob_indices()[0]])<<net_->blob_by_name("data")->shape_string()<<endl;
  net_->blob_by_name("data")->Reshape(1, 3, height, width);
  Blob<float> * input_blobs= net_->input_blobs()[0];
  Blob<float> * input_blobs_image= net_->input_blobs()[1];


  //       // memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
  //       //memcpy(input_blobs_image->mutable_cpu_data(), im_info, sizeof(float) * input_blobs_image->count());
  //   switch(Caffe::mode()){
  //   case Caffe::CPU:
  //   {
  //   //std::cout<<"cpu copy data  "<<endl;

  // //net_->blob_by_name("im_info")->set_cpu_data(im_info);
  //       memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
  //       //memcpy(input_blobs_image->mutable_cpu_data(), im_info, sizeof(float) * input_blobs_image->count());
  //   }
  //       break;
  //   case Caffe::GPU:
  //   {
  //   std::cout<<"gpu copy data  "<<endl;
  //       caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
  //       //caffe_gpu_memcpy(sizeof(float)* input_blobs_image->count(), im_info, input_blobs_image->mutable_gpu_data());
  //   }
  //       break;
  //   default:
  //       LOG(FATAL)<<"Unknow Caffe mode";
  //   }
  net_->blob_by_name("data")->set_cpu_data(data_buf);
    std::cout<<" ForwardFrom  "<<endl;
  net_->ForwardFrom(0);
    std::cout<<" ForwardFrom over "<<endl;
    out_blob.CopyFrom(*(net_->blob_by_name(layername)),false,true);
  // p_freature = net_->blob_by_name(layername)->cpu_data(); //pool5 1 2048 1 1 fc368
  // num = net_->blob_by_name(layername)->channels();//pool5 fc368
  // //std::cout<<" pool5 num:  "<<net_->blob_by_name("pool5")->num()<<endl;
  //   // std::cout<<" pool5 channels:  "<<net_->blob_by_name("pool5")->channels()<<endl;
  //   // std::cout<<" pool5 height:  "<<net_->blob_by_name("pool5")->height()<<endl;
  //   // std::cout<<" pool5 width:  "<<net_->blob_by_name("pool5")->width()<<endl;
  // // num = net_->blob_by_name("rois")->num();

  // // rois = net_->blob_by_name("rois")->cpu_data();
  // // pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
  // // boxes = new float[num*4];
  // // pred = new float[num*5*m_iclass_num];
  // //p_freature = new float[num];
  // // sorted_pred_cls = new float[num*5];
  // //keep = new int[num];

  // for (int n = 0; n < num; n++)
  // {
  //   vec_feature.push_back(p_freature[n]);
  // }



  // //delete []p_freature;
  //   //cv::imwrite("vis.jpg",cv_img);
  // //delete []boxes;
  // //delete []pred;
  // //delete []pred_per_class;
  // //delete []keep;
  // //delete []sorted_pred_cls;
  delete []data_buf;
}
void Detector::Detect(cv::Mat & cv_img, vector<Result_detect> & detection_result_class )
{
  const float CONF_THRESH = 0.45;//0.45;
  const int class_others = 100;
  const float CONF_for_others =  0.6;
  const float NMS_THRESH = 0.2;
    const int  max_input_side=1000;
    const int  min_input_side=600;

    //std::cout<<" enter Detect  "<<endl;
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

    //std::cout<<"data_buf"<<endl;
  float im_info[3];
  //float data_buf[height*width*3];
  float* data_buf = new float[height*width*3];
  if(!data_buf)
  {
      std::cout<<"data_buf new error"<<endl;
    return;
  }
    //std::cout<<"data_buf1"<<endl;
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

  //net_->Reshape();
    //std::cout<<(net_->blob_names()[net_->input_blob_indices()[0]])<<net_->blob_by_name("data")->shape_string()<<endl;
  net_->blob_by_name("data")->Reshape(1, 3, height, width);
    //std::cout<<(net_->blob_names()[net_->input_blob_indices()[0]])<<net_->blob_by_name("data")->shape_string()<<endl;
  //net_->blob_by_name("im_info")->Reshape(1, 3,1,1);
  //net_->blob_by_name("im_info")->set_cpu_data(im_info);
    //std::cout<<(net_->blob_names()[net_->input_blob_indices()[1]])<<net_->blob_by_name("im_info")->shape_string()<<endl;
  // net_->blob_by_name("data")->set_cpu_data(data_buf);
  //net_->blob_by_name("data")->set_cpu_data(data_buf);
  Blob<float> * input_blobs= net_->input_blobs()[0];
  Blob<float> * input_blobs_image= net_->input_blobs()[1];


        // memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        memcpy(input_blobs_image->mutable_cpu_data(), im_info, sizeof(float) * input_blobs_image->count());
    switch(Caffe::mode()){
    case Caffe::CPU:
    {
    //std::cout<<"cpu copy data  "<<endl;

  net_->blob_by_name("im_info")->set_cpu_data(im_info);
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        //memcpy(input_blobs_image->mutable_cpu_data(), im_info, sizeof(float) * input_blobs_image->count());
    }
        break;
    case Caffe::GPU:
    {
    std::cout<<"gpu copy data  "<<endl;
        caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
        //caffe_gpu_memcpy(sizeof(float)* input_blobs_image->count(), im_info, input_blobs_image->mutable_gpu_data());
    }
        break;
    default:
        LOG(FATAL)<<"Unknow Caffe mode";
    }
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
    
    //cpu_nms(sorted_pred_cls, num, 5, NMS_THRESH,( i == class_others ? CONF_for_others : CONF_THRESH ), detection_result_class,i);
    switch (Caffe::mode()) 
    {
      case Caffe::CPU:
      {
    //std::cout<<"cpu nms  "<<endl;
      cpu_nms(sorted_pred_cls, num, 5, NMS_THRESH,( i == class_others ? CONF_for_others : CONF_THRESH ), detection_result_class,i);
      }
      break;
      case Caffe::GPU:
      {
    //std::cout<<"gpu nms "<<endl;
      _nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, m_gpuid>=0 ? m_gpuid: 0);
      int k=0;
      while(k < num_out  && sorted_pred_cls[keep[k]*5+4]>( i == class_others ? CONF_for_others : CONF_THRESH ) )
      {
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
        obj_Result_detect.bdelete = false;
        //float tmpscale = 1000.0/float(cv_img.rows);
        obj_Result_detect.area = float(obj_Result_detect.y2-obj_Result_detect.y1)*float(obj_Result_detect.x2-obj_Result_detect.x1);
        // if((i-1)>=0)
        // {
        //   if(obj_Result_detect.area > 4*sumarray[i])
        //   {
        //     k++;
        //     continue;
        //   }
        // }
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


    //cv::imwrite("vis.jpg",cv_img);
  delete []boxes;
  delete []pred;
  delete []pred_per_class;
  delete []keep;
  delete []sorted_pred_cls;
  delete []data_buf;
    //std::cout<<" over  "<<endl;

  //DelDetect_after_cut_detect(detection_result_class);
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
void Detector::Detect(cv::Mat & cv_img, vector<cv::Rect> & detection_result )
{
  float CONF_THRESH = 0.3;
  float NMS_THRESH = 0.3;
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
