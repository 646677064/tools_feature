#ifndef CAFFE_DETECT_HPP
#define CAFFE_DETECT_HPP
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
using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))


#define		EV2641_ERR_SUCCESS 0
//矩形定义
typedef struct {
	int x;
	int y;
	int w;
	int h;
}EV2641Rect;

typedef struct
{
	unsigned char * imagedata;			//图像矩阵指针
	int width;							//图像宽度
	int height;							//图像高度
	int widthStep;						//每行像素的字节数
	int type;							//图像类型
	char useROI;						//感兴趣区域表示
	EV2641Rect mROI;					//感兴趣区域
}EV2641Image;

typedef struct Result_detect{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int iclass;
	bool bcalibrate;
	int calibrate_preclass;
	Result_detect(){bcalibrate=false;calibrate_preclass=-1;};
}Result_detect;


class Detector;

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  EV2641_InitCarDetector
 *  Description:  Load the model file and weights file ,set GPUID
 * =====================================================================================
 */
int EV2641_InitCarDetector(const char * model_file,const  char * weights_file, int classnum,const int GPUID ,Detector * &handle,const char * lib_dir);
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  EV2641_ReleaseCarDetector
 *  Description:  Release required resource
 * =====================================================================================
 */
int EV2641_ReleaseCarDetector();
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detector
 *  Description:  Detect Car and return detection result
 * =====================================================================================
 */

int EV2641_A_GetCarRect(const EV2641Image * image, int &max_ret_num, EV2641Rect * rect, Detector * &handle);

//background and car
//const int class_num=101;// #lius
const bool  bagnostic = true;

/*
 * ===  Class  ======================================================================
 *         Name:  Detector
 *  Description:  FasterRCNN CXX Detector
 * =====================================================================================
 */
class Detector {
public:
	Detector(const string& model_file, const string& weights_file, int classnum,const int GPUID);
	void Detect(cv::Mat & cv_image, vector<cv::Rect> & detection_result_rect ,float nms_threhold=0.3,float conf_threhold=0.3);
	void Detect(cv::Mat & cv_img, vector<Result_detect> & detection_result_class ,float nms_threhold=0.3,float conf_threhold=0.3);
	void Cut9ImageDetect(cv::Mat& img, vector<Result_detect> & detection_result_class,float nms_threhold=0.3,float conf_threhold=0.3);
	void Cut4ImageDetect(cv::Mat& img, vector<Result_detect> & detection_result_class,float nms_threhold=0.3,float conf_threhold=0.3);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void boxes_sort(int num, const float* pred, float* sorted_pred,float fTHRESH,int& num_THRESHout);
	void cpu_nms(const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, float confidence_threshold,vector<Result_detect> & detection_result_class ,int iclass);

private:
	int m_gpuid;
	shared_ptr<Net<float> > net_;
	int m_iclass_num;
	Detector(){}
};

//Using for box sort
struct Info
{
	float score;
	const float* head;
};
bool compare(const Info& Info1, const Info& Info2)
{
	return Info1.score > Info2.score;
}

bool comparescore(const Result_detect& Info1, const Result_detect& Info2)
{
	return Info1.score > Info2.score;
}

typedef struct Min_distace_index_{
  // float x1;
  // float y1;
  // float x2;
  // float y2;
	bool bconfuse;
	int confuse_index;
   float lenght;
   float height;
  float distace;
  int index;
  Min_distace_index_(){confuse_index=-1;bconfuse=false;};
}Min_distace_index_;

bool comparedistance(const Min_distace_index_& Info1, const Min_distace_index_& Info2)
{
	return Info1.distace < Info2.distace;
}
#endif
