#include "caffe_detect.hpp"
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <sys/time.h>
#include <assert.h>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
//#include "google/protobuf/text_format.h"
using namespace cv;

DEFINE_string(model_file, "", "model prototxt file");
DEFINE_string(weights_file, "", "model weight file");
DEFINE_string(strJpegDir, "", "JPEG dir");
DEFINE_string(saveJpegDir, "", "JPEG result dir");
DEFINE_string(out_file, "", "out txt file");
DEFINE_string(list_file, "", "test file list");
DEFINE_bool(single_multi, false, "single or multi");//
DEFINE_bool(cutornot, false, "is 4 cut");//
DEFINE_int32(GPUID, 0, "gpu id");
DEFINE_double(NMS_threhold, 0.3, "NMS threhold");
DEFINE_double(CONF_threhold, 0.3, "conf threhold");
DEFINE_string(libdir, "/home/liushuai/tiannuocaffe/py-rfcn-gpu//lib/", "py/lib/ dir");
// DEFINE_string(file_type, "image",
//     "The file type in the list_file. Currently support image and video.");

int GetInitClass(const string& model_file)
{
 
   ifstream fin;
   fin.open(model_file.c_str());
   if(!fin)
   {
    cout<<"open file fail"<<endl;
    return 0;
   }
 string str;
 string output = "output_dim";
 while (!fin.eof())
 {
  getline(fin, str);

  int idx=str.find(output);
  if(idx!=string::npos)
  {
   int index=str.find(output);
   index = index+output.length();
   string str1=str.substr(index);
   
   index=str1.find(":")+1;
   string str2=str1.substr(index);
            

   index=str2.find("#");
   string str3=str2.assign(str2.c_str(),index);
   str3.erase(0,str3.find_first_not_of(" "));
   str3.erase(str3.find_last_not_of(" ")+1);
   
   int numclass = atoi(str3.c_str());
            fin.close();
   return numclass;

  }
  
 }
 fin.close();
}

void singledetect(int GPUID = 0)
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    int imageCount = 1;
    string model_file = FLAGS_model_file;
    string weights_file = FLAGS_weights_file;//"/storage/liushuai/RFCN/py-R-FCN-master/output/rfcn_end2end_ohem/cookieproj1_trainval/resnet50_cookie_rfcn_ohem_iter_100000.caffemodel";
    GPUID= FLAGS_GPUID;
    int max_ret_num=30;
    string lib_dir = FLAGS_libdir;
    int classnum = GetInitClass(model_file);
    float nms_threhold= FLAGS_NMS_threhold;
    float conf_threhold= FLAGS_CONF_threhold;
    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle,lib_dir.c_str());
    // vector<cv::Rect> detection_result;
    cv::Mat img = cv::imread(FLAGS_strJpegDir);//"/storage/liushuai/RFCN/R_fcn_bin/16/MDLZ05161001003.jpg");
    // handle->Detect(inputimage, detection_result);
    // for(int i=0;i < detection_result.size(); i++){
    //     cv::rectangle(inputimage,cv::Point(detection_result[i].x,detection_result[i].y),
    //                              cv::Point(detection_result[i].x + detection_result[i].width,detection_result[i].y + detection_result[i].height),
    //                              cv::Scalar(0,255,0));

    // }
    // cv::imwrite("MDLZ05161001003.jpg",inputimage);

              vector<Result_detect>  detection_result_class;
              if(FLAGS_cutornot)
              {
              	handle->Cut4ImageDetect(img, detection_result_class,nms_threhold,conf_threhold);
              }
              else
              {
              	handle->Detect(img, detection_result_class,nms_threhold,conf_threhold);
              }
              //handle->Detect(img, detection_result_class);
                for(int i=0;i < detection_result_class.size(); i++)
                {

                    cv::rectangle(img,cv::Point(detection_result_class[i].x1,detection_result_class[i].y1),
                                             cv::Point(detection_result_class[i].x2,detection_result_class[i].y2),
                                             cv::Scalar(0,0,255),3);
                    char tmp_label[256] ={0};
                    //sprintf(tmp_label,"%d",detection_result_class[i].iclass);
                    //std::cout<<detection_result_class[i].x2<<" "<<detection_result_class[i].x1<<std::endl;
                    sprintf(tmp_label,"%d,%d",detection_result_class[i].iclass,static_cast<int>(detection_result_class[i].x2-detection_result_class[i].x1));
                    if (detection_result_class[i].bcalibrate==true)
                    {
                    sprintf(tmp_label,"%d<-%d,%d",detection_result_class[i].iclass,detection_result_class[i].calibrate_preclass,static_cast<int>(detection_result_class[i].x2-detection_result_class[i].x1));
                    }
                    int i_font = 1;
                    // if(img.rows<600)
                    //     i_font = 1;
                    // else if(600<=img.rows && img.rows<1200) 
                    //     i_font = 2;
                    // else if(1200<=img.rows && img.rows<1800) 
                    //     i_font = 3;
                    // else if(1800<=img.rows && img.rows<2400)
                    //     i_font = 4;
                    // else
                    //     i_font = 5;
                    putText(img, tmp_label , Point(static_cast<int>(detection_result_class[i].x1), static_cast<int>(detection_result_class[i].y1)), \
                      CV_FONT_HERSHEY_COMPLEX, i_font, Scalar(0, 0, 255));
                }
                cv::imwrite(FLAGS_out_file,img);//"MDLZ05161001003.jpg",img);
    gettimeofday(&end,NULL);
    std::cout << "Total Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<imageCount<< endl;
}

int multi_det();

void *print_msg(void *ptr)
{
  std::cout << "print_msg "<<endl;
  //singledetect();
    cv::Mat img = cv::imread("/storage/dataset/tiannuo_models/0912_nestlecoffee4.0/nco00295.jpg");
    // handle->Detect(inputimage, detection_result);
    // for(int i=0;i < detection_result.size(); i++){
    //     cv::rectangle(inputimage,cv::Point(detection_result[i].x,detection_result[i].y),
    //                              cv::Point(detection_result[i].x + detection_result[i].width,detection_result[i].y + detection_result[i].height),
    //                              cv::Scalar(0,255,0));

    // }
    // cv::imwrite("MDLZ05161001003.jpg",inputimage);

              vector<Result_detect>  detection_result_class;
              ((Detector *)ptr)->Detect(img, detection_result_class);
  std::cout << "detect over "<<endl;
              return 0;
}

void *print_msg1(void *ptr)
{
  std::cout << "print_msg "<<endl;
  singledetect(1);
}

int main(int argc, char** argv)
{
    gflags::SetUsageMessage("for test\n"
        "format used as input for test caffe.\n"
        "Usage:\n"
        "    test\n");
    caffe::GlobalInit(&argc, &argv);

    // 输入参数不足时报错
    // if (argc < 4)
    // {
    //     gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    //     return 1;
    // }
	if(FLAGS_single_multi)
	{
		singledetect();
	}
	else
	{
  		multi_det();
	}
    // 
    return 0;
    string model_file = FLAGS_model_file;//"/storage/liushuai/RFCN/py-R-FCN-master/models/cookie/ResNet-50/rfcn_end2end/test_agnostic.prototxt";
    string weights_file = FLAGS_weights_file;//"/storage/liushuai/RFCN/py-R-FCN-multiGPU/output/rfcn_end2end_ohem/cookieproj1_trainval/resnet50_cookie_rfcn_ohem_iter_20.caffemodel";
    int GPUID=FLAGS_GPUID;
    int max_ret_num=30;
    int classnum = GetInitClass(model_file);;


    Detector * handle = NULL;
    Detector * handle1 = NULL;
    Detector * handle2 = NULL;
    Detector * handle3 = NULL;
    string lib_dir = FLAGS_libdir;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, 1 , handle,lib_dir.c_str());
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, 2, handle1,lib_dir.c_str());
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, 3 , handle2,lib_dir.c_str());
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, 4 , handle3,lib_dir.c_str());
  
  pthread_t thread1, thread2;
  pthread_t thread3;
  pthread_t thread4;
  pthread_create(&thread1,NULL, &print_msg, (void *)handle);
  pthread_create(&thread2,NULL, &print_msg, (void *)handle1);
  pthread_create(&thread1,NULL, &print_msg, (void *)handle2);
  pthread_create(&thread2,NULL, &print_msg, (void *)handle3);
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  pthread_join(thread3, NULL);
  pthread_join(thread4, NULL);
}


int multi_det()
{
    // singledetect();
    // return 0;
    //const std::string& file_type = FLAGS_file_type;
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/models/nestlecoffee/ResNet-101/rfcn_end2end/s16_14/b14_test_16_s_4_8_16_32_agnostic.prototxt";
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/output/rfcn_end2end_ohem/nestlecoffeeproj1_trainval/101_b14_16_s_4_8_16_32resnet50_nestlecoffee87_rfcn_ohem_iter_80000.caffemodel";
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/models/nestlecoffee/ResNet-50/rfcn_end2end/test_agnostic.prototxt";
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/output/rfcn_end2end_ohem/nestlecoffeeproj2_trainval/nestlecoffee87_resnet50_rfcn_ohem_iter_100000.caffemodel";
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/models/nestlericeflour/ResNet-101/rfcn_end2end/s16_14/b14_test_16_s_4_8_16_32_agnostic.prototxt";
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/output/rfcn_end2end_ohem/nestlericeflourproj1_trainval/101_b14_16_s_4_8_16_32_nestlericeflour50_rfcn_ohem_iter_95000.caffemodel";

    // std::string model_file = "/storage/dataset/tiannuo_models/0828_beer9.0/beer9.0.prototxt";
    // std::string weights_file = "/storage/dataset/tiannuo_models/0828_beer9.0/beer9.0.caffemodel";
    //
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/models/nestlebiscuit/ResNet-101/rfcn_end2end/s16_14/b14_test_16_s_4_8_16_32_agnostic.prototxt";
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/output/rfcn_end2end_ohem/nestlebiscuitproj1_trainval/101_b14_16_s_4_8_16_32_nestlebiscuit62_rfcn_ohem_iter_100000.caffemodel";
    
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/models/nestleoatmeal/ResNet-101/rfcn_end2end/s16_14/b14_test_16_s_4_8_16_32_agnostic.prototxt";
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/output/rfcn_end2end_ohem/nestleoatmealproj1_trainval/101_b14_16_s_4_8_16_32_nestleoatmeal36_rfcn_ohem_iter_100000.caffemodel";
        
    std::string model_file = FLAGS_model_file;//"/storage/dataset/tiannuo_models/0904_nestlemilk3.0//nestlemilk3.0.prototxt";
    std::string weights_file = FLAGS_weights_file;//"/storage/dataset/tiannuo_models/0904_nestlemilk3.0//nestlemilk3.0.caffemodel";
    int GPUID=FLAGS_GPUID;
    int max_ret_num=30;
    string lib_dir = FLAGS_libdir;
    float nms_threhold= FLAGS_NMS_threhold;
    float conf_threhold = FLAGS_CONF_threhold;
    int classnum = GetInitClass(model_file);//coffee 87  oatmeal 37 mifen 50 binggan 62 liannai 13 beer134

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle,lib_dir.c_str());

    // std::string strJpegDir = "/storage/liushuai/data/nestlericeflour/nestlericeflourproj1/JPEGImages/";
    // std::string saveJpegDir = "/storage/liushuai/data/nestlericeflour/nestlericeflourproj1/result/";
    // std::string out_file  = "101_delLOGO_nestlericeflour.txt"; //FLAGS_out_file;
    // const char * list_file = "/storage/liushuai/data/nestlericeflour/nestlericeflourproj1//ImageSets/Main/test.txt";

    std::string strJpegDir = FLAGS_strJpegDir;//"/home/liushuai/download/kuang_liannai//JPEG/";
    std::string saveJpegDir = FLAGS_saveJpegDir;//"/home/liushuai/download/kuang_liannai//result/";
    std::string out_file  = FLAGS_out_file;//"/home/liushuai/download/kuang_liannai//101_delLOGO_kuang_liannai.txt"; //FLAGS_out_file;
    const char * list_file = FLAGS_list_file.c_str();//"/home/liushuai/download/kuang_liannai//test.txt";

    // std::string strJpegDir = "/storage/liushuai/data/nestlemilk/nestlemilkproj1/JPEGImages/";
    // std::string saveJpegDir = "/storage/liushuai/data/nestlemilk/nestlemilkproj1/result/";
    // std::string out_file  = "cut_LOGO_nestlemilk.txt"; //FLAGS_out_file;
    //const char * list_file = "/storage/liushuai/data/nestlemilk/nestlemilkproj1//ImageSets/Main/test.txt";
    // std::string strJpegDir = "/storage/liushuai/data/nestleoatmeal/nestleoatmealproj1/JPEGImages/";
    // std::string saveJpegDir = "/storage/liushuai/data/nestleoatmeal/nestleoatmealproj1/result/";
    // std::string out_file  = "101del_LOGO_nestleoatmeal.txt"; //FLAGS_out_file;
    // const char * list_file = "/storage/liushuai/data/nestleoatmeal/nestleoatmealproj1//ImageSets/Main/test.txt";
    // std::string strJpegDir = "/storage/liushuai/data/nestlebiscuit/nestlebiscuitproj1//JPEGImages/";
    // std::string saveJpegDir = "/storage/liushuai/data/nestlebiscuit/nestlebiscuitproj1/result/";
    // std::string out_file  = "101del_LOGO_nestlebiscuit.txt"; //FLAGS_out_file;
    // const char * list_file = "/storage/liushuai/data/nestlebiscuit/nestlebiscuitproj1/ImageSets/Main/test.txt";
    // std::string strJpegDir = "/storage/liushuai/data/nestlecoffee/nestlecoffeeproj1//JPEGImages/";
    // std::string saveJpegDir = "/storage/liushuai/data/nestlecoffee/nestlecoffeeproj1/result/";
    // std::string out_file  = "101del_LOGO_coffe.txt"; //FLAGS_out_file;
    // const char * list_file = "/storage/liushuai/data/nestlecoffee/nestlecoffeeproj1/ImageSets/Main/test.txt";
    // std::string strJpegDir = "//storage/dataset/tiannuo_data/test_beer//";
    // std::string saveJpegDir = "//storage/dataset/tiannuo_data//beer_result/";
    // std::string out_file  = "beer_result.txt"; //FLAGS_out_file;
    // const char * list_file = "//storage/dataset/tiannuo_data/test_beer//test.txt";

    // std::string strJpegDir = "/storage/liushuai/data/cookie/cookieproj1/JPEGImages/";
    // std::string saveJpegDir = "/nas/public/liushuai/test/16_4_8_32/";
    // std::string out_file  = "post_16_s_4_8_16_32_gpu_0.45conf_0.2gpunms_others0.75conf_0.7RPN_nms.txt"; //FLAGS_out_file;
    // const char * list_file = "/storage/liushuai/RFCN/R_fcn_bin/262_test_supermarket.txt";

    // std::string strJpegDir = "/mnt/storage/liushuai/data/chutty/chuttyproj1/JPEGImages_test_cut/";
    // std::string saveJpegDir = "/nas/public/liushuai/test/result_picture/chutty_test/";
    // std::string out_file  = "chutty_gpu82_0.45conf_0.2gpunms_others0.6conf_0.7RPN_nms.txt"; //FLAGS_out_file;
    // const char * list_file = "/mnt/storage/liushuai/data/chutty/chuttyproj1/JPEGImages_test_cut/test.txt";
    // std::string strJpegDir = "/mnt/storage/liushuai/RFCN/R_fcn_bin/16/";//"/mnt/storage/liushuai/RFCN/R_fcn_bin/16/";
    // std::string saveJpegDir = "./J_test/";
    // std::string out_file  = "cpu_0.45conf_0.2nms_others0.6conf_0.7RPN_nms.txt"; //FLAGS_out_file;
    // const char * list_file = "/mnt/storage/liushuai/RFCN/R_fcn_bin/16/test.txt";

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty()) 
    {
        outfile.open(out_file.c_str());
        if (outfile.good()) 
        {
            buf = outfile.rdbuf();
        }

    }
    else
        return 0;

    std::ostream out(buf); 
    // Process image one by one.
    std::ifstream infile(list_file);
    std::string file;
    int imageCount = 0;
    //double time_use = 0;
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);

    while (infile >> file) 
    {
        //if (file_type == "image") 
        {
            //      string fileName = "JPEGImages/" + file + ".jpg";
              std::string fileName = strJpegDir + file + ".jpg";
             std::cout<<file<<std::endl; 
              cv::Mat img = cv::imread(fileName, -1);
              CHECK(!img.empty()) << "Unable to decode image " << file;
              imageCount ++;
              // img.copyTo(show);
              // std::vector<vector<float> > detections = detector.Detect(img);
              
              // dealDifClass(detections, img.cols, img.rows, confidence_threshold);
              
                //vector<cv::Rect> detection_result;
                //handle->Detect(img, detection_result);
                // for(int i=0;i < detection_result.size(); i++){

                //     cv::rectangle(img,cv::Point(detection_result[i].x,detection_result[i].y),
                //                              cv::Point(detection_result[i].x + detection_result[i].width,detection_result[i].y + detection_result[i].height),
                //                              cv::Scalar(0,255,0));

                // }
                // cv::imwrite(saveJpegDir+ + file +".jpg",img);
              vector<Result_detect>  detection_result_class;
              if(FLAGS_cutornot)
              {
              	handle->Cut4ImageDetect(img, detection_result_class,nms_threhold,conf_threhold);
              }
              else
              {
              	handle->Detect(img, detection_result_class,nms_threhold,conf_threhold);
              }
                for(int i=0;i < detection_result_class.size(); i++)
                {

                    cv::rectangle(img,cv::Point(detection_result_class[i].x1,detection_result_class[i].y1),
                                             cv::Point(detection_result_class[i].x2,detection_result_class[i].y2),
                                             cv::Scalar(0,0,255),3);
                    char tmp_label[256] ={0};
                    sprintf(tmp_label,"%d,%d",detection_result_class[i].iclass,detection_result_class[i].x2-detection_result_class[i].x1);
                    int i_font = 1;
                    if(img.rows<600)
                        i_font = 1;
                    else if(600<=img.rows && img.rows<1200) 
                        i_font = 2;
                    else if(1200<=img.rows && img.rows<1800) 
                        i_font = 3;
                    else if(1800<=img.rows && img.rows<2400)
                        i_font = 4;
                    else
                        i_font = 5;
                    
                    putText(img, tmp_label , Point(static_cast<int>(detection_result_class[i].x1), static_cast<int>(detection_result_class[i].y1)), \
                      CV_FONT_HERSHEY_COMPLEX, i_font, Scalar(0, 0, 255),3);
                    // char tmp_label[256] ={0};
                    // sprintf(tmp_label,"%d",detection_result_class[i].iclass);
                    // putText(img, tmp_label , Point(static_cast<int>(detection_result_class[i].x1), static_cast<int>(detection_result_class[i].y1)), \
                    //   CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
                    out << file << " ";
                    out << static_cast<int>(detection_result_class[i].iclass) << " ";
                    out << img.cols << " ";
                    out << img.rows << " ";
                    out << static_cast<int>(detection_result_class[i].x1) << " ";
                    out << static_cast<int>(detection_result_class[i].y1) << " ";
                    out << static_cast<int>(detection_result_class[i].x2) << " ";
                    out << static_cast<int>(detection_result_class[i].y2) << std::endl;

                }
                std::string savfilename=saveJpegDir + file +".jpg";
                cv::imwrite(savfilename,img);
        }
    }
    //count train time
    gettimeofday(&end,NULL);
    std::cout << "Total Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<imageCount<< endl;
    return 0;
}
