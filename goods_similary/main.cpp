#include "caffe_detect.hpp"
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <fstream>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
extern "C" {
#include <cblas.h>
}
#ifdef _OPENMP
#include <omp.h>
#endif
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
using namespace cv;


// DEFINE_string(file_type, "image",
//     "The file type in the list_file. Currently support image and video.");

void writeMatToFile(cv::Mat& m, const char* filename) {
//     ofstream fout(filename);
//     if(!fout) {
//         cout<<"File Not Opened"<<endl;  return;
//     }
//     fout << m;
//     fout.close();
    ofstream fout(filename);
    if(!fout) {
        cout<<"File Not Opened"<<endl;  return;
    }
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            for (int k = 0; k < m.channels(); ++k) {
                fout << m.at<float>(i, j * m.channels() + k);
                if (j * m.channels() + k < m.cols * m.channels() - 1) {
                    fout << ", ";
                }
            }
        }
        if (i < m.rows - 1) fout << "; " << endl;
    }
    fout.close();
}
/*----------------------------
 * 功能 : 将 cv::Mat 数据写入到 .txt 文件
 *----------------------------
 * 函数 : WriteData
 * 访问 : public 
 * 返回 : -1：打开文件失败；0：写入数据成功；1：矩阵为空
 *
 * 参数 : fileName    [in]    文件名
 * 参数 : matData [in]    矩阵数据
 */
int WriteData(cv::Mat& matData,string fileName ,int size)
{
    int retVal = 0;

    // 检查矩阵是否为空
    if (matData.empty())
    {
        cout << "矩阵为空" << endl; 
        retVal = 1;
        return (retVal);
    }

    // 打开文件
    ofstream outFile(fileName.c_str(), std::ios::binary | ios_base::out);   //按新建或覆盖方式写入
    if (!outFile.is_open())
    {
        cout << "打开文件失败" << endl; 
        retVal = -1;
        outFile.close();
        return (retVal);
    }

    // 写入数据
    // for (int r = 0; r < matData.rows; r++)
    // {
    //  for (int c = 0; c < matData.cols; c++)
    //  {
    //      float data = matData.at<float>(r,c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式
    //      outFile << data;    //每列数据用 tab 隔开
    //  }
    //  //outFile << endl;  //换行
    // }
    outFile.write((const char*)matData.data, sizeof(float) * size);
    outFile.close();

    return (retVal);
}


/*----------------------------
 * 功能 : 从 .txt 文件中读入数据，保存到 cv::Mat 矩阵
 *      - 默认按 float 格式读入数据，
 *      - 如果没有指定矩阵的行、列和通道数，则输出的矩阵是单通道、N 行 1 列的
 *----------------------------
 * 函数 : LoadData
 * 访问 : public 
 * 返回 : -1：打开文件失败；0：按设定的矩阵参数读取数据成功；1：按默认的矩阵参数读取数据
 *
 * 参数 : fileName    [in]    文件名
 * 参数 : matData [out]   矩阵数据
 * 参数 : matRows [in]    矩阵行数，默认为 0
 * 参数 : matCols [in]    矩阵列数，默认为 0
 * 参数 : matChns [in]    矩阵通道数，默认为 0
 */
int LoadData(string fileName,vector<float>&  out_vec,/* cv::Mat& matData,*/int matCols = 2048, int matRows = 1,  int matChns = 1)
{
    int retVal = 0;

    // 打开文件
    ifstream inFile(fileName.c_str(), std::ios::binary | ios_base::in);
    if(!inFile.is_open())
    {
        cout << "读取文件失败" << endl;
        retVal = -1;
        inFile.close();
        return (retVal);
    }
    //float tmp_feature[2048]={0.0};
    float* tmp_feature=(float*)new float[matCols];
    if (!tmp_feature)
    {
        cout << "not enought memory" << endl;
        retVal = -1;
        inFile.close();
        return (retVal);
        /* code */
    }
    inFile.read((char*)tmp_feature, sizeof(float) * matCols);
    out_vec.clear();
    for (int i = 0; i < matCols; ++i)
    {
        out_vec.push_back(tmp_feature[i]);
    }
    delete tmp_feature;
    tmp_feature=NULL;
    size_t dataLength =out_vec.size();

    // // 载入数据
    // istream_iterator<float> begin(inFile);   //按 float 格式取文件数据流的起始指针
    // istream_iterator<float> end;         //取文件流的终止位置
    // vector<float> inData(begin,end);     //将文件数据保存至 std::vector 中
    // //out_vec.assign(begin, end);//将v2赋值给v1
    // cv::Mat tmpMat = cv::Mat(inData);        //将数据由 std::vector 转换为 cv::Mat
    // inFile.close();
    // cout <<out_vec.size()<<" "<<*begin<<endl;
    // copy(begin,end,back_inserter(inData));

    // 输出到命令行窗口
    //copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t")); 

    // // 检查设定的矩阵尺寸和通道数
    // size_t dataLength = inData.size();
    // //1.通道数
    // if (matChns == 0)
    // {
    //  matChns = 1;
    // }
    // //2.行列数
    // if (matRows != 0 && matCols == 0)
    // {
    //  matCols = dataLength / matChns / matRows;
    // } 
    // else if (matCols != 0 && matRows == 0)
    // {
    //  matRows = dataLength / matChns / matCols;
    // }
    // else if (matCols == 0 && matRows == 0)
    // {
    //  matRows = dataLength / matChns;
    //  matCols = 1;
    // }
    //3.数据总长度
    if (dataLength != (matRows * matCols * matChns))
    {
        cout << "读入的数据长度 不满足 设定的矩阵尺寸与通道数要求，将按默认方式输出矩阵！" <<dataLength<< endl;
        retVal = 1;
        matChns = 1;
        matRows = dataLength;
    } 

    // 将文件数据保存至输出矩阵
    //matData = tmpMat.reshape(matChns, matRows).clone();
    
    return (retVal);
}

#define DEFINE_VSL_BINARY_FUNC_2(name, operation) \
template<typename Dtype> \
void v_2##name(const int n, const vector<Dtype>& a, const vector<Dtype>& b, vector<Dtype>& y) { \
CHECK_GT(n, 0); \
for (int i = 0; i < n; ++i) { operation; } \
} \
inline void vs_2##name( \
const int n, const vector<float>& a, const vector<float>& b, vector<float>& y) { \
v_2##name<float>(n, a, b, y); \
} \
inline void vd_2##name( \
  const int n, const vector<double>& a, const vector<double>& b, vector<double>& y) { \
v_2##name<double>(n, a, b, y); \
  }
DEFINE_VSL_BINARY_FUNC_2(Sub, y[i] = a[i] - b[i]);

float compute_similary(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size = vec_feature1.size();
    float multiply=0.0;
    float square1=0.0;
    float square2=0.0;
    float tmp=0.0;
    float dot=0.0;
    //std::cout << "size:"<<size<<endl;
    for(int i =0;i<size;i++)
    {
        // multiply+=vec_feature1[i]*vec_feature2[i];
        // square1+=vec_feature1[i]*vec_feature1[i];
        // square2+=vec_feature2[i]*vec_feature2[i];
        tmp=vec_feature1[i]-vec_feature2[i];
        dot=dot+tmp*tmp;

    }
    //float dot =square1+square2-2*multiply;
    // float sqrt1 = sqrt(square1);
    // float sqrt2 = sqrt(square2);
    // float ret = (square1+square2-2*multiply)/(sqrt1*sqrt2);
    // std::cout << "similary:"<<ret<<endl;
    // return ret;

    //std::cout << "euclidean cpu:"<<dot<<endl;
    return dot;
}
float compute_similary_MP(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size = vec_feature1.size();
    float multiply=0.0;
    float square1=0.0;
    float square2=0.0;
    //std::cout << "size:"<<size<<endl;
    // vector<float> mp_vec;
    // mp_vec.reserve(size);
    float dot = 0.0;
    float tmp_f=0.0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for private(tmp_f) reduction(+:dot) 
        for(int i =0;i<size;i++)
        {
            //mp_vec[i]=
            //dot = dot+vec_feature2[i]*vec_feature2[i]+vec_feature1[i]*vec_feature1[i]-2*vec_feature1[i]*vec_feature2[i];
            tmp_f = vec_feature1[i]-vec_feature2[i];
            //dot=cblas_sdot(1, &tmp_f, 1, &tmp_f, 1);
            dot=dot+tmp_f*tmp_f;
            //dot = dot+vec_feature2[i]*vec_feature2[i]+vec_feature1[i]*vec_feature1[i]-2*vec_feature1[i]*vec_feature2[i];
        }
    }
// #ifdef _OPENMP
//     #pragma omp parallel
//     {
//         #pragma omp for reduction(+:dot)
//         for(int i =0;i<size/4;i++)
//         {
//             //mp_vec[i]=
//             dot = dot+vec_feature2[i]*vec_feature2[i]+vec_feature1[i]*vec_feature1[i]-2*vec_feature1[i]*vec_feature2[i] \
//             +dot+vec_feature2[i+size/4]*vec_feature2[i+size/4]+vec_feature1[i+size/4]*vec_feature1[i+size/4]-2*vec_feature1[i+size/4]*vec_feature2[i+size/4] \
//             +dot+vec_feature2[i+size*2/4]*vec_feature2[i+size*2/4]+vec_feature1[i+size*2/4]*vec_feature1[i+size*2/4]-2*vec_feature1[i+size*2/4]*vec_feature2[i+size*2/4] \
//             +dot+vec_feature2[i+size*3/4]*vec_feature2[i+size*3/4]+vec_feature1[i+size*3/4]*vec_feature1[i+size*3/4]-2*vec_feature1[i+size*3/4]*vec_feature2[i+size*3/4];
//         }
//     }
#endif
    // float sqrt1 = sqrt(square1);
    // float sqrt2 = sqrt(square2);
    // float ret = (square1+square2-2*multiply)/(sqrt1*sqrt2);
    // std::cout << "similary:"<<ret<<endl;
    // return ret;

    //float dot =square1+square2-2*multiply;
    //std::cout << "euclidean cpu:"<<dot<<endl;
    return dot;
}

float compute_similary_cos(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size_n = vec_feature1.size();
    float multiply=0.0;
    float square1=0.0;
    float square2=0.0;
    float tmp=0.0;
    float dot=0.0; 
    //std::cout << "size:"<<size<<endl;
    // float Molecular =cblas_sdot(size_n, vec_feature1.data(), 1, vec_feature2.data(), 1);
    // float denominator_1=cblas_sdot(size_n, vec_feature1.data(), 1, vec_feature1.data(), 1);
    // float denominator_2=cblas_sdot(size_n, vec_feature2.data(), 1, vec_feature2.data(), 1);
    // dot=Molecular/sqrt(denominator_1)/sqrt(denominator_2);
    for(int i =0;i<size_n;i++)
    {
        multiply+=vec_feature1[i]*vec_feature2[i];
        square1+=vec_feature1[i]*vec_feature1[i];
        square2+=vec_feature2[i]*vec_feature2[i];

    }
    dot=multiply/sqrt(square1)/sqrt(square2);
    //float dot =square1+square2-2*multiply;
    // float sqrt1 = sqrt(square1);
    // float sqrt2 = sqrt(square2);
    // float ret = (square1+square2-2*multiply)/(sqrt1*sqrt2);
    // std::cout << "similary:"<<ret<<endl;
    // return ret;

    //std::cout << "euclidean cpu:"<<dot<<endl;
    return dot;
}
float compute_similary_cos_blas(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size_n = vec_feature1.size();
    float multiply=0.0;
    float square1=0.0;
    float square2=0.0;
    float tmp=0.0;
    float dot=0.0; 
    //std::cout << "size:"<<size<<endl;
    float Molecular =cblas_sdot(size_n, vec_feature1.data(), 1, vec_feature2.data(), 1);
    float denominator_1=cblas_sdot(size_n, vec_feature1.data(), 1, vec_feature1.data(), 1);
    float denominator_2=cblas_sdot(size_n, vec_feature2.data(), 1, vec_feature2.data(), 1);
    dot=Molecular/sqrt(denominator_1)/sqrt(denominator_2);
    // for(int i =0;i<size;i++)
    // {
    //     // multiply+=vec_feature1[i]*vec_feature2[i];
    //     // square1+=vec_feature1[i]*vec_feature1[i];
    //     // square2+=vec_feature2[i]*vec_feature2[i];
    //     tmp=vec_feature1[i]-vec_feature2[i];
    //     dot=dot+tmp*tmp;

    // }
    //float dot =square1+square2-2*multiply;
    // float sqrt1 = sqrt(square1);
    // float sqrt2 = sqrt(square2);
    // float ret = (square1+square2-2*multiply)/(sqrt1*sqrt2);
    // std::cout << "similary:"<<ret<<endl;
    // return ret;

    //std::cout << "euclidean cpu:"<<dot<<endl;
    return dot;
}
float compute_similary_sub_blas(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size_n = vec_feature1.size();
    vector<float>  tmp_feature;
    tmp_feature.reserve(2048);
    vs_2Sub(size_n,vec_feature1,vec_feature2,tmp_feature);
    //cblas_saxpy(N, alpha, X, 1, Y, 1);

// #ifdef _OPENMP
//     #pragma omp parallel
//     {
//         #pragma omp for 
//         for(int i =0;i<size_n;i++)
//         {
//             tmp_feature[i]=vec_feature1[i]-vec_feature2[i];
//         }
//     }
// #endif
    float dot =cblas_sdot(size_n, tmp_feature.data(), 1, tmp_feature.data(), 1);
    // float alpha=-1.0;
    // //tmp_feature=vec_feature2;
    // cblas_scopy(size_n, vec_feature2.data(), 1, tmp_feature.data(), 1);
    // cblas_saxpy(size_n, alpha, vec_feature1.data(), 1, tmp_feature.data(), 1);
    // float dot =cblas_sdot(size_n, tmp_feature.data(), 1, tmp_feature.data(), 1);
    //dot=dot/size_n;
    //std::cout << "euclidean blas:"<<dot<<endl;
    return dot;
}
float compute_similary_blas(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size_n = vec_feature1.size();
    vector<float>  tmp_feature;
    tmp_feature.reserve(2048);
//     vs_2Sub(size_n,vec_feature1,vec_feature2,tmp_feature);
//     //cblas_saxpy(N, alpha, X, 1, Y, 1);

// // #ifdef _OPENMP
// //     #pragma omp parallel
// //     {
// //         #pragma omp for 
// //         for(int i =0;i<size_n;i++)
// //         {
// //             tmp_feature[i]=vec_feature1[i]-vec_feature2[i];
// //         }
// //     }
// // #endif
//     float dot =cblas_sdot(size_n, tmp_feature.data(), 1, tmp_feature.data(), 1);
    float alpha=-1.0;
    //tmp_feature=vec_feature2;
    cblas_scopy(size_n, vec_feature2.data(), 1, tmp_feature.data(), 1);
    cblas_saxpy(size_n, alpha, vec_feature1.data(), 1, tmp_feature.data(), 1);
    float dot =cblas_sdot(size_n, tmp_feature.data(), 1, tmp_feature.data(), 1);
    //dot=dot/size_n;
    //std::cout << "euclidean blas:"<<dot<<endl;
    return dot;
}

float compute_similary_gpu(Blob<float> & Blob_1,Blob<float> & Blob_2)
{
    
  int count = Blob_1.count();
  Blob<float> diff_;
  diff_.ReshapeLike(Blob_1);
  caffe_gpu_sub(
      count,
      Blob_1.gpu_data(),
      Blob_2.gpu_data(),
      diff_.mutable_gpu_data());
  float dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  //float loss = dot / bottom[0]->num() / Dtype(2);
    //std::cout << "euclidean gpu:"<<dot<<endl;
    return dot;
}

void compare_cut4_freature()
{
    std::string model_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    std::string weights_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_110000.caffemodel";
    int GPUID=5;
    int max_ret_num=30;
    int classnum = 134;//134;//17; 21 120 38

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle);
    std::string strJpegDir = "/mnt/storage/dataset/tiannuo_data/seconde_data/baiwei082224_2152/test_beer/";
    //std::string out_file  = "/home/liushuai/work/crop_1108/paomian/outfile.txt"; //FLAGS_out_file;
    std::string out_file  = "/home/liushuai/work/goods_similary/build/paomian//res50_near5_neg_4_outfile.txt";
    const char * list_file = "/home/liushuai/work/crop_1108/paomian/10pos.txt";
    const char * pos_list_file = "/home/liushuai/work/crop_1108/paomian//pos.txt";
    const char * neg_list_file = "/home/liushuai/work/crop_1108/paomian//neg.txt";
    std::ifstream infile(list_file);
    std::string file;
    int imageCount = 0;
    //double time_use = 0;
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    
    vector<vector<vector<float> > > vec_4_feature;
    //vec_feature.push_back();;

    while (infile >> file) 
    {
        //if (file_type == "image") 
        //      string fileName = "JPEGImages/" + file + ".jpg";
        vector<float>  tmp_feature;
        tmp_feature.reserve(2048);
        std::string fileName = file;
        cv::Mat img1 = cv::imread(fileName.c_str());
        vector<vector<float> > vec_feature;
        handle->get_feature(img1, tmp_feature);
        vec_feature.push_back(tmp_feature);
        cv::Mat tmp_img;
        for (int i_cut = 0; i_cut < 2; i_cut++)
        {
            for (int j_cut = 0; j_cut < 2; j_cut++)
            {
                img1(cv::Rect(i_cut*0.5*img1.cols, j_cut*0.5*img1.rows, min(img1.cols*0.5,img1.cols), min(img1.rows*0.5,img1.rows))).copyTo(tmp_img);
                handle->get_feature(tmp_img, tmp_feature);
                vec_feature.push_back(tmp_feature);
            }
        }

        img1(cv::Rect(0.25*img1.cols, 0.25*img1.rows, min(img1.cols*0.75,img1.cols), min(img1.rows*0.75,img1.rows))).copyTo(tmp_img);
        handle->get_feature(tmp_img, tmp_feature);
        vec_feature.push_back(tmp_feature);
        vec_4_feature.push_back(vec_feature);
    }

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
    {
        //std::cout << "out_file  empty"<<endl;
        //return 0;
    }

    //std::cout << "multi_det begin 1 "<<endl;
    std::ostream out(buf); 

    std::ifstream neg_infile(neg_list_file);
    std::string neg_file;
    while (neg_infile >> neg_file) 
    {
        //if (file_type == "image") 
        //      string fileName = "JPEGImages/" + file + ".jpg";
        vector<float>  tmp_neg_feature;
        tmp_neg_feature.reserve(2048);
        std::string fileName = neg_file;
        cv::Mat img1 = cv::imread(fileName.c_str());
        vector<vector<float> > neg_vec_feature;
        handle->get_feature(img1, tmp_neg_feature);
        neg_vec_feature.push_back(tmp_neg_feature);
        cv::Mat tmp_img;
        for (int i_cut = 0; i_cut < 2; i_cut++)
        {
            for (int j_cut = 0; j_cut < 2; j_cut++)
            {
                img1(cv::Rect(i_cut*0.5*img1.cols, j_cut*0.5*img1.rows, min(img1.cols*0.5,img1.cols), min(img1.rows*0.5,img1.rows))).copyTo(tmp_img);
                handle->get_feature(tmp_img, tmp_neg_feature);
                neg_vec_feature.push_back(tmp_neg_feature);
            }
        }

        img1(cv::Rect(0.25*img1.cols, 0.25*img1.rows, min(img1.cols*0.75,img1.cols), min(img1.rows*0.75,img1.rows))).copyTo(tmp_img);
        handle->get_feature(tmp_img, tmp_neg_feature);
        neg_vec_feature.push_back(tmp_neg_feature);
        //handle->get_feature(img1, tmp_neg_feature);
        //vec_feature.push_back(tmp_feature);
        out<<fileName<<" : ";
        float arr_f[3]={9999.0000,9999.0000,9999.0000};
        //out<<" "<<arr_f[0]<<arr_f[1]<<arr_f[2]<<std::endl;
        float max_score=0.0000;
        for(int i=0;i<vec_4_feature.size();i++)
        {
            float sim_aver=0.0000;
            for(int zz=0;zz<5;zz++)
            {
                float sim=compute_similary(vec_4_feature[i][zz],neg_vec_feature[zz]);
                if(0==zz)
                {
                    sim_aver+=5.0*sim;
                }
                else
                {
                    sim_aver+=sim;
                }
            }
            sim_aver=sim_aver/10.0;
            //out<<sim<<" ";
            max_score=arr_f[0];
            bool bfind=false;

            int jmax=0;
            for(int j=0;j<3;j++)
            {
                if(max_score<arr_f[j])
                {
                    max_score=arr_f[j];
                    jmax = j;
                }
                if(sim_aver<arr_f[j])
                {
                    bfind=true;
                }
            }
            if(bfind)
            {
                arr_f[jmax]=sim_aver;
            }
        }
        //out<<" "<<arr_f[0]<<arr_f[1]<<arr_f[2]<<std::endl;
        float faverage_score= (arr_f[0]+arr_f[1]+arr_f[2])/3.0000;
        //out<< std::endl;
        out<<"===="<<faverage_score<< std::endl;
    }
}

void compare_multi_feature()
{
    std::string model_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    std::string weights_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_110000.caffemodel";
    std::string model_file_res101 = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/resnet101_nodropout/deploy.prototxt";;
    std::string weights_file_res101 = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/resnet101_nodropout/res101_classfy_train_iter_90000.caffemodel";
    int GPUID=5;
    int max_ret_num=30;
    int classnum = 134;//134;//17; 21 120 38

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle);
    Detector * handle2 = NULL;
    EV2641_InitCarDetector(model_file_res101.c_str(), weights_file_res101.c_str(),classnum, GPUID , handle2);
    std::string strJpegDir = "/mnt/storage/dataset/tiannuo_data/seconde_data/baiwei082224_2152/test_beer/";
    //std::string out_file  = "/home/liushuai/work/crop_1108/paomian/res50_near5_pos_outfile.txt"; //FLAGS_out_file;
    std::string out_file  = "/home/liushuai/work/goods_similary/build/paomian//combine_neg_outfile.txt";
    const char * list_file = "/home/liushuai/work/crop_1108/paomian/10pos.txt";
    const char * pos_list_file = "/home/liushuai/work/crop_1108/paomian//pos.txt";
    const char * neg_list_file = "/home/liushuai/work/crop_1108/paomian//neg.txt";
    std::ifstream infile(list_file);
    std::string file;
    int imageCount = 0;
    //double time_use = 0;
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    
    vector<vector<float> > vec_feature;
    vector<vector<float> > vec_feature_101;
    //vec_feature.push_back();;

    while (infile >> file) 
    {
        //if (file_type == "image") 
        //      string fileName = "JPEGImages/" + file + ".jpg";
        vector<float>  tmp_feature;
        vector<float>  tmp_feature_101;
        tmp_feature.reserve(2048);
        tmp_feature_101.reserve(2048);
        std::string fileName = file;
        cv::Mat img1 = cv::imread(fileName.c_str());
        handle->get_feature(img1, tmp_feature);
        handle2->get_feature(img1, tmp_feature_101,"fc368");
        vec_feature.push_back(tmp_feature);
        vec_feature_101.push_back(tmp_feature_101);
    }

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
    {
        //std::cout << "out_file  empty"<<endl;
        //return 0;
    }

    //std::cout << "multi_det begin 1 "<<endl;
    std::ostream out(buf); 

    std::ifstream neg_infile(neg_list_file);
    std::string neg_file;
    int iiii=0;
    int jjjj=0;
    int eeee=0;
    int sum=0;
    while (neg_infile >> neg_file) 
    {
        //if (file_type == "image") 
        //      string fileName = "JPEGImages/" + file + ".jpg";
        vector<float>  tmp_neg_feature;
        tmp_neg_feature.reserve(2048);
        vector<float>  tmp_neg_feature_101;
        tmp_neg_feature_101.reserve(2048);
        std::string fileName = neg_file;
        cv::Mat img1 = cv::imread(fileName.c_str());
        handle->get_feature(img1, tmp_neg_feature);
        handle2->get_feature(img1, tmp_neg_feature_101,"fc368");
        //vec_feature.push_back(tmp_feature);
        out<<fileName<<" : ";
        float arr_f[3]={9999.0000,9999.0000,9999.0000};
        //out<<" "<<arr_f[0]<<arr_f[1]<<arr_f[2]<<std::endl;
        float max_score=0.0000;
        float arr_f_101[3]={9999.0000,9999.0000,9999.0000};
        //out<<" "<<arr_f[0]<<arr_f[1]<<arr_f[2]<<std::endl;
        float max_score_101=0.0000;
        for(int i=0;i<vec_feature.size();i++)
        {
            float sim=compute_similary(vec_feature[i],tmp_neg_feature);
            float sim_101=compute_similary(vec_feature_101[i],tmp_neg_feature_101);
            //out<<sim<<" ";
            max_score=arr_f[0];
            bool bfind=false;

            int jmax=0;
            max_score_101=arr_f_101[0];
            bool bfind_101=false;

            int jmax_101=0;
            for(int j=0;j<3;j++)
            {
                if(max_score<arr_f[j])
                {
                    max_score=arr_f[j];
                    jmax = j;
                }
                if(sim<arr_f[j])
                {
                    bfind=true;
                }
                if(max_score_101<arr_f_101[j])
                {
                    max_score_101=arr_f_101[j];
                    jmax_101 = j;
                }
                if(sim_101<arr_f_101[j])
                {
                    bfind_101=true;
                }
            }
            if(bfind)
            {
                arr_f[jmax]=sim;
            }
            if(bfind_101)
            {
                arr_f_101[jmax_101]=sim_101;
            }
        }
        //out<<" "<<arr_f[0]<<arr_f[1]<<arr_f[2]<<std::endl;
        float faverage_score= (arr_f[0]+arr_f[1]+arr_f[2])/3.0000;
        float faverage_score_101= (arr_f_101[0]+arr_f_101[1]+arr_f_101[2])/3.0000;
        //out<< std::endl;
        bool bset =false;
        if(faverage_score_101>0.3)
        {
            jjjj+=1;
            out<<"===="<<" neg ";
            bset=true;
        }
        if(faverage_score<0.26)
        {
            iiii+=1;
            out<<"===="<<" pos ";
            bset=true;
        }
        if(!bset)
        {
            eeee+=1;
            out<<"===="<<" error ";
        }
        out<<"===="<<faverage_score<<"===="<<faverage_score_101<< std::endl;
        sum+=1;
    }
    out<<"***** pos="<<iiii<<" =neg="<<jjjj<<" =error="<<eeee<<" =sum="<<sum<< std::endl;
}

const    int countteme=100000;
float compare_feature_two_pic_blob(const char* p1,const char* p2)
{
    // std::string model_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    // std::string weights_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_110000.caffemodel";
    // std::string model_file = "/storage2/for_gs4/compare/classify_siamese.2.0.prototxt";
    // std::string weights_file = "/storage2/for_gs4/compare/classify_siamese.2.0.caffemodel";
    std::string model_file = "/storage2/for_gs4/compare/res101_dropout_calssfy.prototxt";
    std::string weights_file = "/storage2/for_gs4/compare/res101_dropout_calssfy.caffemodel";
    // std::string model_file = "/storage2/liushuai/gs3/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    // std::string weights_file = "/storage2/liushuai/gs3/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_200000.caffemodel";
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/resnet101_nodropout/deploy.prototxt";;
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/resnet101_nodropout/res101_classfy_train_iter_90000.caffemodel";
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/50_siamese/deploy.prototxt";;
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/50_siamese/mnist_siamese_iter_95000.caffemodel";
    int GPUID=1;
    int max_ret_num=30;
    int classnum = 2348;//134;//17; 21 120 38

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle);
    cv::Mat img1 = cv::imread(p1);
    cv::Mat img33 = cv::imread(p2);
    // vector<float>  vec_feature1;
    // vec_feature1.reserve(2048);
    caffe::Blob<float> blob1;
    handle->get_feature_blob(img1, blob1);

    Mat M_feature( 1, 2048, CV_32FC1 );
            memcpy(M_feature.data,blob1.cpu_data(),blob1.count()*sizeof(float));
            stringstream ss;
            ss<<1;
            string save_dir="/storage2/liushuai/tiannuocaffe/goods_similary/build/test/";
            string tmp_string;
            ss>>tmp_string;
            string tmp_file=save_dir+tmp_string+"blob_feature_1.c++";
            writeMatToFile(M_feature,tmp_file.c_str());
            // WriteData(M_feature,tmp_string.c_str(),goods[i].vec_feature101.size());
            // vector<float>  out_vec;
            // LoadData(tmp_string,out_vec,goods[i].vec_feature101.size());
            // bool bequal=true;
            // int jjj=0;
            // if(equal(out_vec.begin(),out_vec.end(),goods[i].vec_feature101.begin()))
            // //if(bequal)
            // {
            //     cout<<"**********************************************************vector = vector"<<endl;
            // }
            // else
            // {
            //     cout<<"**********************************************************vector != vector"<<jjj<<" "<<out_vec[0]<<" "<<goods[i].vec_feature101[0]<<" "<<out_vec[1]<<" "<<goods[i].vec_feature101[1]<<endl;
            // }
    // vector<float>  vec_feature2;
    // vec_feature2.reserve(2048);
    caffe::Blob<float> blob2;
    handle->get_feature_blob(img33, blob2);
    // ss<<2;
    // ss>>tmp_string;
    // tmp_file=save_dir+tmp_string+"_feature_2.c++";
    // writeMatToFile(M_feature,tmp_file.c_str());
    // if(vec_feature1.size()==0 || vec_feature1.size()!=vec_feature2.size())
    // {
    //   return 100.0;
    // }
    //float ret = compute_similary(vec_feature1,vec_feature2);

    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    int icc =0;
    float ret=-1.0;
    for(icc=0;icc<countteme;icc++)
    {
         ret=compute_similary_gpu(blob1,blob2); 
    }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_gpuTotal Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    
    cout<<"compute_similary_gpu:"<<ret<<endl;
    return ret;
}
#include<algorithm>
void adjust(float arr[], int len, int index)
{
    int left = 2*index + 1;
    int right = 2*index + 2;
    int maxIdx = index;
    if(left<len && arr[left] < arr[maxIdx]) maxIdx = left;
    if(right<len && arr[right] < arr[maxIdx]) maxIdx = right;  // maxIdx是3个数中最大数的下标
    if(maxIdx != index)                 // 如果maxIdx的值有更新
    {
        swap(arr[maxIdx], arr[index]);
        adjust(arr, len, maxIdx);       // 递归调整其他不满足堆性质的部分
    }

}
void heapSort(float arr[], int size,int min=32)
{
    for(int i=size/2 - 1; i >= 0; i--)  // 对每一个非叶结点进行堆调整(从最后一个非叶结点开始)
    {
        adjust(arr, size, i);
    }
    for(int i = size - 1; i >= 1; i--)
    {
        swap(arr[0], arr[i]);           // 将当前最大的放置到数组末尾
        adjust(arr, i, 0);            // 将未完成排序的部分继续进行堆排序
        if (size-i==32)
        {
            return;
        }
    }
}
float compare_feature_two_pic(const char* p1,const char* p2)
{
    // std::string model_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    // std::string weights_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_110000.caffemodel";
    // std::string model_file = "/storage2/for_gs4/compare/classify_siamese.2.0.prototxt";
    // std::string weights_file = "/storage2/for_gs4/compare/classify_siamese.2.0.caffemodel";
    std::string model_file = "/storage2/for_gs4/compare/res101_dropout_calssfy.prototxt";
    std::string weights_file = "/storage2/for_gs4/compare/res101_dropout_calssfy.caffemodel";
    // std::string model_file = "/storage2/liushuai/gs3/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    // std::string weights_file = "/storage2/liushuai/gs3/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_200000.caffemodel";
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/resnet101_nodropout/deploy.prototxt";;
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/resnet101_nodropout/res101_classfy_train_iter_90000.caffemodel";
    // std::string model_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/50_siamese/deploy.prototxt";;
    // std::string weights_file = "/home/liushuai/RFCN/py-R-FCN-master/caffe/models/50_siamese/mnist_siamese_iter_95000.caffemodel";
    int GPUID=5;
    int max_ret_num=30;
    int classnum = 2348;//134;//17; 21 120 38

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle);
    cv::Mat img1 = cv::imread(p1);
    cv::Mat img33 = cv::imread(p2);
    vector<float>  vec_feature1;
    vec_feature1.reserve(2048);
    handle->get_feature(img1, vec_feature1);

    Mat M_feature( 1, 2048, CV_32FC1 );
            memcpy(M_feature.data,vec_feature1.data(),vec_feature1.size()*sizeof(float));
            stringstream ss;
            ss<<1;
            string save_dir="/home/liushuai/work/goods_similary/build/test//";
            string tmp_string;
            ss>>tmp_string;
            string tmp_file=save_dir+tmp_string+"_feature_1.c++";
            writeMatToFile(M_feature,tmp_file.c_str());
            // WriteData(M_feature,tmp_string.c_str(),goods[i].vec_feature101.size());
            // vector<float>  out_vec;
            // LoadData(tmp_string,out_vec,goods[i].vec_feature101.size());
            // bool bequal=true;
            // int jjj=0;
            // if(equal(out_vec.begin(),out_vec.end(),goods[i].vec_feature101.begin()))
            // //if(bequal)
            // {
            //     cout<<"**********************************************************vector = vector"<<endl;
            // }
            // else
            // {
            //     cout<<"**********************************************************vector != vector"<<jjj<<" "<<out_vec[0]<<" "<<goods[i].vec_feature101[0]<<" "<<out_vec[1]<<" "<<goods[i].vec_feature101[1]<<endl;
            // }
    vector<float>  vec_feature2;
    vec_feature2.reserve(2048);
    handle->get_feature(img33, vec_feature2);
    // ss<<2;
    // ss>>tmp_string;
    // tmp_file=save_dir+tmp_string+"_feature_2.c++";
    // writeMatToFile(M_feature,tmp_file.c_str());
    if(vec_feature1.size()==0 || vec_feature1.size()!=vec_feature2.size())
    {
      return 100.0;
    }
    int icc =0;
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    float ret=-1.0;
    for(icc=0;icc<countteme;icc++)
    {
         ret= compute_similary(vec_feature1,vec_feature2);
    }
    gettimeofday(&end,NULL);
    std::cout << "compute_similaryTotal Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    
    gettimeofday(&start,NULL);
    float ret_MP=1000;//reduction(+:ret_1)
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for  
        for(icc=0;icc<countteme;icc++)
        {
            ret_MP=compute_similary(vec_feature1,vec_feature2);
        }
    }
#endif
    gettimeofday(&end,NULL);
    std::cout << "compute_similary+MP Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    

    gettimeofday(&start,NULL);
    float ret_1_MP=1000;//reduction(+:ret_1)
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for  
        for(icc=0;icc<countteme;icc++)
        {
            ret_1_MP=compute_similary_blas(vec_feature1,vec_feature2);
        }
    }
#endif
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_blasTotal+MP Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    
    gettimeofday(&start,NULL);
    float ret_1=1000;//reduction(+:ret_1)
    for(icc=0;icc<countteme;icc++)
    {
        ret_1 = compute_similary_blas(vec_feature1,vec_feature2);
    }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_blasTotal Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    

    gettimeofday(&start,NULL);
    //float ret_1s=-1.0;
    float ret_1s_cos_blas=1000;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for 
        for(icc=0;icc<countteme;icc++)
        {
            ret_1s_cos_blas=compute_similary_cos_blas(vec_feature1,vec_feature2);
        }
    }
#endif
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
    // }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_cos_blas+MP Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    
    gettimeofday(&start,NULL);
    //float ret_1s=-1.0;
    float ret_1s_cos=1000;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for 
        for(icc=0;icc<countteme;icc++)
        {
            ret_1s_cos=compute_similary_cos(vec_feature1,vec_feature2);
        }
    }
#endif
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
    // }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_cos+MP Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    

    gettimeofday(&start,NULL);
    //float ret_1s=-1.0;
    float ret_1s_MP=1000;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for 
        for(icc=0;icc<countteme;icc++)
        {
            ret_1s_MP=compute_similary_sub_blas(vec_feature1,vec_feature2);
        }
    }
#endif
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
    // }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_sub_blas+MP Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    
    gettimeofday(&start,NULL);
    //float ret_1s=-1.0;
    float ret_1s_MParray=1000;
    float ff_array[countteme];
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for 
        for(icc=0;icc<countteme;icc++)
        {
            ff_array[icc]=compute_similary_sub_blas(vec_feature1,vec_feature2);
        }
    }
#endif
    for(icc=0;icc<countteme;icc++)
    {
        if (ret_1s_MParray<ff_array[icc])
        {
            ret_1s_MParray=ff_array[icc];
        }
    }
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
    // }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_sub_blas+MP for array Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
     std::cout << "compute_similary_sub_blas+MP for array Time: "<<ret_1s_MParray<<endl;

    gettimeofday(&start,NULL);
    //float ret_1s=-1.0;
    float ret_1s_MParray_16=1000;
    float ff_array_16[countteme];
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for 
        for(icc=0;icc<countteme;icc++)
        {
            ff_array_16[icc]=compute_similary_sub_blas(vec_feature1,vec_feature2);
        }
    }
#endif
    heapSort(ff_array_16,countteme,16);
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
    // }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
     std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[0]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[1]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[2]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[3]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[4]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[5]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[6]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[7]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[8]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[9]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[10]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[11]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[12]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[13]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[14]<<endl;
      std::cout << "compute_similary_sub_blas+MP_16 for array Time: "<<ff_array_16[15]<<endl;

    gettimeofday(&start,NULL);
    //float ret_1s=-1.0;
    float ret_1s_MP__=1000;
    float minii_shared =0;
#ifdef _OPENMP
    #pragma omp parallel shared(minii_shared) private(icc)
    {
        #pragma omp for 
        for(icc=0;icc<countteme;icc++)
        {
            ret_1s_MP__=compute_similary_sub_blas(vec_feature1,vec_feature2);
            #pragma omp critical
            {
                if (minii_shared<icc)
                {
                    minii_shared=icc;
                }
            }
        }
    }
#endif
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
    // }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_sub_blas+MP private+critical Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    std::cout << "compute_similary_sub_blas+MP private+critical Time: "<<minii_shared<<endl;

    gettimeofday(&start,NULL);
    float ret_1s=1000;
    float ret_1s_min=10000;
    for(icc=0;icc<countteme;icc++)
    {
        ret_1s = compute_similary_sub_blas(vec_feature1,vec_feature2);
        if (ret_1s_min>ret_1s)
        {
            ret_1s_min = ret_1s;
        }
    }
    gettimeofday(&end,NULL);
    std::cout << "compute_similary_sub_blas Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    

    // gettimeofday(&start,NULL);
    // float ret_2=-1.0;
    // for(icc=0;icc<countteme;icc++)
    // {
    //     ret_2 = compute_similary_MP(vec_feature1,vec_feature2);
    // }
    // gettimeofday(&end,NULL);
    // std::cout << "compute_similary_MP Time: "<< end.tv_sec-start.tv_sec << " s " << end.tv_usec-start.tv_usec << " us " <<" imageCount "<<countteme<< endl;
    
    cout<<"compute_similary:"<<ret<<endl;
    cout<<"compute_similary_blas:"<<ret_1<<endl;
    //cout<<"compute_similary_MP:"<<ret_2<<endl;
    cout<<"compute_similary_sub_blas:"<<ret_1s<<endl;
    cout<<"compute_similary_cos:"<<ret_1s_cos<<endl;
    cout<<"compute_similary_cos_blas:"<<ret_1s_cos_blas<<endl;
    cout<<"test:"<<12/4/3<<endl;


    return ret;
}

float compare_feature()
{

    std::string model_file = "/storage2/liushuai/gs3/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    std::string weights_file = "/storage2/liushuai/gs3/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_200000.caffemodel";
    int GPUID=1;
    int max_ret_num=30;
    int classnum = 134;//134;//17; 21 120 38

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle);
    cv::Mat img1 = cv::imread("/storage/liushuai/work/test/gallery/g12//103.jpg");
    cv::Mat img33 = cv::imread("/storage/liushuai//work/test/query/q12/11078.jpg");
    cv::Mat img34 = cv::imread("/storage/liushuai//work/test/query/q12/11079.jpg");
    cv::Mat img40 = cv::imread("/storage/liushuai//work/test/query/q12/11085.jpg");
    //cv::Mat img = cv::imread("/mnt/storage/liushuai/data/beercut134/beercut134proj1/JPEGImages/budweiser10129_1.jpg");
    // handle->Detect(inputimage, detection_result);
    // for(int i=0;i < detection_result.size(); i++){
    //     cv::rectangle(inputimage,cv::Point(detection_result[i].x,detection_result[i].y),
    //                              cv::Point(detection_result[i].x + detection_result[i].width,detection_result[i].y + detection_result[i].height),
    //                              cv::Scalar(0,255,0));

    // }
    // cv::imwrite("MDLZ05161001003.jpg",inputimage);

    vector<float>  vec_feature1;
    vec_feature1.reserve(2048);
    handle->get_feature(img1, vec_feature1);
    cout<<endl;
    int i=0;
    for( i=0;i<vec_feature1.size();i++)
    {
        cout<<i<<":"<<vec_feature1[i]<<" \n";
    }
    cout<<endl;
    cout<<i<<endl;

    vector<float>  vec_feature2;
    vec_feature2.reserve(2048);
    handle->get_feature(img33, vec_feature2);
    if(vec_feature1.size()==0 || vec_feature1.size()!=vec_feature2.size())
    {
      return 100.0;
    }
    cout<<"                   33 :"<<compute_similary(vec_feature1,vec_feature2)<<endl;


    vector<float>  vec_feature3;
    vec_feature3.reserve(2048);
    handle->get_feature(img34, vec_feature3);
    if(vec_feature1.size()==0 || vec_feature1.size()!=vec_feature3.size())
    {
      return 100.0;
    }
    cout<<"                   34 :"<<compute_similary(vec_feature1,vec_feature3)<<endl;

    vector<float>  vec_feature4;
    vec_feature4.reserve(2048);
    handle->get_feature(img40, vec_feature4);
    if(vec_feature1.size()==0 || vec_feature1.size()!=vec_feature4.size())
    {
      return 100.0;
    }
    cout<<"                    40 :"<<compute_similary(vec_feature1,vec_feature4)<<endl;
}

DEFINE_string(model_file, "", "model prototxt file");
DEFINE_string(weights_file, "", "model weight file");
DEFINE_string(strJpegDir, "", "JPEG dir");
DEFINE_string(saveJpegDir, "", "JPEG result dir");
DEFINE_string(out_file, "", "out txt file");
DEFINE_string(list_file, "", "test file list");
DEFINE_bool(single_multi, false, "single or multi");//
DEFINE_bool(cutornot, false, "is 4 cut");//
DEFINE_int32(GPUID, 0, "gpu id");
DEFINE_int32(Batch_size, 32, "Batch_size");
DEFINE_double(NMS_threhold, 0.3, "NMS threhold");
DEFINE_double(CONF_threhold, 0.3, "conf threhold");
DEFINE_string(libdir, "/home/liushuai/tiannuocaffe/py-rfcn-gpu//lib/", "py/lib/ dir");
int main(int argc, char** argv)
{
    gflags::SetUsageMessage("for test\n"
        "format used as input for test caffe.\n"
        "Usage:\n"
        "    test\n");
    caffe::GlobalInit(&argc, &argv);
        cout<<"main :"<<endl;
    int i_batch_size=FLAGS_Batch_size;
    std::string strJpegDir = FLAGS_strJpegDir;
    const char * list_file = FLAGS_list_file.c_str();
    // const char* p1 = "/storage2/liushuai/gs6_env/market1501_extract_freature/test/patch_dir_NG/snow15/162d1d74073w6jz1yznatvf3_0.jpg";
    // const char* p2 = "/storage2/liushuai/gs6_env/market1501_extract_freature/test/patch_dir_NG/qingdao1/162d1fdd1733k3mwmqpuzkdm_0.jpg";
    const char* p1 = "/home/liushuai/work/goods_similary/build/test/5_10.jpg";//
    const char* p2 = "/home/liushuai/work/goods_similary/build/test/9_10.jpg";
    // const char* p1 = "/storage2/liushuai/data/similary_data/new_trainsimilary/naifen_crop/1Nestle04/1484_1Nestle04229576.jpg";
    // const char* p2 = "/storage2/liushuai/data/similary_data/new_trainsimilary/naifen_crop/1Nestle04_t_b_flip/1489_1Nestle04_t_b_flip230756.jpg";
    // const char* p3="/storage2/liushuai/data/similary_data/new_trainsimilary/naifen_crop/1Nestle04/1484_1Nestle04229575.jpg";
    //float ret1=compare_feature_two_pic(p1,p2);

    std::ifstream fin(list_file,ios_base::in);
    const int N=1024;
    char buf[N];
    int lineCnt=0;
    while(fin.getline(buf,N))
    {   
        lineCnt++;
    }
    fin.close();
    cout<<lineCnt<<endl;

    std::ifstream infile(list_file);
        cout<<"main : "<<list_file<<endl;
    std::string file;
    int imageCount = 0;
    int do_count=0;
    vector< cv::Mat > vec_imgs;
    std::string model_file = "/storage2/for_gs4/compare/res101_dropout_calssfy.prototxt";
    std::string weights_file = "/storage2/for_gs4/compare/res101_dropout_calssfy.caffemodel";
    int GPUID=FLAGS_GPUID;
    int max_ret_num=30;
    int classnum = 2348;//134;//17; 21 120 38

    Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(),classnum, GPUID , handle);
    vector<vector<float> > vec_feature1;
    //vec_feature1.reserve(i_batch_size);
     
        cout<<"main : file begin"<<endl;
    while (infile >> file) 
    {
        imageCount++;
        std::string fileName = strJpegDir + file + ".jpg";
        cv::Mat img = cv::imread(fileName, -1);
        cout<<"fileName :"<<fileName<<endl;
        CHECK(!img.empty()) << "Unable to decode image " << file;
        vec_imgs.push_back(img);
        if (0 == imageCount%i_batch_size ||imageCount==lineCnt)
        {
            do_count++;
            handle->get_feature(vec_imgs, vec_feature1,(imageCount==lineCnt) ? (imageCount%i_batch_size) : i_batch_size);
            for (int i_vec = 0; i_vec < vec_feature1.size(); ++i_vec)
            {
                
                Mat M_feature( 1, 2048, CV_32FC1 );
                memcpy(M_feature.data,vec_feature1[i_vec].data(),vec_feature1[i_vec].size()*sizeof(float));
                int zz=(imageCount-1)/i_batch_size;
                stringstream ss;
                ss<<zz<<i_vec;
        cout<<"imageCount/i_batch_size : file"<<zz<<" "<<i_vec<<endl;
                string save_dir="/home/liushuai/work/goods_similary/test//";
                string tmp_string;
                ss>>tmp_string;
                string tmp_file=save_dir+tmp_string+"_blob_feature_.txt";
                writeMatToFile(M_feature,tmp_file.c_str());
            }
            vec_imgs.clear();
            vec_feature1.clear();
            /* code */
        }
    }
    if (do_count<((lineCnt+i_batch_size-1)/i_batch_size))
    {
        cout<<"not over all"<<std::endl;
            // handle->get_feature(vec_imgs, vec_feature1,imageCount%i_batch_size);
            // for (int i_vec = 0; i_vec < vec_feature1.size(); ++i_vec)
            // {
                
            //     Mat M_feature( 1, 2048, CV_32FC1 );
            //     memcpy(M_feature.data,vec_feature1[i_vec].data(),vec_feature1[i_vec].size()*sizeof(float));
            //     stringstream ss;
            //     ss<<(imageCount/i_batch_size)<<i_vec;
            //     string save_dir="/home/liushuai/work/goods_similary/build/test/";
            //     string tmp_string;
            //     ss>>tmp_string;
            //     string tmp_file=save_dir+tmp_string+"_blob_feature_.txt";
            //     writeMatToFile(M_feature,tmp_file.c_str());
            // }
            // vec_imgs.clear();
    }
    //float ret2=compare_feature_two_pic(p1,p2);
    //  float ret3=compare_feature_two_pic_blob(p1,p2);
    // cout<<"12 :"<<ret2<<endl;
    // cout<<"13 :"<<ret2<<endl;
    //compare_cut4_freature();
    //compare_multi_feature();
    //compare_feature();

        cout<<"main : return"<<endl;
    return 0;
}
