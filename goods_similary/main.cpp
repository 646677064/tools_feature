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


float compute_similary(vector<float> & vec_feature1,vector<float> & vec_feature2)
{
    int size = vec_feature1.size();
    float multiply=0.0;
    float square1=0.0;
    float square2=0.0;
    std::cout << "size:"<<size<<endl;
    for(int i =0;i<size;i++)
    {
        multiply+=vec_feature1[i]*vec_feature2[i];
        square1+=vec_feature1[i]*vec_feature1[i];
        square2+=vec_feature2[i]*vec_feature2[i];
    }
    float sqrt1 = sqrt(square1);
    float sqrt2 = sqrt(square2);
    float ret = (square1+square2-2*multiply)/(sqrt1*sqrt2);
    std::cout << "similary:"<<ret<<endl;
    return ret;
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

float compare_feature_two_pic(const char* p1,const char* p2)
{
    // std::string model_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/dev.proto";
    // std::string weights_file = "/home/liushuai/work/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_110000.caffemodel";
    std::string model_file = "/storage2/for_gs4/compare/classify_siamese.2.0.prototxt";
    std::string weights_file = "/storage2/for_gs4/compare/classify_siamese.2.0.caffemodel";
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
    vector<float>  vec_feature1;
    vec_feature1.reserve(2048);
    handle->get_feature(img1, vec_feature1);

    Mat M_feature( 1, 2048, CV_32FC1 );
            memcpy(M_feature.data,vec_feature1.data(),vec_feature1.size()*sizeof(float));
            stringstream ss;
            ss<<1;
            string save_dir="/storage2/liushuai/gs6_env/market1501_extract_freature/test//";
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
    ss<<2;
    ss>>tmp_string;
    tmp_file=save_dir+tmp_string+"_feature_2.c++";
    writeMatToFile(M_feature,tmp_file.c_str());
    if(vec_feature1.size()==0 || vec_feature1.size()!=vec_feature2.size())
    {
      return 100.0;
    }
    float ret = compute_similary(vec_feature1,vec_feature2);
    //cout<<"                   33 :"<<ret<<endl;
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

int main()
{
    const char* p1 = "/storage2/liushuai/gs6_env/market1501_extract_freature/test/patch_dir_NG/snow15/162d1d74073w6jz1yznatvf3_0.jpg";
    const char* p2 = "/storage2/liushuai/gs6_env/market1501_extract_freature/test/patch_dir_NG/qingdao1/162d1fdd1733k3mwmqpuzkdm_0.jpg";
    // const char* p1 = "/storage2/liushuai/data/similary_data/new_trainsimilary/naifen_crop/1Nestle04/1484_1Nestle04229576.jpg";
    // const char* p2 = "/storage2/liushuai/data/similary_data/new_trainsimilary/naifen_crop/1Nestle04_t_b_flip/1489_1Nestle04_t_b_flip230756.jpg";
    // const char* p3="/storage2/liushuai/data/similary_data/new_trainsimilary/naifen_crop/1Nestle04/1484_1Nestle04229575.jpg";
    float ret1=compare_feature_two_pic(p1,p2);
    // float ret2=compare_feature_two_pic(p1,p3);
    cout<<"12 :"<<ret1<<endl;
    // cout<<"13 :"<<ret2<<endl;
    //compare_cut4_freature();
    //compare_multi_feature();
    //compare_feature();
    return 0;
}
