/*
* convertImgToSiamese.cpp
*/

#include <algorithm>
#include <fstream>
#include <string>
#include <cstdio>
#include <utility>
#include <vector>
//#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "leveldb/db.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/math_functions.hpp"

#include "opencv2/opencv.hpp"
#include "google/protobuf/text_format.h"
#include "stdint.h"
#include <cstdio>
#include <iostream>
#include <cmath>

#ifdef USE_LEVELDB
#include "leveldb/db.h"
using namespace caffe;
using std::pair;
using boost::scoped_ptr;
using namespace cv;
using namespace std;

DEFINE_bool(gray, false, "when this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false, "randomly shuffle the order of images and their labels");
DEFINE_string(backend, "leveldb", "the backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 224, "Width images are resized to");
DEFINE_int32(resize_height, 224, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_string(rootfold, "/", "rootfold");
DEFINE_string(listfle, "/home/liushuai/work/similary/list_file.txt", "listfle");
DEFINE_string(output, "/home/liushuai/work/train.leveldb", "output");
DEFINE_int32(channel, 3, "channel numbers of the image");     //1

//static bool ReadImageToMemory(const string &FileName, const int Height, const int Width, char *Pixels)   //2
static bool ReadImageToMemory(const string &FileName, const int Height, const int Width, char *Pixels)
{
    //read image
    //cv::Mat OriginImage = cv::imread(FileName, cv::IMREAD_GRAYSCALE);
    cv::Mat OriginImage = cv::imread(FileName);     //3. read color image
    CHECK(OriginImage.data) << "Failed to read the image.\n";

    //resize the image
    cv::Mat ResizeImage;
    cv::resize(OriginImage, ResizeImage, cv::Size(Width, Height));
    CHECK(ResizeImage.rows == Height) << "The heighs of Image is no equal to the input height.\n";
    CHECK(ResizeImage.cols == Width) << "The width of Image is no equal to the input width.\n";
    CHECK(ResizeImage.channels() == 3) << "The channel of Image is no equal to three.\n";    //4. should output the warning here

    // LOG(INFO) << "height " << ResizeImage.rows << " ";
    //LOG(INFO) << "weidth " << ResizeImage.cols << " ";
    //LOG(INFO) << "channels " << ResizeImage.channels() << "\n";

    // copy the image data to Pixels
    for (int HeightIndex = 0; HeightIndex < Height; ++HeightIndex)
    {
        const uchar* ptr = ResizeImage.ptr<uchar>(HeightIndex);
        int img_index = 0;
        for (int WidthIndex = 0; WidthIndex < Width; ++WidthIndex)
        {
            for (int ChannelIndex = 0; ChannelIndex < ResizeImage.channels(); ++ChannelIndex)
            {
                int datum_index = (ChannelIndex * Height + HeightIndex) * Width + WidthIndex;
                *(Pixels + datum_index) = static_cast<char>(ptr[img_index++]);
            }
        }
    }
    return true;
}

int main(int argc, char** argv)
{
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
    //::google::InitGoogleLogging(argv[0]);
// #ifndef GFLAGS_GFLAGS_H_
//     namespace gflags = google;
// #endif
// google::InitGoogleLogging(argv[0]);  // 初始化 glog
//     google::ParseCommandLineFlags(&argc, &argv, true);  // 初始化 gflags
    gflags::SetUsageMessage("Convert a set of color images to the leveldb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
    //caffe::GlobalInit(&argc, &argv);
    // 输入参数不足时报错
    // if (argc < 4)
    // {
    //     gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    //     return 1;
    // }

    // 读取图像名字和标签
        LOG(INFO) << FLAGS_listfle;
    std::ifstream infile(FLAGS_listfle.c_str());
    std::vector<std::pair<std::string, int> > lines;
    std::string filename;
    int ilable;
    //std::string pairname;
    int label;
  int mx_label = -1;
  int mi_label = INT_MAX;
  vector<vector<size_t> > label_set;
    while (infile >> filename >> ilable)
    {
        lines.push_back(std::make_pair(filename, ilable));
    mx_label = std::max(mx_label, ilable);
    mi_label = std::min(mi_label, ilable);
    }

    // 打乱图片顺序
    if (FLAGS_shuffle)
    {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        //shuffle(lines.begin(), lines.end());
        // 设置图像的高度和宽度
        int resize_height = std::max<int>(0, FLAGS_resize_height);
        int resize_width = std::max<int>(0, FLAGS_resize_width);
        int channel = std::max<int>(1, FLAGS_channel);     //5. add channel info

        // 打开数据库
        // Open leveldb
        // leveldb::DB* db;
        // leveldb::Options options;
        // options.create_if_missing = true;
        // options.error_if_exists = true;
        // leveldb::Status status = leveldb::DB::Open(
        //     options, FLAGS_output, &db);
        // CHECK(status.ok()) << "Failed to open leveldb " << argv[3]
        //     << ". Is it already existing?";
  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(FLAGS_output, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

        // 保存到leveldb
        // Storing to leveldb
        std::string root_folder(FLAGS_rootfold);
        //char* Pixels = new char[2 * resize_height * resize_width];
        char* Pixels = new char[2 * resize_height * resize_width * channel];    //6. add channel
        const int kMaxKeyLength = 10;   //10
        char key[kMaxKeyLength];
        std::string value;

        caffe::Datum datum;
        //datum.set_channels(2);  // one channel for each image in the pair
        datum.set_channels(2 * channel);                //7. 3 channels for each image in the pair
        datum.set_height(resize_height);
        datum.set_width(resize_width);

        //
        // int line_size = (int)(lines.size()/2);
        // std::cout<<"number of lines: "<<line_size<<endl;
        int LineIndex = 0;
        for (; LineIndex < lines.size(); LineIndex++)
        {

            //int PairIndex = LineIndex + line_size;
            // cout<<PairIndex<<endl;
            // int PairIndex = caffe::caffe_rng_rand() % lines.size();
            int i= caffe::caffe_rng_rand() % lines.size();
            int label_i = lines[i].second;  // pick a random  pair

            //int j = caffe::caffe_rng_rand() % lines.size();
            int j = caffe::caffe_rng_rand() % lines.size();
            int label_j =lines[j].second;

            char* FirstImagePixel = Pixels;
            // cout<<root_folder + lines[LineIndex].first<<endl;
            ReadImageToMemory(root_folder + lines[i].first, resize_height, resize_width, FirstImagePixel);  //8. add channel here

            //char *SecondImagePixel = Pixels + resize_width * resize_height;
            char *SecondImagePixel = Pixels + resize_width * resize_height * channel;       //10. add channel
            ReadImageToMemory(root_folder + lines[j].first, resize_height, resize_width, SecondImagePixel);  //9. add channel here

            // set image pair data
            // datum.set_data(Pixels, 2 * resize_height * resize_width);
            datum.set_data(Pixels, 2 * resize_height * resize_width * channel);     //11. correct

            cout <<LineIndex<<":   "<< label_i<<"===="<<label_j<<endl;
            if (label_i  == label_j) {
              datum.set_label(1);
            } else {
              datum.set_label(0);
            }
            datum.SerializeToString(&value);
            std::string key_str = caffe::format_int(LineIndex, 8);
            txn->Put(key_str, value);

            if (LineIndex % 1000 == 0) {
              // Commit db
              txn->Commit();
              txn.reset(db->NewTransaction());
              LOG(INFO) << "Processed " << LineIndex << " files.";
            }
            //db->Put(leveldb::WriteOptions(), key_str, value);
        }
          // write the last batch
          if (LineIndex % 1000 != 0) {
            txn->Commit();
            LOG(INFO) << "Processed " << LineIndex << " files.";
          }
        //delete db;
        delete[] Pixels;
    }
    else
    {
          CHECK_EQ(mi_label, 0);
          label_set.clear();
          label_set.resize(mx_label+1);
        for (size_t index = 0; index < lines.size(); index++) 
        {
            int tmp_label = lines[index].second;
            label_set[tmp_label].push_back(index);
        }
        for (size_t index = 0; index < label_set.size(); index++) 
        {
            CHECK_GT(label_set[index].size(), 0) << "label : " << index << " has no images";
        }
          float pos_fraction =1.0;
          float neg_fraction =4.0;
          int pos = pos_fraction * 10000;
          int neg = neg_fraction * 10000;


        LOG(INFO) << "A total of " << lines.size() << " images："<<FLAGS_resize_height<<"*"<<FLAGS_resize_width<<"*"<<FLAGS_channel;

        // 设置图像的高度和宽度
        int resize_height = std::max<int>(0, FLAGS_resize_height);
        int resize_width = std::max<int>(0, FLAGS_resize_width);
        int channel = std::max<int>(1, FLAGS_channel);     //5. add channel info

        // 打开数据库
        // // Open leveldb
        // leveldb::DB* db;
        // leveldb::Options options;
        // options.create_if_missing = true;
        // options.error_if_exists = true;
        // leveldb::Status status = leveldb::DB::Open(
        //     options, FLAGS_output, &db);
        // CHECK(status.ok()) << "Failed to open leveldb " << argv[3]
        //     << ". Is it already existing?";
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(FLAGS_output, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

        // 保存到leveldb
        // Storing to leveldb
        std::string root_folder(FLAGS_rootfold);
        //char* Pixels = new char[2 * resize_height * resize_width];
        char* Pixels = new char[2 * resize_height * resize_width * channel];    //6. add channel
        const int kMaxKeyLength = 10;   //10
        char key[kMaxKeyLength];
        std::string value;

        caffe::Datum datum;
        //datum.set_channels(2);  // one channel for each image in the pair
        datum.set_channels(2 * channel);                //7. 3 channels for each image in the pair
        datum.set_height(resize_height);
        datum.set_width(resize_width);

        //
        // int line_size = (int)(lines.size()/2);
        // std::cout<<"number of lines: "<<line_size<<endl;

        int LineIndex = 0;
        for (; LineIndex < lines.size(); LineIndex++)
        {

            //int PairIndex = LineIndex + line_size;
            // cout<<PairIndex<<endl;
            // int PairIndex = caffe::caffe_rng_rand() % lines.size();
            int i= caffe::caffe_rng_rand() % lines.size();
            int label_i = lines[i].second;  // pick a random  pair

            //int j = caffe::caffe_rng_rand() % lines.size();
            int x = caffe::caffe_rng_rand() % (pos + neg);
            int label_j;
        if (x < pos) {
          label_j = label_i;//
        } else {
            label_j = (caffe::caffe_rng_rand() %16+label_i-8)%(label_set.size()-1);
          //label_j = caffe::caffe_rng_rand() % (label_set.size()-1);
          if (label_j >= label_i) label_j ++;
        }
        //cout<<label_j<<endl;
        const vector<size_t>& sets = label_set[label_j];
        int j=sets[caffe::caffe_rng_rand() % sets.size()];

            char* FirstImagePixel = Pixels;
            // cout<<root_folder + lines[LineIndex].first<<endl;
            ReadImageToMemory(root_folder + lines[i].first, resize_height, resize_width, FirstImagePixel);  //8. add channel here

            //char *SecondImagePixel = Pixels + resize_width * resize_height;
            char *SecondImagePixel = Pixels + resize_width * resize_height * channel;       //10. add channel
            ReadImageToMemory(root_folder + lines[j].first, resize_height, resize_width, SecondImagePixel);  //9. add channel here

            // set image pair data
            // datum.set_data(Pixels, 2 * resize_height * resize_width);
            datum.set_data(Pixels, 2 * resize_height * resize_width * channel);     //11. correct

            // set label
            // for training, first 1000 pairs are true; for testing,first 1000 pairs are true
            // if (LineIndex<4000)   
            //train: 912,3000 true pairs, 81,1080 false pairs;
            //test: 35600 true pairs, 33500 false pairs
            // if (LineIndex<9123000)
            // {
            //     datum.set_label(1);
            // }
            // else
            // {
            //     datum.set_label(0);
            // }
                cout <<LineIndex<<":   "<< label_i<<"===="<<label_j<<endl;
            if (label_i  == label_j) {
              datum.set_label(1);
            } else {
              datum.set_label(0);
            }
            // printf("first index: %d, second index: %d, labels: %d \n", lines[LineIndex].second, lines[PairIndex].second, datum.label());

            // // serialize datum to string
            // datum.SerializeToString(&value);
            // int key_value = (int)(LineIndex);
            // _snprintf(key, kMaxKeyLength, "%08d", key_value);
            // string keystr(key);
            // cout << "label: " << datum.label() << ' ' << "key index: " << keystr << endl;
            // //sprintf_s(key, kMaxKeyLength, "%08d", LineIndex);     

            // db->Put(leveldb::WriteOptions(), std::string(key), value);

            datum.SerializeToString(&value);
            std::string key_str = caffe::format_int(LineIndex, 8);
            //db->Put(leveldb::WriteOptions(), key_str, value);
            txn->Put(key_str, value);

            if (LineIndex % 1000 == 0) {
              // Commit db
              txn->Commit();
              txn.reset(db->NewTransaction());
              LOG(INFO) << "Processed " << LineIndex << " files.";
            }
          // write the last batch
          if (LineIndex % 1000 != 0) {
            txn->Commit();
            LOG(INFO) << "Processed " << LineIndex << " files.";
          }

        }

        //delete db;
        delete[] Pixels;
    }
#endif
    return 0;
}
#endif