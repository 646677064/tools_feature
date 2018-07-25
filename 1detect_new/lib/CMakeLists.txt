
#This part is used for compile faster_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)

#project (caffe_detect)

#add_definitions( -DUSE_OPENCV)
SET (SRC_LIST set_python.cpp
    caffe_detect.cpp)

#AUX_SOURCE_DIRECTORY(. DIR_HELLO_SRCS)  
#ADD_LIBRARY(hello STATIC ${DIR_HELLO_SRCS})


message(STATUS "777=====OpenCV found (${OpenCV_INCLUDE_DIRS})")
message(STATUS "777=====OpenCV found (${OpenCV_LIBS})")
include_directories ( "${RFCN_DIR}/caffe/include"
    "${RFCN_DIR}/caffe/build/src/"
    "${RFCN_DIR}/caffe/python"
    "${RFCN_DIR}/py-rfcn-gpu/caffe/lib" 
    "${RFCN_DIR}/lib/nms" 
    "${PROJECT_SOURCE_DIR}/lib"
    /usr/include
    /usr/local/include 
    /usr/local/cuda-8.0/include
    /usr/local/cuda
    /usr/lib
    /usr/include/python2.7
    /usr/lib/python2.7/dist-packages/numpy/core/include 
    /usr/local/cuda/include 
    ${OpenCV_INCLUDE_DIRS})


add_library(caffe_detect SHARED ${SRC_LIST})

FIND_LIBRARY(caffe_LIBRARY NAMES libcaffe.so PATHS ${caffe_DIR})
    message(STATUS "caffe===${caffe_LIBRARY}")
IF (caffe_LIBRARY)
    target_link_libraries(caffe_detect ${caffe_LIBRARY})
ENDIF(caffe_LIBRARY)

target_link_libraries(caffe_detect "${PROJECT_SOURCE_DIR}/lib/libgpu_nms.so"
    ${OpenCV_LIBS}
    libglog.so
    libboost_system.so
    libboost_python.so
    libglog.so
    libpython2.7.so
    /usr/local/opencv_3_1_0/lib/libopencv_imgproc.so
    /usr/local/opencv_3_1_0/lib/libopencv_highgui.so
    )

#ADD_LIBRARY(hello_static STATIC ${DIR_HELLO_SRCS})  
#SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")  
#SET_TARGET_PROPERTIES(hello_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)  
  
#ADD_LIBRARY(hello_shared SHARED ${DIR_HELLO_SRCS})  
#SET_TARGET_PROPERTIES(hello_shared PROPERTIES OUTPUT_NAME "hello")  
#SET_TARGET_PROPERTIES(hello_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)  
#INSTALL(TARGETS sample RUNTIME DESTINATION bin)  #${PREFIX}/bin camke -DCMAKE_INSTALL_PREFIX:PATH="${CMAKE_INSTALL_PREFIX}" 

#  
# configuration to find DB_CXX library  
#  
  
# Include dir  
#find_path(THREAD_DB_INCLUDE_DIR  
#  NAMES thread_db.h  
#  PATHS /usr/include/ /usr/local/include/  
#)  
  
## Finally the library itself  
#find_library(THREAD_DB_LIBRARY  
#  NAMES  thread_db  
#  PATHS /usr/lib/ /usr/local/lib/  
#)