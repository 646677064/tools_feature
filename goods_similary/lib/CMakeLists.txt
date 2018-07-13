
#This part is used for compile faster_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)

#project (caffe_detect)

#add_definitions( -DUSE_OPENCV)
SET (SRC_LIST caffe_detect.cpp)

#AUX_SOURCE_DIRECTORY(. DIR_HELLO_SRCS)  
#ADD_LIBRARY(hello STATIC ${DIR_HELLO_SRCS})

# include_directories ( "${RFCN_DIR}//include"
#     "${RFCN_DIR}//.build_release/src/"
#     "${RFCN_DIR}//python"
#     "${RFCN_DIR}/lib" 
#     "${RFCN_DIR1}/lib/nms" 
#     "${PROJECT_SOURCE_DIR}/lib"
#     /usr/include
#     /usr/local/include 
#     /usr/local/cuda-7.5/include
#     /usr/local/cuda
#     /usr/lib
#     /usr/include/python2.7
#     /usr/lib/python2.7/dist-packages/numpy/core/include 
#     /usr/local/cuda/include )
message("4Found OpenCV-=-=-=lib: " ${OpenCV_INCLUDE_DIRS})
include_directories ( "${RFCN_DIR}/caffe/include"
    "${RFCN_DIR}/caffe/build/include/"
    "${RFCN_DIR}/caffe/python"
    "${RFCN_DIR}/py-R-FCN-master/lib" 
    "${RFCN_DIR}/lib/nms" 
    "${PROJECT_SOURCE_DIR}/lib"
    ${Caffe_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}
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
IF (caffe_LIBRARY)
    message(${caffe_LIBRARY})
    target_link_libraries(caffe_detect ${caffe_LIBRARY})
ENDIF(caffe_LIBRARY)

#message(${OpenCV_LIBS})
# target_link_libraries(caffe_detect "${PROJECT_SOURCE_DIR}/lib/libgpu_nms.so"
#     ${OpenCV_LIBS}
#     libglog.so
#     libboost_system.so
#     libboost_python.so
#     libglog.so
#     libpython2.7.so
#     )
target_link_libraries(caffe_detect "${PROJECT_SOURCE_DIR}/lib/libgpu_nms.so"
     libpython2.7.so
    ${Caffe_LINKER_LIBS}
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