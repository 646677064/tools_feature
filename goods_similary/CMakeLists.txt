
#This part is used for compile faster_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)
SET( CMAKE_VERBOSE_MAKEFILE on )
#SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH}
project (main_demo)
set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -msse -msse2 -msse3 -mmmx -march=native")
#set(CMAKE_BUILD_TYPE Debug CACHE STRING "set build type to debug")
#DUSE_CUDNN
add_definitions( -DUSE_OPENCV -DUSE_CUDNN)


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(RFCN_DIR /home/liushuai/tiannuocaffe/py-rfcn-gpu/)
#set(RFCN_DIR /mnt/storage/liushuai/work/caffe-reid)

# include_directories ( "${RFCN_DIR}//include"
#     "${RFCN_DIR}//build/include/"
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
include_directories ( "${RFCN_DIR}/caffe/include"
    "${RFCN_DIR}/caffe/.build_release/src/"
    "${RFCN_DIR}/caffe/python"
    "${RFCN_DIR}/lib" 
    "${RFCN_DIR}/lib/nms" 
    "${PROJECT_SOURCE_DIR}/lib"
    /usr/include
    /usr/local/include 
    /usr/local/cuda-8.0/include
    /usr/local/cuda
    /usr/lib
    /usr/include/python2.7
    /usr/lib/python2.7/dist-packages/numpy/core/include 
    /usr/local/cuda/include )

set(BLAS open)
set(BLAS_INCLUDE /opt/OpenBLAS/include)
set(BLAS_LIB /opt/OpenBLAS/lib)
set(nms_DIR ${RFCN_DIR}/lib/nms)
set(caffe_DIR ${RFCN_DIR}/caffe/build/lib)
#set(caffe_DIR ${RFCN_DIR}//build/lib)
set(python_DIR ${RFCN_DIR}/caffe/python)
set(DETECT_DIR ${PROJECT_SOURCE_DIR}/lib)

message(${RFCN_DIR})
message(${nms_DIR})
message(${caffe_DIR})
message(${python_DIR})
message(${DETECT_DIR})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
include(cmake/Dependencies.cmake)
find_package( OpenCV REQUIRED )
#if(OpenCV_FOUND)
   message("5Found OpenCV-=-=-=")
   message("5Includes: " ${OpenCV_INCLUDE_DIRS})
   include_directories( ${OpenCV_INCLUDE_DIRS})
#endif(OpenCV_FOUND)
#SUBDIRS(hello sample)
ADD_SUBDIRECTORY(lib)

add_executable(main main.cpp)

link_directories( ${DETECT_DIR}
    ${caffe_DIR}
    ${python_DIR}
    /opt/OpenBLAS/include
    /usr/lib
    /usr/local/lib
    /usr/local/cuda-8.0/lib64)

    find_package(OpenBLAS REQUIRED)
  #message(STATUS "66 OpenBLAS found=== (${OpenBLAS_INCLUDE_DIR}) {${OpenBLAS_LIB}}")
    include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})
#FIND_LIBRARY(rfcn_LIBRARY NAMES libcaffe_detect.so PATHS ${DETECT_DIR})
#IF (rfcn_LIBRARY)
#    message(${rfcn_LIBRARY})
#    target_link_libraries(main ${rfcn_LIBRARY})
#ENDIF(rfcn_LIBRARY)


#FIND_LIBRARY(nms_LIBRARY NAMES gpu_nms.so PATHS ${nms_DIR})
#IF (nms_LIBRARY)
 #   message(${nms_LIBRARY})
 #   target_link_libraries(main ${nms_LIBRARY})
#ENDIF(nms_LIBRARY)

#FIND_LIBRARY(caffe_LIBRARY NAMES libcaffe.so PATHS ${caffe_DIR})
#IF (caffe_LIBRARY)
#   message(${caffe_LIBRARY})
#    target_link_libraries(main ${caffe_LIBRARY})
#ENDIF(caffe_LIBRARY)



#set(opencv_DIR /usr/local/lib)
#FIND_LIBRARY(opencv_LIBRARY NAMES libopencv_videoio.so libopencv_video.so libopencv_objdetect.so libopencv_highgui.so libopencv_core.so libopencv_imgcodecs.so libopencv_imgproc.so PATHS ${opencv_DIR})
#IF (opencv_LIBRARY)
#    message(${opencv_LIBRARY})
#    target_link_libraries(main opencv_LIBRARY)
#ENDIF(opencv_LIBRARY)

#    opencv_videoio
#    opencv_video
#   opencv_objdetect
 #   opencv_highgui 
 #   opencv_core
 #   opencv_imgcodecs
  #  opencv_imgproc
  
FIND_LIBRARY(caffe_LIBRARY NAMES libcaffe.so PATHS ${caffe_DIR})
IF (caffe_LIBRARY)
    message("aaaa" ${caffe_LIBRARY})
    target_link_libraries(main ${caffe_LIBRARY})
ENDIF(caffe_LIBRARY)
target_link_libraries(main ${Caffe_LINKER_LIBS})
target_link_libraries(main caffe_detect
    ${OpenCV_LIBS}
    pthread
    /opt/OpenBLAS/lib/libopenblas.so
    ${caffe_LIBRARY}
    ${Caffe_LINKER_LIBS}
    )

#target_link_libraries(main "${PROJECT_SOURCE_DIR}/lib/libgpu_nms.so" 
 #   caffe_detect
 #   ${OpenCV_LIBS}
 #   libglog.so
 #   libboost_system.so
 #   libboost_python.so
 #   libglog.so
 #   libpython2.7.so
 #   )
   
 #   libopencv_imgcodecs.so

#target_link_libraries(main libfaster_rcnn.so
#    "${PROJECT_SOURCE_DIR}/../../py-R-FCN-master/caffe/.build_release/lib/libcaffe.so"
#    "${PROJECT_SOURCE_DIR}/../../py-R-FCN-master/lib/nms/gpu_nms.so "
#   /share/apps/local/lib/libopencv_highgui.so 
#    /share/apps/local/lib/libopencv_core.so 
#    /share/apps/local/lib/libopencv_imgproc.so 
#    /share/apps/local/lib/libopencv_imgcodecs.so
#    /share/apps/local/lib/libglog.so
#    /share/apps/local/lib/libboost_system.so
#    /share/apps/local/lib/libboost_python.so
#    /share/apps/local/lib/libglog.so
#    /opt/rh/python27/root/usr/lib64/libpython2.7.so
#    )

