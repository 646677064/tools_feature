# Find the OpenCV (and Lapack) libraries
#
# The following variables are optionally searched for defaults
#  OpenCV_ROOT_DIR:            Base directory where all OpenCV components are found
#
# The following are set after configuration is done:
#  OpenCV_FOUND
#  OpenCV_INCLUDE_DIRS
#  OpenCV_LIBRARIES
#  OpenCV_LIBRARYRARY_DIRS

set(OpenCV_INCLUDE_SEARCH_PATHS
  /usr/include/OpenCV
  /usr/include/OpenCV-base
  /usr/include/
  /usr/local/opencv_3_1_0/include
  $ENV{OpenCV_ROOT_DIR}
  $ENV{OpenCV_ROOT_DIR}/include
)

set(OpenCV_LIB_SEARCH_PATHS
  /usr/lib/OpenCV
  /usr/lib/OpenCV-base
  /usr/lib
  /usr/lib64
  /usr/local/opencv_3_1_0/lib
  $ENV{OpenCV_ROOT_DIR}
  $ENV{OpenCV_ROOT_DIR}/lib
)

message(STATUS "1======${OpenCV_INCLUDE_SEARCH_PATHS}")
find_path(OpenCV_CBLAS_INCLUDE_DIR   NAMES opencv2/core/core.hpp   PATHS ${OpenCV_INCLUDE_SEARCH_PATHS})
find_path(OpenCV_CLAPACK_INCLUDE_DIR NAMES opencv2/highgui/highgui.hpp PATHS ${OpenCV_INCLUDE_SEARCH_PATHS})
message(STATUS "2======${OpenCV_CBLAS_INCLUDE_DIR}")

find_library(OpenCV_LAPACK_LIBRARY NAMES  opencv_videoio PATHS ${OpenCV_LIB_SEARCH_PATHS})
find_library(OpenCV_CBLAS_LIBRARY1 NAMES    opencv_core PATHS ${OpenCV_LIB_SEARCH_PATHS})
find_library(OpenCV_CBLAS_LIBRARY2 NAMES    opencv_imgproc  PATHS ${OpenCV_LIB_SEARCH_PATHS})
find_library(OpenCV_CBLAS_LIBRARY3 NAMES    opencv_highgui  PATHS ${OpenCV_LIB_SEARCH_PATHS})
find_library(OpenCV_BLAS_LIBRARY NAMES   opencv_imgcodecs PATHS ${OpenCV_LIB_SEARCH_PATHS})

set(LOOKED_FOR 
  OpenCV_CBLAS_INCLUDE_DIR
  OpenCV_CLAPACK_INCLUDE_DIR

  OpenCV_CBLAS_LIBRARY1
  OpenCV_BLAS_LIBRARY
  OpenCV_LAPACK_LIBRARY
  OpenCV_CBLAS_LIBRARY2
  OpenCV_CBLAS_LIBRARY3
)
message(STATUS "${LOOKED_FOR}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCV DEFAULT_MSG ${LOOKED_FOR})

if(OPENCV_FOUND)
  set(OpenCV_INCLUDE_DIRS ${OpenCV_CBLAS_INCLUDE_DIR} ${OpenCV_CLAPACK_INCLUDE_DIR})
  set(OpenCV_LIBS ${OpenCV_LAPACK_LIBRARY} ${OpenCV_CBLAS_LIBRARY1} ${OpenCV_CBLAS_LIBRARY2} ${OpenCV_CBLAS_LIBRARY3} ${OpenCV_BLAS_LIBRARY})
  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "3===Found OpenCV (include: ${OpenCV_INCLUDE_DIRS}, library: ${OpenCV_LIBS})")
endif(OPENCV_FOUND)

