# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support

find_package( Protobuf REQUIRED )
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${PROTOBUF_LIBRARIES})

# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
# and should be installed separately as in: sudo apt-get install protobuf-compiler
if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()


# place where to generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/proto")
include_directories("${PROJECT_BINARY_DIR}/include")

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)
