cmake_minimum_required(VERSION 3.1)
project(custom_op_fma LANGUAGES CXX)

if (DEFINED ONNXRUNTIME_TOP_DIR)
	message(STATUS "ONNXRUNTIME_TOP_DIR: ${ONNXRUNTIME_TOP_DIR}")
else ()
	set(ONNXRUNTIME_TOP_DIR onnxruntime)
	message(STATUS "ONNXRUNTIME_TOP_DIR: ${ONNXRUNTIME_TOP_DIR}")
endif ()

find_path(ONNXRUNTIME_INCLUDE_DIR
	NAMES
		onnxruntime_c_api.h
		onnxruntime_cxx_api.h
		onnxruntime_cxx_inline.h
	HINTS ${ONNXRUNTIME_TOP_DIR}
	PATH_SUFFIXES include/onnxruntime/core/session
)

if (${ONNXRUNTIME_INCLUDE_DIR} STREQUAL "ONNXRUNTIME_INCLUDE_DIR-NOTFOUND")
    message(ERROR " ONNXRuntime not found. set ONNXRUNTIME_TOP_DIR correctly")
else ()
    message(STATUS "ONNXRUNTIME_INCLUDE_DIR: ${ONNXRUNTIME_INCLUDE_DIR}")
endif ()

add_library(my_custom_t SHARED fma_use_t.cpp)
target_include_directories(my_custom_t SYSTEM PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
target_compile_features(my_custom_t PUBLIC cxx_std_17)
