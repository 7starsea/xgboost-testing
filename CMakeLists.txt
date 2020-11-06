cmake_minimum_required(VERSION 3.13)
project(xgb-shannon LANGUAGES C CXX VERSION 0.0.1)

#set(xgb_root_dir xgboost)
set(xgb_root_dir /home/aimin/Projects/github/xgboost)
IF(IS_DIRECTORY ${xgb_root_dir})
	message(STATUS "xgb_root_dir: ${xgb_root_dir}")
else()
	message(FATAL_ERROR "please set xgb_root_dir")
endif()
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
include_directories(${xgb_root_dir}/include)
find_library(xgb_lib xgboost PATHS ${xgb_root_dir}/lib)

# # used for xgbboost python version v1.2.1
# find_library(xgb_lib xgboost PATHS anaconda3/envs/cuda/lib/python3.8/site-packages/xgboost/lib)



find_package (Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND AND Python3_Interpreter_FOUND)
    message(STATUS "Python3_INCLUDE_DIRS: " ${Python3_INCLUDE_DIRS})
    message(STATUS "Python3_LIBRARIES: " ${Python3_LIBRARIES})
    message(STATUS "Python3_EXECUTABLE: " ${Python3_EXECUTABLE})
    message(STATUS "Python3_VERSION: " ${Python3_VERSION})

    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
    set(PYTHONLIBS_VERSION_STRING ${Python3_VERSION})
else()
    message(FATAL_ERROR "FAILED TO FIND PYTHON3")
endif()




message(STATUS "XgbBoost lib: ${xgb_lib}")

INCLUDE_DIRECTORIES(SYSTEM third-party/pybind11-2.5.0/include)


Python3_add_library(XgbShannon ex_xgb.cpp)
target_link_libraries(XgbShannon PUBLIC ${xgb_lib})
