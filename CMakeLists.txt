cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(testvision)
add_executable(${PROJECT_NAME} main2.cpp)

#set(pybind11_DIR /path/to/pybind11/share/cmake/pybind11)
set(Torch_DIR /home/duwei/libtorch/share/cmake/Torch)
set(TorchVision_DIR /home/duwei/torchvision/share/cmake/TorchVision)
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)

include_directories("/home/duwei/torchvision/include")
include_directories("/home/duwei/libtorch/libtorch/include/torch/csrc/api/include/")
include_directories("/home/duwei/libtorch/libtorch/include/")

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Python3 REQUIRED)

# find_package (Python3 COMPONENTS Interpreter Development)
#target_link_libraries(${PROJECT_NAME} PUBLIC TorchVision::TorchVision)

#target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")
# add_compile_definitions(WITH_CUDA) #cuda bianyihao  keyi yong
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${OpenCV_LIBS} TorchVision::TorchVision /home/duwei/anaconda3/lib/libpython3.7m.so /home/duwei/torchvision/lib/libtorchvision.so /usr/lib/x86_64-linux-gnu/libtiff.so.5)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
