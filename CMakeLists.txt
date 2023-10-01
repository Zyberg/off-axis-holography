cmake_minimum_required(VERSION 3.5)
project(h_reconstruction LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(h_reconstruction main.cu
        src/HologramProcessor.cpp
        src/HologramProcessor.h)

set_target_properties(h_reconstruction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


target_link_libraries(h_reconstruction ${OpenCV_LIBS})

