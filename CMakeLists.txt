cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_BUILD_TYPE Debug)

project(llama3.cuda LANGUAGES CUDA)

add_definitions(-DUSE_CUBLAS=1)

add_executable(runcuda llama3.cu)
target_link_libraries(runcuda -lm -lcublas)