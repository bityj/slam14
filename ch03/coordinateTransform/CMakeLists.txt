cmake_minimum_required(VERSION 3.24)
project(coordinateTransform)

set(CMAKE_CXX_STANDARD 17)
#include_directories("/usr/include/eigen3")
add_executable(coordinateTransform coordinateTransform.cpp)
