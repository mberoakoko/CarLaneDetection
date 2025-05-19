cmake_minimum_required(VERSION 3.27)
project(CarLane)

set(CMAKE_CXX_STANDARD 23)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(CarLane main.cpp
        src/utils/histogram.hpp
        src/utils/feature_extraction.hpp)
target_link_libraries( CarLane ${OpenCV_LIBS} )