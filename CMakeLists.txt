cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(CVFinal)

include_directories(include)

add_library(CVFinal
    include/utils.h
    lib/utils.cpp
    include/performance_metrics.h
    lib/performance_metrics.cpp
    include/face_detector.h
    lib/face_detector.cpp
    )

add_executable(out
    src/main.cpp
    )

target_link_libraries(out CVFinal)

find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV library found.")
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(out
        opencv_highgui
        opencv_core
        opencv_imgproc
        opencv_objdetect
        )
else()
    message(FATAL_ERROR "OpenCV library not found.")
endif()
