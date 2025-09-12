#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include "../include/utils.h"
#include "opencv2/core/types.hpp"


/*
    This class relize the purpose of summing up all the performance metrics functions, data and 
    functionalities. 
*/
class PerformanceMetrics{

    public:

        // CONSTRUCTORS:

        // Main constructor: initializes both the detected faces positions and label faces positions.
        PerformanceMetrics(const std::vector<cv::Rect>& detected_faces, const std::vector<cv::Rect>& face_labels) : 
             detected_faces(detected_faces),  face_labels(face_labels){}

        //MEMBER FUNCTIONS:

        // This member function compute the IOUs of the detected faces.
        std::vector<float> get_label_IOUs();

        // Function to compute the mean over IOUs of recatangles (MIOU).
        float compute_MIOU();

        // Function to write in a file and computing in terminal the metrics for the scenepath.
        void print_metrics(std::string file_name = "");

    private:

        // DATA MEMEBERS: 

        // Vectors in which memorize the coordinate of the read values from the label txt file.
        std::vector<cv::Rect> detected_faces;
        std::vector<cv::Rect> face_labels;

};

// HELPER FUNCTIONS: 

// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(cv::Rect& box1, cv::Rect& box2);



#endif