#include <iostream>
#include <unistd.h>
#include <fstream>
#include <ostream>
#include <sstream>
#include <numeric>
#include "../include/utils.h"
#include "../include/performance_metrics.h"


// MEMBER FUNCTIONS

// This member function compute the IOUs of the detected faces.
std::vector<float> PerformanceMetrics::get_label_IOUs(){
    float current_IoU;
    std::vector<float> IOUs( face_labels.size(), 0.0f); 

    for(int i = 0; i <  face_labels.size(); i++){
        for(int j = 0; j <  detected_faces.size(); j++){
            current_IoU = compute_IOU( face_labels[i],  detected_faces[j]);
            if(current_IoU > IOUs[i]){
                IOUs[i] = current_IoU;
            }
        }
    }
    return IOUs;
}

// Function to write in a file and computing in terminal the metrics for the scenepath.
void PerformanceMetrics::print_metrics(){
    // Print in a file and in the terminal
    std::ofstream outfile("metrics.txt", std::ios::app);
    if (outfile.is_open())
    {
        //std::cout <<  path_true_labels << "Metrics : \n\n";
        //outfile <<  path_true_labels << " Metrics : \n";
        outfile << "IOUs of labels : \n";
        std::vector<float> IOUs =  get_label_IOUs();
        for (int i = 0; i <  face_labels.size(); i++)
        {
            outfile<< "IOU of box number "<< i << " is: "<< IOUs[i] <<std::endl;
        }
        outfile << "\n";
        outfile.close();
    }
    else
    {
        std::cerr << "Impossibile to open the file\n";
    }
}

// Function to compute the mean over IOUs (MIOU).
float PerformanceMetrics::compute_MIOU(){

    std::vector<float> IOUs =  get_label_IOUs();

    // Handle the edge case of an empty vector to prevent division by zero.
    if (IOUs.empty()) {
        return 0.0f;
    }

    // Calculate the sum of all elements in the vector.
    //    std::accumulate(begin, end, initial_value)
    float total_iou = std::accumulate(IOUs.begin(), IOUs.end(), 0.0f);

    // Divide the sum by the number of elements to get the mean.
    return total_iou / IOUs.size();
}


// HELPER FUNCTIONS: 

// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(cv::Rect& box1, cv::Rect& box2){
    // Define the variable to store the areas of intersection, union and the respective IoU.
    double areas_int;
    double areas_union;

    // Compute intersection union of boxes. 
    cv::Rect intersect = box1 & box2;
    areas_int = intersect.area();
    areas_union = box1.area() + box2.area() - areas_int;

    // Compute and return the IoU.
    float IoU = areas_int / areas_union;
    return IoU;
}
