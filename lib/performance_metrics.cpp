#include "../include/performance_metrics.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <numeric>

#define BLUE  "\033[34m"
#define RESET   "\033[0m"

namespace fs = std::filesystem;


// -------------- MEMBER FUNCTIONS --------------
// This member function compute the IOUs of the detected faces.
std::vector<float> PerformanceMetrics::get_label_IOUs(std::vector<cv::Rect>& detection, std::vector<cv::Rect>& labels){
    float current_IoU;
    std::vector<float> IOUs( labels.size(), 0.0f); 

    for(int i = 0; i <  labels.size(); i++){
        for(int j = 0; j <  detection.size(); j++){
            current_IoU = compute_IOU( labels[i],  detection[j]);
            if(current_IoU > IOUs[i]){
                IOUs[i] = current_IoU;
            }
        }
    }
    return IOUs;
}

// Function to write in a file and computing in terminal the metrics for the scenepath.
void PerformanceMetrics::print_metrics(bool verbose){
    // Print in a file and in the terminal
    std::ofstream outfile(metrics_file_path, std::ios::app);
    std::vector<float> all_IOUs;
    float precision = 0.0f;
    float recall = 0.0f;
    float avg_IOU = 0.0f;
    int count_detected = 0;
    if (outfile.is_open()) {
        //std::cout <<  path_true_labels << "Metrics : \n\n";
        //outfile <<  path_true_labels << " Metrics : \n";
        outfile << "IOUs of labels: \n";
        for(int j = 0; j < face_labels.size(); j++){
            outfile << "\nIOUs of image " << j<< ": \n";

            count_detected += detected_faces[j].size();
            std::vector<float> IOUs =  get_label_IOUs(detected_faces[j], face_labels[j]);
            all_IOUs.insert(all_IOUs.end(), IOUs.begin(), IOUs.end());

            for (int i = 0; i <  face_labels[j].size(); i++)
            {
                outfile<< "    IOU of box number "<< i << " is: "<< IOUs[i] <<std::endl;
            }
        }
        outfile <<std::endl;
        avg_IOU = compute_MIOU(all_IOUs);
        precision = compute_precision(all_IOUs, IOU_THRESHOLD, count_detected);
        recall = compute_recall(all_IOUs, IOU_THRESHOLD);
        outfile<<"The precision over analized images is: "<<precision<<std::endl;
        outfile<<"The recall over analized images is: "<<recall<<std::endl;
        outfile<<"The avarage IOUs is: "<<avg_IOU<<std::endl;
        outfile.close();
    }
    else
    {
        std::cerr << "Impossibile to open the file\n";
    }
    if(verbose){
        std::cout<<std::endl<<BLUE<<"The avarage IOUs is: "<< RESET << avg_IOU<<std::endl;
        std::cout<< BLUE <<"The precision over analized images is: "<< RESET <<precision<<std::endl;
        std::cout<< BLUE <<"The recall over analized images is: "<< RESET <<recall<<std::endl; 
    }
}


void PerformanceMetrics::clean_metrics(){
    try {
        // The remove function returns true if a file was deleted, false otherwise
        fs::remove(metrics_file_path);
    } catch (const fs::filesystem_error &e) {
        // This catch block handles errors like permission issues
        std::cerr << "Error deleting file: " << e.what() << std::endl;
        return;
    }
}



void PerformanceMetrics::add_image_detections(std::vector<cv::Rect>& detection, std::vector<cv::Rect>& labels){
    face_labels.push_back(labels);
    detected_faces.push_back(detection);
}


// -------------- HELPER FUNCTIONS --------------

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


// Function to compute the mean over IOUs (MIOU).
float compute_MIOU(std::vector<float> IOUs){

    // Handle the edge case of an empty vector to prevent division by zero.
    if (IOUs.empty()) {
        return 0.0f;
    }

    int detection_count = 0;
    for(int i = 0; i < IOUs.size(); i++){
        if(IOUs[i] > 0){
            detection_count ++;
        }
    }

    // Calculate the sum of all elements in the vector.
    //    std::accumulate(begin, end, initial_value)
    float total_iou = std::accumulate(IOUs.begin(), IOUs.end(), 0.0f);

    // Divide the sum by the number of elements to get the mean.
    return total_iou / detection_count;
}


// Function to compute the precision.
float compute_precision(std::vector<float> all_IOUs, float threshold, int all_detections){
    float correct_detections = 0;
    for(int i = 0; i < all_IOUs.size(); i++){
        if(all_IOUs[i] > threshold){
            correct_detections ++;
        }
    }
    return (correct_detections/all_detections);
}


// Function to compute the recall.
float compute_recall(std::vector<float> all_IOUs, float threshold){
    float correct_detections = 0;
    float all_labels = all_IOUs.size();
    for(int i = 0; i < all_labels; i++){
        if(all_IOUs[i] > threshold){
            correct_detections ++;
        }
    }
    return (correct_detections/all_labels);  
}
