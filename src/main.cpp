#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include <string>

#include "../include/utils.h"

void vj_detect(cv::Mat frame, cv::CascadeClassifier f_cascade);

int main(int argc, char* argv[]) {

    // Parse command line.
    std::string input_path{}, label_path{};
    parse_command_line(argc, argv, input_path, label_path);

    if (input_path.empty()) {
        std::cerr << "Error in parsing the command line...\n";
        return 1;
    }

    if (label_path.empty())
        std::cerr << "Info: label file not provided, IoU will not be computed.\n";

    // Print args found.
    std::cout << "INPUT FILE PATH " << input_path << "\n";
    if (!label_path.empty())
        std::cout << "LABEL FILE PATH " << label_path << "\n";

    // ----- FACE DETECTION -----
    cv::CascadeClassifier face_cascade;

    // Load the cascades.
    if (!face_cascade.load("../classifiers/haarcascade_frontalface_alt.xml")){

        std::cout << "Error loading face cascade\n";
        return -1;
    };

    cv::Mat img = cv::imread(input_path);
    // Detect and save the faces in a specific folder.
    vj_detect(img, face_cascade);

    // Call the python pipeline to classify the faces
    FILE* pipe = popen("python3 ../src/sender.py", "r"); // "r" to read
    if (!pipe) {
        std::cerr << "Error in opening the pipe" << std::endl;
        return -1;
    }

    std::vector<int> vals;
    int val;

    if (fscanf(pipe, "%d", &val) != 1){
        std::cerr << "Error in opening the pipe" << std::endl;
        pclose(pipe);
        return -1;
    }

    while (fscanf(pipe, "%d", &val) == 1) {  // Read a int in the pipe
        vals.push_back(val);
    }
    // Close the pipe
    pclose(pipe);  

    if (val == 0) {
        std::cerr << "Recivied 0 possible default" << std::endl;
    } else {
        for (size_t i = 0; i < vals.size(); i++)
        {
            std::cout << "Recivied val:" << vals[i] << std::endl;
        }
         
    }
    return 0;
}

// Detection function using the ViolaJones algorithm.
void vj_detect(cv::Mat frame , cv::CascadeClassifier f_cascade){

    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

    // Plotting the image equalized.
    cv::namedWindow("Window");
    cv::imshow("Window", frame_gray);
    cv::waitKey(0);

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    f_cascade.detectMultiScale(frame_gray, faces);

    // Folder path in which will be saved the images.
    std::string folder_path_cropped_imgs = "../cropped_imgs/";
    // Vector of cropped images and vector of bounding boxes.
    std::vector<cv::Mat>  cropped_imgs; 

    for (size_t i = 0; i < faces.size(); i++){

        // Cropping the detected faces.
        cv::Mat faceROI = frame(faces[i]);
        cropped_imgs.push_back(faceROI.clone());
        // Saving the cropped images.
        cv::imwrite(folder_path_cropped_imgs + "cut_" + std::to_string(i)+".png", cropped_imgs[i]);        
    }

    // Draw the box over the detection.
    for (size_t i = 0; i < faces.size(); i++){
        cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 255), 4);      
    }
    
    // Show the images detected.
    cv::imshow("Window", frame);
    cv::waitKey(0);

    for (size_t i = 0; i < cropped_imgs.size(); i++){

        std::string window_name = "Window " + std::to_string(i);
        cv::imshow(window_name, cropped_imgs[i]);
    }

    cv::waitKey(0);
}