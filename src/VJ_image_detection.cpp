#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include "../include/utils.h"

void vj_detect(cv::Mat frame, cv::CascadeClassifier f_cascade);

int main(void){

    cv::CascadeClassifier face_cascade;

    // Load the cascades.
    if (!face_cascade.load("../classifiers/haarcascade_frontalface_alt.xml")){

        std::cout << "Error loading face cascade\n";
        return -1;
    };

    std::string img_path = "../dataset_detection/images/happy_1.jpg";
    cv::Mat img = cv::imread(img_path);
    // Detect and save the faces in a specific folder.
    vj_detect(img, face_cascade);

    // Call the python pipeline to classify the faces
    /*FILE* pipe = popen("python3 sender.py", "r"); // "r" to read
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
*/
    return 0;
}

// Detection function using the ViolaJones algorithm.
void vj_detect(cv::Mat frame , cv::CascadeClassifier f_cascade){

    std::string img_path = "../dataset_detection/labels/happy_1.txt";
    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

    // Plotting the image equalized.
    cv::namedWindow("Window",cv::WINDOW_NORMAL);
    cv::imshow("Window", frame);
    cv::waitKey(0);

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    std::vector<int> rejectLevels;   
    std::vector<double> levelWeights;   // This will hold the confidence scores
    std::vector<double> blurScore;
    f_cascade.detectMultiScale(
        frame_gray,
        faces,
        rejectLevels,
        levelWeights,
        1.1, // scaleFactor
        5,   // minNeighbors
        0,   // flags
        cv::Size(70,70), // minSize
        cv::Size(),       // maxSize
        true              // outputRejectLevels -> SET TO TRUE
    );

    std::cout << "Found " << faces.size() << " faces." << std::endl;

    // Loop through each detected face
    std::vector<cv::Rect> filtered_faces; 
    int min_score = 1000000;
    int score = 0;
    for(size_t i = 0; i < faces.size(); i++){
        // Print the confidence score (level weight) for the corresponding face
        score = calculateBlurScore(frame, faces[i]) * faces[i].area();
        std::cout << "Face " << i 
                  << " -> Confidence Score: " << score <<std::endl;
        // Here we filter the detection: if they're both not defined and small we filter out.
        if(score > min_score){
            filtered_faces.push_back(faces[i]);
        }
    }
    std::cout<<std::endl;


    printRectDetails(filtered_faces);
    print_IOU(img_path, filtered_faces, frame.cols, frame.rows);
    std::vector<cv::Rect> label_rects = compute_rectangles(img_path, frame.cols, frame.rows);

    
    /*
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
    }*/

    // Draw the box over the detection.
    for (size_t i = 0; i < filtered_faces.size(); i++){
        cv::rectangle(frame, filtered_faces[i], cv::Scalar(255, 0, 255), 4);      
    }
    // Draw the label box.
    for (size_t i = 0; i < label_rects.size(); i++){
        cv::rectangle(frame, label_rects[i], cv::Scalar(255, 255, 0), 4);      
    }
    
    // Show the images detected.

    double scale = 0.5;
    cv::resize(frame, frame, cv::Size(), scale, scale);
    cv::imshow("Window", frame);
    cv::waitKey(0);

    /*
    for (size_t i = 0; i < cropped_imgs.size(); i++){

        std::string window_name = "Window " + std::to_string(i);
        cv::imshow(window_name, cropped_imgs[i]);
    }

    cv::waitKey(0);
    */
}