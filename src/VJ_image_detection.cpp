#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include "../include/utils.h"

void vj_detect(cv::Mat frame, std::vector<cv::CascadeClassifier>& f_cascade);

const std::vector<std::string> filepaths= {
"../classifiers/haarcascade_eye_tree_eyeglasses.xml",
"../classifiers/haarcascade_eye.xml",
"../classifiers/haarcascade_frontalface_alt_tree.xml",
"../classifiers/haarcascade_frontalface_alt.xml",
"../classifiers/haarcascade_frontalface_alt2.xml",
"../classifiers/haarcascade_frontalface_default.xml",
"../classifiers/haarcascade_lefteye_2splits.xml",
"../classifiers/haarcascade_profileface.xml",
"../classifiers/haarcascade_righteye_2splits.xml",
};


int main(void){

    std::vector<cv::CascadeClassifier> face_cascades (filepaths.size());

    // Load the cascades.
    for(int i = 0; i < filepaths.size(); i++ ){
        if (!face_cascades[i].load(filepaths[i])){
            std::cout << "Error loading face cascade\n";
            return -1;
        }
    }

    std::string img_path = "../dataset_detection/images/disgust_1.jpg";
    cv::Mat img = cv::imread(img_path);
    // Detect and save the faces in a specific folder.
    vj_detect(img, face_cascades);

    // Call the python pipeline to classify the faces
    /*FILE* pipe = popen("python3 sender.py", "r"); // "r" to read
    if (!pipe) {
        std::cerr << "Error in opening the pipe" << std::endl;
        return -1;
    }

    std::vector<int> vals;
    int val;
s file
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
void vj_detect(cv::Mat frame , std::vector<cv::CascadeClassifier>& f_cascades){

    std::string img_path = "../dataset_detection/labels/disgust_1.txt";
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


    cv::CascadeClassifier final_classifier;
    int score = 0;
    int best_score = 0;
    int best_index = 0;
    std::vector<double> scores (f_cascades.size());
    for(int i = 0; i < f_cascades.size(); i++){
        std::cout<< "Testing classifier number: "<<i<<std::endl;
        faces.clear();
        f_cascades[i].detectMultiScale(
            frame_gray,
            faces
        );

        // Loop through each detected face
        for(size_t j = 0; j < faces.size(); j++){
            // Print the confidence score (level weight) for the corresponding face
            score += calculateBlurScore(frame, faces[j]) * faces[j].area();
            
        }
        if (score > best_score){
            best_index = i;
            best_score = score;
        }
    }

    std::cout<<"Best classifier is: "<<filepaths[best_index]<<std::endl;    
    std::cout<<"  with score: "<<best_score<<std::endl;

    final_classifier = f_cascades[best_index];

    faces.clear();
    final_classifier.detectMultiScale(
        frame_gray,
        faces
    );
    std::cout << "Found " << faces.size() << " faces." << std::endl;

    std::vector<cv::Rect> filtered_faces; 
    int min_score = 10000;
    score = 0;
    for(size_t j = 0; j < faces.size(); j++){
        // Print the confidence score (level weight) for the corresponding face
        score = calculateBlurScore(frame, faces[j]) * faces[j].area();
        std::cout << "Face " << j
                << " -> Confidence Score: " << score <<std::endl;
        scores[j] += score; 
        // Here we filter the detection: if they're both not defined and small we filter out.
        if(score > min_score){
            filtered_faces.push_back(faces[j]);
        }
    }


    std::cout<<"Detected rectangles position and size: "<<std::endl;
    printRectDetails(filtered_faces);
    std::cout<<std::endl;
    std::vector<cv::Rect> label_rects = compute_rectangles(img_path, frame.cols, frame.rows);
    std::cout<<"Label rectangles position and size: "<<std::endl;
    printRectDetails(label_rects);

    if(filtered_faces.size() == 0){
        exit(1);
    }
    print_IOU(img_path, filtered_faces, frame.cols, frame.rows);
    
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