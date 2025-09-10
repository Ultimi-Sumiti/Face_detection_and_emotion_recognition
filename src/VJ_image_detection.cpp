#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include "../include/utils.h"

std::vector<cv::Rect> vj_detect(std::string& filename);

const std::vector<std::string> file_paths= {
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

const std::string image_dir = "../dataset_detection/images/";

const std::string label_dir = "../dataset_detection/labels/";

const std::string image_extension = ".jpg";

const std::string label_extension = ".txt";

int main(void){

    std::string img_name = "sad_2";
    // Detect and save the faces in a specific folder.
    vj_detect(img_name);

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
std::vector<cv::Rect> vj_detect(std::string& filename){
    std::string image_path = image_dir + filename + image_extension;
    std::string label_path = label_dir + filename + label_extension;

    cv::Mat frame = cv::imread(image_path);

    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

    // Plotting the image equalized.
    cv::namedWindow("Window",cv::WINDOW_NORMAL);
    cv::imshow("Window", frame);
    cv::waitKey(0);

    // Load the cascades.
    std::vector<cv::CascadeClassifier> f_cascades (file_paths.size());
    for(int i = 0; i < file_paths.size(); i++ ){
        if (!f_cascades[i].load(file_paths[i])){
            std::cout << "Error loading face cascade\n";
            exit(1);
        }
    }

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    std::vector<int> rejectLevels;   
    std::vector<double> levelWeights;   // This will hold the confidence scores
    std::vector<double> blurScore;


    cv::CascadeClassifier final_classifier;
    int score = 0;
    std::vector<cv::Rect> filtered_faces; 
    std::vector<cv::Rect> best_detections; 
    int min_score = 1000000;
    int best_score = 0;
    int best_index = 0;
    std::vector<float> IOUs (f_cascades.size());
    float best_MIOU = 0.0f;
    float curr_MIOU = 0.0f;

    // Checking all possible faces cascades. 
    for(int i = 0; i < f_cascades.size(); i++){
        std::cout<<std::endl<< "Testing classifier number: "<<i<<std::endl;
        faces.clear();
        f_cascades[i].detectMultiScale(
            frame_gray,
            faces,
            rejectLevels,
            levelWeights,
            1.1, // scaleFactor
            5,   // minNeighbors
            0,   // flags
            cv::Size(30, 30), // minSize
            cv::Size(),       // maxSize
            true              // outputRejectLevels true
        );


        std::cout << "Found " << faces.size() << " faces." << std::endl;
        // Loop through each detected face
        for(int j = 0; j < faces.size(); j++){
            // Print the confidence score (level weight) for the corresponding face
            score = calculateBlurScore(frame, faces[j]) * faces[j].area();
            std::cout << "Face " << j
                << " -> Score: " << score <<std::endl;
            // Here we filter the detection: if they're both not defined and small we filter out.
            if(score > min_score){
                filtered_faces.push_back(faces[j]);
            }
        }
        
        if(faces.size() > 0){
            std::cout<< "(Selected "<<filtered_faces.size()<<")"<<std::endl;
        }

        IOUs = get_label_IOUs(label_path, filtered_faces, frame.cols, frame.rows);
        for(int k = 0; k < IOUs.size(); k++){
            std::cout<<"Rect "<<k<<" IOUs: "<<IOUs[k]<<std::endl;
        }
        curr_MIOU = compute_MIOU(IOUs);
        if (curr_MIOU > best_MIOU){
            best_index = i;
            best_MIOU = curr_MIOU;
            best_detections = filtered_faces;
        }

        filtered_faces.clear();
    }

    std::cout<<std::endl<<"Best classifier is: "<<file_paths[best_index]<<std::endl; 
    std::cout<<"  with MIOU: "<<best_MIOU<<std::endl<<std::endl;

    final_classifier = f_cascades[best_index];

    std::cout<<"Detected rectangles position and size: "<<std::endl;
    printRectDetails(best_detections);
    std::cout<<std::endl;
    std::vector<cv::Rect> label_rects = compute_rectangles(label_path, frame.cols, frame.rows);
    std::cout<<"Label rectangles position and size: "<<std::endl;
    printRectDetails(label_rects);

    if(best_detections.size() == 0){
        exit(1);
    }
    get_label_IOUs(label_path, best_detections, frame.cols, frame.rows);
    
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
    for (int i = 0; i < best_detections.size(); i++){
        cv::rectangle(frame, best_detections[i], cv::Scalar(255, 0, 255), 4);      
    }
    // Draw the label box.
    for (int i = 0; i < label_rects.size(); i++){
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
   return best_detections;
}