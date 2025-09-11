#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include <string>
#include <fstream>
#include <map>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h> 
#include <thread>

#include "../include/utils.h"

std::vector<cv::Rect> vj_detect(cv::Mat frame, cv::CascadeClassifier f_cascade);
void draw_bbox(cv::Mat frame, std::vector<cv::Rect> faces, const std::vector<std::string>& labels);
void fifo_creation(const char* fifo_name);
void recognition_pipeline_call();

std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),     // red
        cv::Scalar(0, 255, 255),   // yellow
        cv::Scalar(0,0,0),         //black
        cv::Scalar(0, 255, 0),     // green
        cv::Scalar(255, 255, 255),   // white
        cv::Scalar(255, 0, 0),     // blue
        cv::Scalar(128, 0, 128),   // purple    
};

std::map<std::string, cv::Scalar> label_color = {
    {"angry", colors[0]},
    {"disgust", colors[1]},
    {"fear", colors[2]},
    {"happy", colors[3]},
    {"neutral", colors[4]},
    {"sad", colors[5]},
    {"surprise", colors[6]}
};

int main(int argc, char* argv[]) {

    // Call the python pipeline to classify the faces
    std::thread emotion_rec_thread = std::thread(recognition_pipeline_call);

    // Creation of the 2 fifo to communicate with the emotion_classifier pipeline
    fifo_creation("cpp_to_py.fifo");
    fifo_creation("py_to_cpp.fifo");

    // For per ciclare su tutte le immagini di test
    // con un flag che se attivato, non cicla su tutte , prende solo un immagine d auna cartella separata


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

    // -------------------------------------- FACE DETECTION --------------------------------------
    cv::CascadeClassifier face_cascade;

    // Load the cascades.
    if (!face_cascade.load("../classifiers/haarcascade_frontalface_alt.xml")){

        std::cout << "Error loading face cascade\n";
        return -1;
    };

    cv::Mat img = cv::imread(input_path);
    // Detect and save the faces in a specific folder.
    std::vector<cv::Rect> faces = vj_detect(img, face_cascade);
    // Folder path in which will be saved the images.
    std::string folder_path_cropped_imgs = "../cropped_imgs/";
    // Vector of cropped images and vector of bounding boxes.
    std::vector<cv::Mat>  cropped_imgs; 

    for (size_t i = 0; i < faces.size(); i++){
        // Cropping the detected faces.
        cv::Mat faceROI = img(faces[i]);
        cropped_imgs.push_back(faceROI.clone());
        // Saving the cropped images.
        cv::imwrite(folder_path_cropped_imgs + "cut_" + std::to_string(i)+".png", cropped_imgs[i]);        
    }
    
    // ------------------------------------ EMOTION RECOGNITION ------------------------------------
    // Signal (to Python)
    std::cout<<"Prima di python\n";   
    std::ofstream to_server("cpp_to_py.fifo");
    if(faces.empty()){
        std::cout <<"No faces are detected, the program terminates\n";
        // Singal (to Python) for closing its pipeline 
        to_server << "exit" << std::endl; 
        // Wait the thread ends
        emotion_rec_thread.join();
        return 1;
    }
    to_server << "Required Emotion recognition" << std::endl;
    to_server.close();

    // **** Python program to detect *****

    // Waiting (from Python)
    std::cout << "In attesa della risposta da Python...\n";
    std::ifstream from_server("py_to_cpp.fifo");
    std::string line;
    std::vector<std::string> labels;

    // Read all the output stream
    while (std::getline(from_server, line)) {
        
        std::cout << "Python output: " << line << std::endl;
        labels.push_back(line);
    } 
    from_server.close();
    
    draw_bbox(img, faces, labels);
    namedWindow("Window", cv::WINDOW_NORMAL);
    cv::imshow("Window", img);
    cv::waitKey(0);

    // Wait the thread ends
    emotion_rec_thread.join();
    return 0;
}



    void draw_bbox(cv::Mat frame, std::vector<cv::Rect> faces, const std::vector<std::string>& labels){

        // Draw the box over the detection.
        for (size_t i = 0; i < faces.size(); i++){

            cv::Scalar color = cv::Scalar(255,255,255); // default = white
            if (label_color.find(labels[i]) != label_color.end()) {
                color = label_color[labels[i]];
            }

            // Draw the Bounding box
            cv::rectangle(frame, faces[i], color, 4); 
            // Draw the corrispective label on the BBox
            cv::putText(frame, labels[i],
                cv::Point(faces[i].x, faces[i].y - 5),             // 5 pixels above the top-left
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,                              // font scale
                color,
                1,
                cv::LINE_AA);     
        }
    }
    // Detection function using the ViolaJones algorithm.
    std::vector<cv::Rect> vj_detect(cv::Mat frame , cv::CascadeClassifier f_cascade){

    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    f_cascade.detectMultiScale(frame_gray, faces);

    return faces;   

    }

    // Function to check if a filesystem file .fifo is already present otherwise it create it.
    void fifo_creation(const char* fifo_name) {

        if (access(fifo_name, F_OK) != 0) { // If fifo doesn't exist.
            if (mkfifo(fifo_name, 0666) == -1) { // If creation doesn't work
                std::cerr << "Error in the creation of "
                        << fifo_name << ": "
                        << std::strerror(errno) << std::endl;
            } else {
                std::cout << "Creation of the fifo: " << fifo_name << std::endl;
            }
        } else {
            std::cout << "fifo already exist" << std::endl;
        }
    }

    void recognition_pipeline_call(){
        int ret = system("python3 ../python/emotion_classifier.py 2>/dev/null");

    }


// // Show the images detected.
//    cv::imshow("Window", frame);
//    cv::waitKey(0);
//
//    for (size_t i = 0; i < cropped_imgs.size(); i++){
//
//        std::string window_name = "Window " + std::to_string(i);
//        cv::imshow(window_name, cropped_imgs[i]);
//    }
//
//    cv::waitKey(0);

