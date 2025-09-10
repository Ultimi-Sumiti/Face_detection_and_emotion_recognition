#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include <string>
#include <fstream>


#include "../include/utils.h"

std::vector<cv::Rect> vj_detect(cv::Mat frame, cv::CascadeClassifier f_cascade);
void draw_bbox(cv::Mat frame, std::vector<cv::Rect> faces, const std::vector<std::string>& labels);

int main(int argc, char* argv[]) {

    // Call the python pipeline to classify the faces
    int ret = system("python3 ../python/emotion_classifier.py 2>/dev/null &");
    // TODO gestione chiamata in background del file python (stiamo usando parallelismo)

    /* Presetting Crea i .fifo e li gestisce

    if (mkfifo("cpp_to_py.fifo", 0666) == -1) {
        perror("Errore nella creazione della FIFO");
    }
    if (mkfifo("py_to_cpp.fifo", 0666) == -1) {
        perror("Errore nella creazione della FIFO");
    }
    
    */
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
    
    // ------------------------------------ EMOTION RECOGNITION ------------------------------------
    // Signal (to Python)    
    std::ofstream to_server("cpp_to_py.fifo");
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
    cv::imshow("Window", img);
    cv::waitKey(0);
    return 0;
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
    return faces;   
}

void draw_bbox(cv::Mat frame, std::vector<cv::Rect> faces, const std::vector<std::string>& labels){

    // Draw the box over the detection.
    for (size_t i = 0; i < faces.size(); i++){
        // Draw the Bounding box
        cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 255), 4); 
        // Draw the corrispective label on the BBox
        cv::putText(frame, labels[i],
            cv::Point(faces[i].x, faces[i].y - 5),             // 5 pixels above the top-left
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,                              // font scale
            cv::Scalar(255, 0, 255),
            1,
            cv::LINE_AA);     
    }
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